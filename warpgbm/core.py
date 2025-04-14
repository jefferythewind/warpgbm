import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from warpgbm.cuda import node_kernel
from tqdm import tqdm

histogram_kernels = {
    'hist1': node_kernel.compute_histogram,
    'hist2': node_kernel.compute_histogram2,
    'hist3': node_kernel.compute_histogram3
}

class WarpGBM(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20,
        min_split_gain=0.0,
        verbosity=True,
        histogram_computer='hist3',
        threads_per_block=32,
        rows_per_thread=4,
        device = 'cuda',
        split_type = 'v2',
        L2_reg = 0.000001,
    ):
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.forest = None
        self.bin_edges = None  # shape: [num_features, num_bins-1] if using quantile binning
        self.base_prediction = None
        self.unique_eras = None
        self.device = device
        self.root_gradient_histogram = None
        self.root_hessian_histogram = None
        self.gradients = None
        self.root_node_indices = None
        self.bin_indices = None
        self.Y_gpu = None
        self.num_features = None
        self.num_samples = None
        self.out_feature = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.out_bin = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.best_gain = torch.tensor([-float('inf')], dtype=torch.float32, device=self.device)
        self.best_feature = torch.tensor([-1], dtype=torch.int32, device=self.device)
        self.best_bin = torch.tensor([-1], dtype=torch.int32, device=self.device)
        self.compute_histogram = histogram_kernels[histogram_computer]
        self.threads_per_block = threads_per_block
        self.rows_per_thread = rows_per_thread
        self.split_type = split_type
        self.L2_reg = L2_reg


    def fit(self, X, y, era_id=None):
        if era_id is None:
            era_id = np.ones(X.shape[0], dtype='int32')
        self.bin_indices, era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = self.preprocess_gpu_data(X, y, era_id)
        self.num_samples, self.num_features = X.shape
        self.gradients = torch.zeros_like(self.Y_gpu)
        self.root_node_indices = torch.arange(self.num_samples, device=self.device)
        self.base_prediction = self.Y_gpu.mean().item()
        self.gradients += self.base_prediction
        self.split_gains = torch.zeros((self.num_features, self.num_bins - 1), device=self.device)
        self.forest = self.grow_forest()
        return self

    def compute_quantile_bins(self, X, num_bins):
        quantiles = torch.linspace(0, 1, num_bins + 1)[1:-1]  # exclude 0% and 100%
        bin_edges = torch.quantile(X, quantiles, dim=0)       # shape: [B-1, F]
        return bin_edges.T  # shape: [F, B-1]
    
    def preprocess_gpu_data(self, X_np, Y_np, era_id_np):
        self.num_samples, self.num_features = X_np.shape
        Y_gpu = torch.from_numpy(Y_np).type(torch.float32).to(self.device)
        era_id_gpu = torch.from_numpy(era_id_np).type(torch.int32).to(self.device)
        is_integer_type = np.issubdtype(X_np.dtype, np.integer)
        if is_integer_type:
            max_vals = X_np.max(axis=0)
            if np.all(max_vals < self.num_bins):
                print("Detected pre-binned integer input — skipping quantile binning.")
                bin_indices = torch.from_numpy(X_np).to(self.device).contiguous().to(torch.int8)
    
                # We'll store None or an empty tensor in self.bin_edges
                # to indicate that we skip binning at predict-time
                bin_edges = torch.arange(1, self.num_bins, dtype=torch.float32).repeat(self.num_features, 1)
                bin_edges = bin_edges.to(self.device)
                unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
                return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu
            else:
                print("Integer input detected, but values exceed num_bins — falling back to quantile binning.")
    
        print("Performing quantile binning on CPU...")
        X_cpu = torch.from_numpy(X_np).type(torch.float32)  # CPU tensor
        bin_edges_cpu = self.compute_quantile_bins(X_cpu, self.num_bins).type(torch.float32).contiguous()
        bin_indices_cpu = torch.empty((self.num_samples, self.num_features), dtype=torch.int8)
        for f in range(self.num_features):
            bin_indices_cpu[:, f] = torch.bucketize(X_cpu[:, f], bin_edges_cpu[f], right=False).type(torch.int8)
        bin_indices = bin_indices_cpu.to(self.device).contiguous()
        bin_edges = bin_edges_cpu.to(self.device)
        unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
        return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu

    def compute_histograms(self, bin_indices_sub, gradients):
        grad_hist = torch.zeros((self.num_features, self.num_bins), device=self.device, dtype=torch.float32)
        hess_hist = torch.zeros((self.num_features, self.num_bins), device=self.device, dtype=torch.float32)
    
        self.compute_histogram(
            bin_indices_sub,
            gradients,
            grad_hist,
            hess_hist,
            self.num_bins,
            self.threads_per_block,
            self.rows_per_thread
        )
        return grad_hist, hess_hist

    def find_best_split(self, gradient_histogram, hessian_histogram):
        node_kernel.compute_split(
            gradient_histogram.contiguous(),
            hessian_histogram.contiguous(),
            self.num_features,
            self.num_bins,
            self.min_split_gain,  # 
            self.min_child_weight,  #
            self.L2_reg, # eps
            self.out_feature,
            self.out_bin
        )
        
        f = int(self.out_feature[0])
        b = int(self.out_bin[0])
        return (f, b)
    
    def find_best_split_v2(self, G, H):
        F, B = G.shape

        G_total = torch.sum(G, dim=1, keepdim=True)
        H_total = torch.sum(H, dim=1, keepdim=True)

        G_L = torch.cumsum(G, dim=1)[:, :-1]
        H_L = torch.cumsum(H, dim=1)[:, :-1]
        G_R = G_total - G_L
        H_R = H_total - H_L

        gain = (G_L ** 2) / (H_L + self.L2_reg) + (G_R ** 2) / (H_R + self.L2_reg)
        invalid_mask = (H_L < self.min_child_weight) | (H_R < self.min_child_weight) | (gain < self.min_split_gain)

        # Mask invalid gains
        gain[invalid_mask] = -float('inf')

        # Find the best valid split
        best_flat_idx = torch.argmax(gain)
        best_gain = gain.view(-1)[best_flat_idx]

        if best_gain == -float('inf'):  # No valid splits
            return -1, -1

        best_feat = best_flat_idx // (B - 1)
        best_bin = best_flat_idx % (B - 1)
        return int(best_feat), int(best_bin)
    
    def grow_tree(self, gradient_histogram, hessian_histogram, node_indices, depth):
        if depth == self.max_depth:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": node_indices.numel()}
    
        parent_size = node_indices.numel()
        if self.split_type == 'v1':
            best_feature, best_bin = self.find_best_split(gradient_histogram, hessian_histogram)
        elif self.split_type == 'v2':
            best_feature, best_bin = self.find_best_split_v2(gradient_histogram, hessian_histogram)
    
        if best_feature == -1:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": parent_size}
    
        split_mask = (self.bin_indices[node_indices, best_feature] <= best_bin)
        left_indices = node_indices[split_mask]
        right_indices = node_indices[~split_mask]

        left_size = left_indices.numel()
        right_size = right_indices.numel()

        if left_size == 0 or right_size == 0:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": parent_size}

        if left_size <= right_size:
            grad_hist_left, hess_hist_left = self.compute_histograms( self.bin_indices[left_indices], self.residual[left_indices] )
            grad_hist_right = gradient_histogram - grad_hist_left
            hess_hist_right = hessian_histogram - hess_hist_left
        else:
            grad_hist_right, hess_hist_right = self.compute_histograms( self.bin_indices[right_indices], self.residual[right_indices] )
            grad_hist_left = gradient_histogram - grad_hist_right
            hess_hist_left = hessian_histogram - hess_hist_right

        new_depth = depth + 1
        left_child = self.grow_tree(grad_hist_left, hess_hist_left, left_indices, new_depth)
        right_child = self.grow_tree(grad_hist_right, hess_hist_right, right_indices, new_depth)
    
        return { "feature": best_feature, "bin": best_bin, "left": left_child, "right": right_child }

    def grow_forest(self):
        forest = [{} for _ in range(self.n_estimators)]
        self.training_loss = []
    
        for i in range(self.n_estimators):
            self.residual = self.Y_gpu - self.gradients
    
            self.root_gradient_histogram, self.root_hessian_histogram = \
                self.compute_histograms(self.bin_indices, self.residual)
    
            tree = self.grow_tree(
                self.root_gradient_histogram,
                self.root_hessian_histogram,
                self.root_node_indices,
                depth=0
            )
            forest[i] = tree
            loss = ((self.Y_gpu - self.gradients) ** 2).mean().item()
            self.training_loss.append(loss)
            # print(f"🌲 Tree {i+1}/{self.n_estimators} - MSE: {loss:.6f}")
    
        print("Finished training forest.")
        return forest

    def predict(self, X_np, era_id_np=None):
        is_integer_type = np.issubdtype(X_np.dtype, np.integer)
        if is_integer_type:
            max_vals = X_np.max(axis=0)
            if np.all(max_vals < self.num_bins):
                bin_indices = X_np.astype(np.int8)
                return self.predict_data(bin_indices)

        X_cpu = torch.from_numpy(X_np).type(torch.float32)  # CPU tensor
        bin_indices_cpu = torch.empty((X_np.shape[0], X_np.shape[1]), dtype=torch.int8)
        bin_edges_cpu = self.bin_edges.to('cpu')
        for f in range(self.num_features):
            bin_indices_cpu[:, f] = torch.bucketize(X_cpu[:, f], bin_edges_cpu[f], right=False).type(torch.int8)

        bin_indices = bin_indices_cpu.numpy()  # Use CPU numpy array for predict_data
        return self.predict_data(bin_indices)

    @staticmethod
    def process_node(node, data_idx, bin_indices):
        while 'leaf_value' not in node:
            if bin_indices[data_idx, node['feature']] <= node['bin']:
                node = node['left']
            else:
                node = node['right']
        return node['leaf_value']

    def predict_data(self, bin_indices):
        n = bin_indices.shape[0]
        preds = np.zeros(n)
        proc = self.process_node  # local var for speed
        lr = self.learning_rate
        base = self.base_prediction
        forest = self.forest
    
        for i in tqdm( range(n) ):
            preds[i] = base + lr * np.sum([proc( tree, i, bin_indices ) for tree in forest])
        return preds
