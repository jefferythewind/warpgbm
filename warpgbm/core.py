import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from warpgbm.cuda import node_kernel
from tqdm import tqdm
from typing import Tuple
from torch import Tensor

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
        # histogram_computer='hist3',
        threads_per_block=64,
        rows_per_thread=4,
        L2_reg=1e-6,
        L1_reg=0.0,
        device='cuda'
    ):
        # Validate arguments
        self._validate_hyperparams(
            num_bins=num_bins,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            # histogram_computer=histogram_computer,
            threads_per_block=threads_per_block,
            rows_per_thread=rows_per_thread,
            L2_reg=L2_reg,
            L1_reg=L1_reg
        )

        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.forest = None
        self.bin_edges = None
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
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.threads_per_block = threads_per_block
        self.rows_per_thread = rows_per_thread
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg

    def _validate_hyperparams(self, **kwargs):
        # Type checks
        int_params = [
            "num_bins", "max_depth", "n_estimators", "min_child_weight",
            "threads_per_block", "rows_per_thread"
        ]
        float_params = [
            "learning_rate", "min_split_gain", "L2_reg", "L1_reg"
        ]

        for param in int_params:
            if not isinstance(kwargs[param], (int, np.integer)):
                raise TypeError(f"{param} must be an integer, got {type(kwargs[param])}.")


        for param in float_params:
            if not isinstance(kwargs[param], (float, int)):  # Accept ints as valid floats
                raise TypeError(f"{param} must be a float, got {type(kwargs[param])}.")
            
        if not ( 2 <= kwargs["num_bins"] <= 127 ):
            raise ValueError("num_bins must be between 2 and 127 inclusive.")
        if kwargs["max_depth"] < 1:
            raise ValueError("max_depth must be at least 1.")
        if not (0.0 < kwargs["learning_rate"] <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0].")
        if kwargs["n_estimators"] <= 0:
            raise ValueError("n_estimators must be positive.")
        if kwargs["min_child_weight"] < 1:
            raise ValueError("min_child_weight must be a positive integer.")
        if kwargs["min_split_gain"] < 0:
            raise ValueError("min_split_gain must be non-negative.")
        if kwargs["threads_per_block"] <= 0 or kwargs["threads_per_block"] % 32 != 0:
            raise ValueError("threads_per_block should be a positive multiple of 32 (warp size).")
        if not ( 1 <= kwargs["rows_per_thread"] <= 16 ):
            raise ValueError("rows_per_thread must be positive between 1 and 16 inclusive.")
        if kwargs["L2_reg"] < 0 or kwargs["L1_reg"] < 0:
            raise ValueError("L2_reg and L1_reg must be non-negative.")

    def fit(self, X, y, era_id=None):
        if era_id is None:
            era_id = np.ones(X.shape[0], dtype='int32')
        self.bin_indices, era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = self.preprocess_gpu_data(X, y, era_id)
        self.num_samples, self.num_features = X.shape
        self.gradients = torch.zeros_like(self.Y_gpu)
        self.root_node_indices = torch.arange(self.num_samples, device=self.device)
        self.base_prediction = self.Y_gpu.mean().item()
        self.gradients += self.base_prediction

        # === Initialization for level-wise growth ===
        self.max_nodes = 2 ** self.max_depth

        self.grad_hists = torch.zeros((self.max_nodes, self.num_features, self.num_bins), device=self.device)
        self.hess_hists = torch.zeros_like(self.grad_hists)

        self.split_gains = torch.full((self.max_nodes, self.num_features), -float("inf"), device=self.device)
        self.split_bins = torch.full((self.max_nodes, self.num_features), -1, dtype=torch.int32, device=self.device)

        self.sample_to_node = torch.zeros(self.num_samples, dtype=torch.int32, device=self.device)

        self.best_split_feature = torch.full((self.max_nodes,), -1, dtype=torch.int32, device=self.device)
        self.best_split_bin = torch.full((self.max_nodes,), -1, dtype=torch.int32, device=self.device)


        with torch.no_grad():
            self.forest = self.grow_forest()
        return self
    
    def preprocess_gpu_data(self, X_np, Y_np, era_id_np):
        with torch.no_grad():
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
        
            bin_indices = torch.empty((self.num_samples, self.num_features), dtype=torch.int8, device='cuda')
            bin_edges = torch.empty((self.num_features, self.num_bins - 1), dtype=torch.float32, device='cuda')

            X_np = torch.from_numpy(X_np).to(torch.float32).pin_memory()

            for f in range(self.num_features):
                X_f = X_np[:, f].to('cuda', non_blocking=True)
                quantiles = torch.linspace(0, 1, self.num_bins + 1, device='cuda', dtype=X_f.dtype)[1:-1]
                bin_edges_f = torch.quantile(X_f, quantiles, dim=0).contiguous()  # shape: [B-1] for 1D input
                bin_indices_f = bin_indices[:, f].contiguous()  # view into output
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices_f)
                bin_indices[:,f] = bin_indices_f
                bin_edges[f,:] = bin_edges_f

            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
            return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu
    
    def grow_forest(self):
        forest = []
      
        def node_index(depth, node_id):
          return 2 ** depth - 1 + node_id
          
        for est in tqdm(range(self.n_estimators)):
            self.residual = self.Y_gpu - self.gradients
            tree_nodes = [{} for _ in range(2 ** (self.max_depth + 1))]
            self.sample_to_node.zero_()

            depth_nodes = [0]
            for depth in range(self.max_depth):
                num_nodes = 2**depth
                self.grad_hists[:num_nodes].zero_()
                self.hess_hists[:num_nodes].zero_()
                self.split_gains[:num_nodes].fill_(-float("inf"))
                self.split_bins[:num_nodes].fill_(-1)
                self.best_split_feature[:num_nodes].fill_(-1)
                self.best_split_bin[:num_nodes].fill_(-1)

                depth_tensor = torch.tensor(depth_nodes, device=self.device, dtype=torch.int32)

                if depth == 0:
                    # At root, there's only one node and no subtraction.
                    right_children_mask = torch.ones_like(self.sample_to_node, dtype=torch.bool)
                else:
                    right_children = depth_tensor[2**depth:]  # these are the right children
                    right_children_mask = torch.isin(self.sample_to_node, right_children)

                node_kernel.build_histograms(
                    self.bin_indices[right_children_mask],
                    self.sample_to_node[right_children_mask], 
                    self.residual[right_children_mask],
                    self.grad_hists,
                    self.hess_hists,
                    self.threads_per_block,
                    self.rows_per_thread
                )

                # Subtract right from parent to get left
                if depth > 0:
                    for node_id in depth_nodes[:2**depth]:
                        right_id = 2**depth + node_id
                        self.grad_hists[node_id] -= self.grad_hists[right_id]
                        self.hess_hists[node_id] -= self.hess_hists[right_id]

                node_kernel.find_splits_for_level(
                    depth_tensor,
                    self.grad_hists,
                    self.hess_hists,
                    self.min_split_gain,
                    self.min_child_weight,
                    self.L2_reg,
                    self.split_gains,
                    self.split_bins,
                    self.threads_per_block
                )

                best_features = torch.argmax(self.split_gains[depth_tensor], dim=1)
                best_bins = self.split_bins[depth_tensor, best_features]

                self.best_split_feature[depth_tensor] = best_features.to(torch.int32)
                self.best_split_bin[depth_tensor] = best_bins.to(torch.int32)

                new_depth_nodes = []
                for node_id in depth_nodes:
                    if self.best_split_bin[node_id] > -1:
                        feat = self.best_split_feature[ node_id ].item()
                        thresh = self.best_split_bin[ node_id ].item()
                        new_left_node = node_id
                        new_right_node = 2**depth + node_id
                        tree_node = { 
                            "node_id": node_index(depth, node_id),
                            "left_id": node_index(depth + 1, new_left_node),
                            "right_id": node_index(depth + 1, new_right_node),
                            "best_feature": feat,
                            "split_bin": thresh,
                            "depth": depth
                        }
                        new_depth_nodes += [new_left_node, new_right_node]
                        
                        left_mask = (self.sample_to_node == node_id) & (self.bin_indices[:, feat] <= thresh)
                        right_mask = (self.sample_to_node == node_id) & (self.bin_indices[:, feat] > thresh)
                
                        self.sample_to_node[left_mask] = new_left_node
                        self.sample_to_node[right_mask] = new_right_node
                    else:
                        node_mask = self.sample_to_node == node_id
                        leaf_val = torch.mean( self.residual[ node_mask ] )
                        tree_node = { 
                            "node_id": node_index(depth, node_id),
                            "leaf_value": leaf_val,
                            "depth": depth
                        }
                        #update gradients
                        self.gradients[node_mask] += self.learning_rate * leaf_val
                        self.sample_to_node[node_mask] = -1
                    tree_nodes[ node_index(depth, node_id) ] = tree_node

                depth_nodes = new_depth_nodes
                if len(depth_nodes) == 0:
                    break
            if len(depth_nodes) > 0:
                depth = depth + 1
                for node_id in depth_nodes:
                    node_mask = self.sample_to_node == node_id
                    leaf_val = torch.mean( self.residual[ node_mask ] )
                    tree_node = { 
                        "node_id": node_index(depth, node_id),
                        "leaf_value": leaf_val,
                        "depth": depth
                    }
                    #update gradients
                    self.gradients[node_mask] += self.learning_rate * leaf_val
                    tree_nodes[ node_index(depth, node_id) ] = tree_node
            forest.append(tree_nodes)
        final_corr = torch.corrcoef(torch.stack([self.gradients, self.Y_gpu]))[0, 1]
        print(f"[final model] In-sample correlation: {final_corr.item():.5f}")
        return forest

    def predict(self, X_np, chunk_size=50000):
       #TODO
       return
