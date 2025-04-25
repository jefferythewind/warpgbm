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
        print(self.bin_indices)
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
            # print(f"\n--- Growing estimator {est} ---")
            self.residual = self.Y_gpu - self.gradients
            tree_nodes = [{} for _ in range(2 ** (self.max_depth + 1))]
            self.sample_to_node.zero_()

            self.grad_hists.zero_()
            self.hess_hists.zero_()
            self.split_gains.fill_(-float("inf"))
            self.split_bins.fill_(-1)
            self.best_split_feature.fill_(-1)
            self.best_split_bin.fill_(-1)

            depth_nodes = [0]
            for depth in range(self.max_depth):
                # print(f"\n[Depth {depth}] depth_nodes: {depth_nodes}")
                num_nodes = 2**depth

                depth_tensor = torch.tensor(depth_nodes, device=self.device, dtype=torch.int32)
                half_shape = max(int(depth_tensor.shape[0]/2), 1)

                if depth > 0:
                    self.grad_hists[int( num_nodes/2 ):num_nodes].zero_()
                    self.hess_hists[int( num_nodes/2 ):num_nodes].zero_()
                self.split_gains[:num_nodes].fill_(-float("inf"))
                self.split_bins[:num_nodes].fill_(-1)
                self.best_split_feature[:num_nodes].fill_(-1)
                self.best_split_bin[:num_nodes].fill_(-1)

                if depth == 0:
                    right_children_mask = torch.ones_like(self.sample_to_node, dtype=torch.bool)
                else:
                    right_children = depth_tensor[half_shape:]
                    right_children_mask = torch.isin(self.sample_to_node, right_children)

                # print(f"  Building histograms...")
                node_kernel.build_histograms(
                    self.bin_indices[right_children_mask],
                    self.sample_to_node[right_children_mask], 
                    self.residual[right_children_mask],
                    self.grad_hists,
                    self.hess_hists,
                    self.threads_per_block,
                    self.rows_per_thread
                )

                if depth > 0:
                    for i, node_id in enumerate( depth_nodes[:half_shape] ):
                        right_id = depth_nodes[half_shape+i]
                        # print(node_id, right_id)
                        self.grad_hists[node_id] = self.grad_hists[node_id] - self.grad_hists[right_id]
                        self.hess_hists[node_id] = self.hess_hists[node_id] - self.hess_hists[right_id]


                # print(f"  Histograms (grad) at depth {depth}:\n", self.grad_hists[:num_nodes])
                # print(f"  Histograms (hess) at depth {depth}:\n", self.hess_hists[:num_nodes])

                # print(f"  Finding best splits...")
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
                # print(f"  Split gains: {self.split_gains[depth_tensor]}")
                # print(f"  Split bins: {self.split_bins[depth_tensor]}")

                best_features = torch.argmax(self.split_gains[depth_tensor], dim=1)
                best_bins = self.split_bins[depth_tensor, best_features]

                self.best_split_feature[depth_tensor] = best_features.to(torch.int32)
                self.best_split_bin[depth_tensor] = best_bins.to(torch.int32)

                # print(f"  Best  features {self.best_split_feature[depth_tensor]}")
                # print(f"  Bin {self.best_split_bin[depth_tensor]}")

                new_right_nodes = []
                new_left_nodes = []
                for node_id in depth_nodes:
                    if self.best_split_bin[node_id] > -1:
                        feat = self.best_split_feature[node_id].item()
                        thresh = self.best_split_bin[node_id].item()
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
                        new_right_nodes.append(new_right_node)
                        new_left_nodes.append(new_left_node)

                        left_mask = (self.sample_to_node == node_id) & (self.bin_indices[:, feat] <= thresh)
                        right_mask = (self.sample_to_node == node_id) & (self.bin_indices[:, feat] > thresh)

                        self.sample_to_node[left_mask] = new_left_node
                        self.sample_to_node[right_mask] = new_right_node
                    else:
                        node_mask = self.sample_to_node == node_id
                        leaf_val = torch.mean(self.residual[node_mask])
                        tree_node = { 
                            "node_id": node_index(depth, node_id),
                            "leaf_value": leaf_val,
                            "depth": depth
                        }
                        self.gradients[node_mask] += self.learning_rate * leaf_val
                        self.sample_to_node[node_mask] = -1
                    tree_nodes[node_index(depth, node_id)] = tree_node

                depth_nodes = new_left_nodes + new_right_nodes
                # depth_nodes = self.process_depth(depth, depth_tensor, tree_nodes)
                # print(f"  New depth_nodes: {depth_nodes}")
                if len(depth_nodes) == 0:
                    break

            if len(depth_nodes) > 0:
                depth = depth + 1
                for node_id in depth_nodes:
                    node_mask = self.sample_to_node == node_id
                    leaf_val = torch.mean(self.residual[node_mask])
                    tree_node = { 
                        "node_id": node_index(depth, node_id),
                        "leaf_value": leaf_val,
                        "depth": depth
                    }
                    self.gradients[node_mask] += self.learning_rate * leaf_val
                    tree_nodes[node_index(depth, node_id)] = tree_node
            forest.append(tree_nodes)
        final_corr = torch.corrcoef(torch.stack([self.gradients, self.Y_gpu]))[0, 1]
        print(f"[final model] In-sample correlation: {final_corr.item():.5f}")
        return forest

    def predict(self, X_np, chunk_size=50000):
        X_tensor = torch.from_numpy(X_np).to(torch.float32).pin_memory()
        num_samples = X_tensor.size(0)
        bin_indices = torch.zeros((num_samples, self.num_features), dtype=torch.int8, device=self.device)

        with torch.no_grad():
            for f in range(self.num_features):
                # print('binning features')
                X_f = X_tensor[:, f].to(self.device, non_blocking=True)
                bin_edges_f = self.bin_edges[f]
                # print(bin_edges_f)
                bin_indices_f = bin_indices[:, f].contiguous()
                # print(X_f)
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices_f)
                bin_indices[:,f] = bin_indices_f
        
        # print(bin_indices)

        tree_tensor = torch.stack([self.flatten_tree(tree, max_nodes=2**(self.max_depth + 1)) for tree in self.forest]).to(self.device)

        # print("tree tensor")
        # print(tree_tensor)

        out = torch.zeros(num_samples, device=self.device)

        node_kernel.predict_forest(
            bin_indices.contiguous(),
            tree_tensor.contiguous(),
            self.learning_rate,
            out
        )

        return out.cpu().numpy()

    def flatten_tree(self, tree, max_nodes):
        flat = torch.full((max_nodes, 6), float('nan'), dtype=torch.float32)
        for node in tree:
            if not node:
                continue
            i = node['node_id']
            if 'leaf_value' in node:
                flat[i, 4] = 1.0  # leaf flag
                flat[i, 5] = node['leaf_value'].item()
            else:
                flat[i, 0] = node['best_feature']
                flat[i, 1] = node['split_bin']
                flat[i, 2] = node['left_id']
                flat[i, 3] = node['right_id']
                flat[i, 4] = 0.0  # internal node
        return flat

    def process_depth(self, 
        depth, depth_nodes, tree_nodes
    ):
        """
        Vectorized processing of tree nodes at a given depth:
        - Splits samples by best_split_feature/bin thresholds
        - Routes samples to children or marks leaves
        - Updates gradients for leaves
        - Builds new depth_nodes list and tree_nodes entries
        """
        def node_index(depth, node_id):
          return 2 ** depth - 1 + node_id

        # Device and constants
        device = self.device
        N = self.sample_to_node.size(0)
        
        # Convert depth_nodes to a tensor
        dn = torch.tensor(depth_nodes, dtype=torch.int64, device=device)
        
        # Gather per-node split parameters
        split_feat   = self.best_split_feature  # [max_nodes]
        split_thresh = self.best_split_bin      # [max_nodes]
        
        # Gather per-sample split info
        stn         = self.sample_to_node       # [N]
        feat_samp   = split_feat[stn]           # [N]
        thresh_samp = split_thresh[stn]         # [N]
        bin_vals    = self.bin_indices[
            :, feat_samp
        ]                                       # [N]
        
        # Determine splitable samples and right-going samples
        splitable = thresh_samp >= 0
        go_right  = splitable & (bin_vals > thresh_samp)
        
        # Compute new node assignments
        new_stn = stn + go_right.long() * (1 << depth)
        
        # Handle leaves: mean residual per leaf-node
        leaf_mask    = ~splitable
        old_leaf_ids = stn[leaf_mask]
        resid_leaf   = self.residual[leaf_mask]
        
        # Unique leaf node IDs and scatter-based mean
        uniq_leaf, inv = torch.unique(old_leaf_ids, return_inverse=True)
        num_leaf_nodes = uniq_leaf.size(0)
        sum_vals = torch.zeros(num_leaf_nodes, device=device)
        count    = torch.zeros(num_leaf_nodes, device=device)
        
        sum_vals = sum_vals.scatter_add(0, inv, resid_leaf)
        count    = count.scatter_add(0, inv, torch.ones_like(inv, dtype=count.dtype))
        
        leaf_vals_per_node   = sum_vals / count
        leaf_vals_per_sample = leaf_vals_per_node[inv]
        
        # Update gradients and mark leaves
        self.gradients[leaf_mask] += self.learning_rate * leaf_vals_per_sample
        new_stn[leaf_mask] = -1
        
        # Commit new assignments
        self.sample_to_node = new_stn
        
        # Build new depth_nodes and tree_nodes entries
        node_mask   = split_thresh[dn] >= 0
        left_nodes  = dn[node_mask]
        right_nodes = left_nodes + (1 << depth)
        
        # Update tree_nodes (still small overhead)
        for node_id in left_nodes.tolist():
            nd = node_index(depth, node_id)
            tree_nodes[nd] = {
                "node_id": nd,
                "left_id": node_index(depth+1, node_id),
                "right_id": node_index(depth+1, node_id + (1 << depth)),
                "best_feature": split_feat[node_id].item(),
                "split_bin": split_thresh[node_id].item(),
                "depth": depth
            }
        
        # Leaf entries
        for i, node_id in enumerate(uniq_leaf.tolist()):
            nd      = node_index(depth, node_id)
            leaf_val = leaf_vals_per_node[i].item()
            tree_nodes[nd] = {
                "node_id": nd,
                "leaf_value": leaf_val,
                "depth": depth
            }
        
        # Return updated depth_nodes for next depth
        return left_nodes.tolist() + right_nodes.tolist()