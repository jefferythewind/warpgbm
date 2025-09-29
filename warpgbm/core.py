import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_log_error
from warpgbm.cuda import node_kernel
from warpgbm.metrics import rmsle_torch
from tqdm import tqdm
from typing import Tuple
from torch import Tensor
import gc

class WarpGBM(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20,
        min_split_gain=0.0,
        threads_per_block=64,
        rows_per_thread=4,
        L2_reg=1e-6,
        L1_reg=0.0,
        device="cuda",
        colsample_bytree=1.0,
    ):
        # Validate arguments
        self._validate_hyperparams(
            num_bins=num_bins,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            threads_per_block=threads_per_block,
            rows_per_thread=rows_per_thread,
            L2_reg=L2_reg,
            L1_reg=L1_reg,
            colsample_bytree=colsample_bytree,
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
        self.best_bin = torch.tensor([-1], dtype=torch.int32, device=self.device)
        self.threads_per_block = threads_per_block
        self.rows_per_thread = rows_per_thread
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.forest = [{} for _ in range(self.n_estimators)]
        self.colsample_bytree = colsample_bytree

    def _validate_hyperparams(self, **kwargs):
        # Type checks
        int_params = [
            "num_bins",
            "max_depth",
            "n_estimators",
            "min_child_weight",
            "threads_per_block",
            "rows_per_thread",
        ]
        float_params = [
            "learning_rate",
            "min_split_gain",
            "L2_reg",
            "L1_reg",
            "colsample_bytree",
        ]

        for param in int_params:
            if not isinstance(kwargs[param], int):
                raise TypeError(
                    f"{param} must be an integer, got {type(kwargs[param])}."
                )

        for param in float_params:
            if not isinstance(
                kwargs[param], (float, int)
            ):  # Accept ints as valid floats
                raise TypeError(f"{param} must be a float, got {type(kwargs[param])}.")

        if not (2 <= kwargs["num_bins"] <= 127):
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
            raise ValueError(
                "threads_per_block should be a positive multiple of 32 (warp size)."
            )
        if not (1 <= kwargs["rows_per_thread"] <= 16):
            raise ValueError(
                "rows_per_thread must be positive between 1 and 16 inclusive."
            )
        if kwargs["L2_reg"] < 0 or kwargs["L1_reg"] < 0:
            raise ValueError("L2_reg and L1_reg must be non-negative.")
        if kwargs["colsample_bytree"] <= 0 or kwargs["colsample_bytree"] > 1:
            raise ValueError(
                f"Invalid colsample_bytree: {kwargs['colsample_bytree']}. Must be a float value > 0 and <= 1."
            )

    def validate_fit_params(
        self, X, y, era_id, X_eval, y_eval, eval_every_n_trees, early_stopping_rounds, eval_metric
    ):
        # ─── Required: X and y ───
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows. Got {X.shape[0]} and {y.shape[0]}."
            )

        # ─── Optional: era_id ───
        if era_id is not None:
            if not isinstance(era_id, np.ndarray):
                raise TypeError("era_id must be a numpy array.")
            if era_id.ndim != 1:
                raise ValueError(
                    f"era_id must be 1-dimensional, got shape {era_id.shape}"
                )
            if len(era_id) != len(y):
                raise ValueError(
                    f"era_id must have same length as y. Got {len(era_id)} and {len(y)}."
                )

        # ─── Optional: Eval Set ───
        eval_args = [X_eval, y_eval, eval_every_n_trees]
        if any(arg is not None for arg in eval_args):
            # Require all of them
            if X_eval is None or y_eval is None or eval_every_n_trees is None:
                raise ValueError(
                    "If using eval set, X_eval, y_eval, and eval_every_n_trees must all be defined."
                )

            if not isinstance(X_eval, np.ndarray) or not isinstance(y_eval, np.ndarray):
                raise TypeError("X_eval and y_eval must be numpy arrays.")
            if X_eval.ndim != 2:
                raise ValueError(
                    f"X_eval must be 2-dimensional, got shape {X_eval.shape}"
                )
            if y_eval.ndim != 1:
                raise ValueError(
                    f"y_eval must be 1-dimensional, got shape {y_eval.shape}"
                )
            if X_eval.shape[0] != y_eval.shape[0]:
                raise ValueError(
                    f"X_eval and y_eval must have same number of rows. Got {X_eval.shape[0]} and {y_eval.shape[0]}."
                )

            if not isinstance(eval_every_n_trees, int) or eval_every_n_trees <= 0:
                raise ValueError(
                    f"eval_every_n_trees must be a positive integer, got {eval_every_n_trees}."
                )

            if early_stopping_rounds is not None:
                if (
                    not isinstance(early_stopping_rounds, int)
                    or early_stopping_rounds <= 0
                ):
                    raise ValueError(
                        f"early_stopping_rounds must be a positive integer, got {early_stopping_rounds}."
                    )
            else:
                # No early stopping = set to "never trigger"
                early_stopping_rounds = self.n_estimators + 1

            if eval_metric not in ["mse", "corr", "rmsle"]:
                raise ValueError(
                    f"Invalid eval_metric: {eval_metric}. Choose 'mse' or 'corr', 'rmsle'."
                )

        return early_stopping_rounds  # May have been defaulted here

    def fit(
        self,
        X,
        y,
        era_id=None,
        X_eval=None,
        y_eval=None,
        eval_every_n_trees=None,
        early_stopping_rounds=None,
        eval_metric = "mse",
    ):
        early_stopping_rounds = self.validate_fit_params(
            X, y, era_id, X_eval, y_eval, eval_every_n_trees, early_stopping_rounds, eval_metric
        )

        if era_id is None:
            era_id = np.ones(X.shape[0], dtype="int32")

        # Train data preprocessing
        self.bin_indices, self.era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = (
            self.preprocess_gpu_data(X, y, era_id)
        )
        self.num_samples, self.num_features = X.shape
        self.num_eras = len(self.unique_eras)
        self.era_indices = self.era_indices.to(dtype=torch.int32)
        self.gradients = torch.zeros_like(self.Y_gpu)
        self.root_node_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
        self.base_prediction = self.Y_gpu.mean().item()
        self.gradients += self.base_prediction
        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            k = self.num_features
        self.feature_indices = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

        # ─── Optional Eval Set ───
        if X_eval is not None and y_eval is not None:
            self.bin_indices_eval = self.bin_inference_data(X_eval)
            self.Y_gpu_eval = torch.from_numpy(y_eval).to(torch.float32).to(self.device)
            self.eval_every_n_trees = eval_every_n_trees
            self.early_stopping_rounds = early_stopping_rounds
            self.eval_metric = eval_metric
        else:
            self.bin_indices_eval = None
            self.Y_gpu_eval = None
            self.eval_every_n_trees = None
            self.early_stopping_rounds = None

        # ─── Grow the forest ───
        with torch.no_grad():
            self.grow_forest()

        del self.bin_indices
        del self.Y_gpu

        gc.collect()

        return self

    def preprocess_gpu_data(self, X_np, Y_np, era_id_np):
        with torch.no_grad():
            self.num_samples, self.num_features = X_np.shape

            Y_gpu = torch.from_numpy(Y_np).type(torch.float32).to(self.device)

            era_id_gpu = torch.from_numpy(era_id_np).type(torch.int32).to(self.device)

            bin_indices = torch.empty(
                (self.num_samples, self.num_features), dtype=torch.int8, device="cuda"
            )

            is_integer_type = np.issubdtype(X_np.dtype, np.integer)
            max_vals = X_np.max(axis=0)

            if is_integer_type and np.all(max_vals < self.num_bins):
                print("Detected pre-binned integer input — skipping quantile binning.")
                for f in range(self.num_features):
                    bin_indices[:,f] = torch.as_tensor( X_np[:, f], device=self.device).contiguous()
                # bin_indices = X_np.to("cuda", non_blocking=True).contiguous()

                # We'll store None or an empty tensor in self.bin_edges
                # to indicate that we skip binning at predict-time
                bin_edges = torch.arange(
                    1, self.num_bins, dtype=torch.float32
                ).repeat(self.num_features, 1)
                bin_edges = bin_edges.to(self.device)
                unique_eras, era_indices = torch.unique(
                    era_id_gpu, return_inverse=True
                )
                #NOTE: feature major bin indices
                return bin_indices.t().contiguous(), era_indices, bin_edges, unique_eras, Y_gpu
            
            print("quantile binning.")

            bin_edges = torch.empty(
                (self.num_features, self.num_bins - 1),
                dtype=torch.float32,
                device="cuda",
            )

            for f in range(self.num_features):
                X_f = torch.as_tensor( X_np[:, f], device=self.device, dtype=torch.float32 ).contiguous()
                quantiles = torch.linspace(
                    0, 1, self.num_bins + 1, device="cuda", dtype=X_f.dtype
                )[1:-1]
                bin_edges_f = torch.quantile(
                    X_f, quantiles, dim=0
                ).contiguous()  # shape: [B-1] for 1D input
                bin_indices_f = bin_indices[:, f].contiguous()  # view into output
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices_f)
                bin_indices[:, f] = bin_indices_f
                bin_edges[f, :] = bin_edges_f

            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
            #NOTE feature major bin indices
            return bin_indices.t().contiguous(), era_indices, bin_edges, unique_eras, Y_gpu

    def compute_histograms(self, sample_indices, feature_indices):
        grad_hist = torch.zeros(
            ( self.num_eras, len(feature_indices), self.num_bins), device=self.device, dtype=torch.float32
        )
        hess_hist = torch.zeros(
            ( self.num_eras, len(feature_indices), self.num_bins), device=self.device, dtype=torch.float32
        )

        node_kernel.compute_histogram3(
            self.bin_indices,
            self.residual,
            torch.ones_like(self.residual),
            sample_indices,
            feature_indices,
            self.era_indices,
            grad_hist,
            hess_hist,
            self.num_bins,
            self.threads_per_block,
            self.rows_per_thread,
        )
        return grad_hist, hess_hist

    def find_best_split(self, gradient_histogram, hessian_histogram):
        node_kernel.compute_split(
            gradient_histogram,
            hessian_histogram,
            self.min_split_gain,
            self.min_child_weight,
            self.L2_reg,
            self.per_era_gain,
            self.per_era_direction,
            self.threads_per_block
        )

        if self.num_eras == 1:
            era_splitting_criterion = self.per_era_gain[0,:,:]  # [F, B-1]
            dir_score_mask = era_splitting_criterion > self.min_split_gain
        else:
            directional_agreement = self.per_era_direction.mean(dim=0).abs()  # [F, B-1]
            era_splitting_criterion = self.per_era_gain.mean(dim=0)  # [F, B-1]
            dir_score_mask = ( directional_agreement == directional_agreement.max() ) & (era_splitting_criterion > self.min_split_gain)

        if not dir_score_mask.any():
            return -1, -1
        
        era_splitting_criterion[dir_score_mask == 0] = float("-inf")
        best_idx = torch.argmax(era_splitting_criterion) #index of flattened tensor
        split_bins = self.num_bins - 1
        best_feature = best_idx // split_bins
        best_bin = best_idx % split_bins

        return best_feature.item(), best_bin.item()


    def grow_tree(self, gradient_histogram, hessian_histogram, node_indices, depth):
        if depth == self.max_depth:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": node_indices.numel()}

        parent_size = node_indices.numel()
        local_feature, best_bin = self.find_best_split(
            gradient_histogram, hessian_histogram
        )

        if local_feature == -1:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": parent_size}
        
        #NOTE feature-major layout
        split_mask = self.bin_indices[self.feat_indices_tree[local_feature], node_indices] <= best_bin
        left_indices = node_indices[split_mask]
        right_indices = node_indices[~split_mask]

        left_size = left_indices.numel()
        right_size = right_indices.numel()

        if left_size <= right_size:
            grad_hist_left, hess_hist_left = self.compute_histograms( left_indices, self.feat_indices_tree )
            grad_hist_right = gradient_histogram - grad_hist_left
            hess_hist_right = hessian_histogram - hess_hist_left
        else:
            grad_hist_right, hess_hist_right = self.compute_histograms( right_indices, self.feat_indices_tree )
            grad_hist_left = gradient_histogram - grad_hist_right
            hess_hist_left = hessian_histogram - hess_hist_right

        new_depth = depth + 1
        left_child = self.grow_tree(
            grad_hist_left, hess_hist_left, left_indices, new_depth
        )
        right_child = self.grow_tree(
            grad_hist_right, hess_hist_right, right_indices, new_depth
        )

        return {
            "feature": self.feat_indices_tree[local_feature],
            "bin": best_bin,
            "left": left_child,
            "right": right_child,
        }
    
    def get_eval_metric(self, y_true, y_pred):
        if self.eval_metric == "mse":
            return ((y_true - y_pred) ** 2).mean().item()
        elif self.eval_metric == "corr":
            return 1 - torch.corrcoef(torch.vstack([y_true, y_pred]))[0, 1].item()
        elif self.eval_metric == "rmsle":
            return rmsle_torch(y_true, y_pred).item()
        else:
            raise ValueError(f"Invalid eval_metric: {self.eval_metric}.")

    def compute_eval(self, i):
        if self.eval_every_n_trees == None:
            return
        
        train_loss = ((self.Y_gpu - self.gradients) ** 2).mean().item()
        self.training_loss.append(train_loss)

        if i % self.eval_every_n_trees == 0:
            eval_preds = self.predict_binned(self.bin_indices_eval)
            eval_loss = self.get_eval_metric( self.Y_gpu_eval, eval_preds )
            self.eval_loss.append(eval_loss)

            if len(self.eval_loss) > self.early_stopping_rounds:
                if self.eval_loss[-(self.early_stopping_rounds+1)] < self.eval_loss[-1]:
                    self.stop = True

            print(
                f"🌲 Tree {i+1}/{self.n_estimators} | Train MSE: {train_loss:.6f} | Eval {self.eval_metric}: {eval_loss:.6f}"
            )

            del eval_preds, eval_loss, train_loss

    def grow_forest(self):
        self.training_loss = []
        self.eval_loss = []  # if eval set is given
        self.stop = False

        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            k = self.num_features
            
        self.per_era_gain = torch.zeros(self.num_eras, k, self.num_bins-1, device=self.device, dtype=torch.float32)
        self.per_era_direction = torch.zeros(self.num_eras, k, self.num_bins-1, device=self.device, dtype=torch.float32)

        for i in range(self.n_estimators):
            self.residual = self.Y_gpu - self.gradients

            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(self.num_features, device=self.device, dtype=torch.int32)[:k]

            self.root_gradient_histogram, self.root_hessian_histogram = self.compute_histograms( self.root_node_indices, self.feat_indices_tree )

            tree = self.grow_tree(
                self.root_gradient_histogram,
                self.root_hessian_histogram,
                self.root_node_indices,
                0,
            )
            self.forest[i] = tree

            self.compute_eval(i)

            if self.stop:
                break

        print("Finished training forest.")

    def bin_data_with_existing_edges(self, X_np):
        num_samples = X_np.shape[0]
        bin_indices = torch.zeros(
            (num_samples, self.num_features), dtype=torch.int8, device=self.device
        )
        with torch.no_grad():
            for f in range(self.num_features):
                X_f = torch.as_tensor( X_np[:, f], device=self.device, dtype=torch.float32 ).contiguous()
                bin_edges_f = self.bin_edges[f]
                bin_indices_f = bin_indices[:, f].contiguous()
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices_f)
                bin_indices[:, f] = bin_indices_f

        return bin_indices

    def predict_binned(self, bin_indices):
        num_samples = bin_indices.size(0)
        tree_tensor = torch.stack(
            [
                self.flatten_tree(tree, max_nodes=2 ** (self.max_depth + 1))
                for tree in self.forest
                if tree
            ]
        ).to(self.device)

        out = torch.zeros(num_samples, device=self.device) + self.base_prediction
        node_kernel.predict_forest(
            bin_indices.contiguous(), tree_tensor.contiguous(), self.learning_rate, out
        )

        return out
    
    def bin_inference_data(self, X_np):
        is_integer_type = np.issubdtype(X_np.dtype, np.integer)

        if is_integer_type and X_np.shape[1] == self.num_features:
            max_vals = X_np.max(axis=0)
            if np.all(max_vals < self.num_bins):
                print("Detected pre-binned input at predict-time — skipping binning.")
                is_prebinned = True
            else:
                is_prebinned = False
        else:
            is_prebinned = False

        if is_prebinned:
            bin_indices = torch.empty(
                X_np.shape, dtype=torch.int8, device="cuda"
            )
            for f in range(self.num_features):
                bin_indices[:,f] = torch.as_tensor( X_np[:, f], device=self.device).contiguous()
        else:
            bin_indices = self.bin_data_with_existing_edges(X_np)
        return bin_indices

    def predict(self, X_np):
        bin_indices = self.bin_inference_data(X_np)
        preds = self.predict_binned(bin_indices).cpu().numpy()
        del bin_indices
        return preds

    def flatten_tree(self, tree, max_nodes):
        flat = torch.full((max_nodes, 6), float("nan"), dtype=torch.float32)
        node_counter = [0]
        node_list = []

        def walk(node):
            curr_id = node_counter[0]
            node_counter[0] += 1

            new_node = {"node_id": curr_id}
            if "leaf_value" in node:
                new_node["leaf_value"] = float(node["leaf_value"])
            else:
                new_node["best_feature"] = float(node["feature"])
                new_node["split_bin"] = float(node["bin"])
                new_node["left_id"] = node_counter[0]
                walk(node["left"])
                new_node["right_id"] = node_counter[0]
                walk(node["right"])

            node_list.append(new_node)
            return new_node

        walk(tree)

        for node in node_list:
            i = node["node_id"]
            if "leaf_value" in node:
                flat[i, 4] = 1.0
                flat[i, 5] = node["leaf_value"]
            else:
                flat[i, 0] = node["best_feature"]
                flat[i, 1] = node["split_bin"]
                flat[i, 2] = node["left_id"]
                flat[i, 3] = node["right_id"]
                flat[i, 4] = 0.0

        return flat
