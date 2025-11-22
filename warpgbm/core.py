import torch
import numpy as np
import pickle
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from warpgbm.cuda import node_kernel
from warpgbm.metrics import rmsle_torch, softmax, log_loss_torch, accuracy_torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from typing import Tuple
from torch import Tensor
import gc

class WarpGBM(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        objective="regression",
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=1e-3, # Changed from 20 to 1e-3 for Newton Hessian support
        min_split_gain=0.0,
        threads_per_block=64,
        rows_per_thread=4,
        L2_reg=1e-6,
        device="cuda",
        colsample_bytree=1.0,
        random_state=None,
        warm_start=False,
    ):
        # Validate arguments
        self._validate_hyperparams(
            objective=objective,
            num_bins=num_bins,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            threads_per_block=threads_per_block,
            rows_per_thread=rows_per_thread,
            L2_reg=L2_reg,
            colsample_bytree=colsample_bytree,
        )

        self.objective = objective
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.forest = None
        self.bin_edges = None
        self.base_prediction = None
        self.unique_eras = None
        self.device = device
        self.num_classes = None
        self.classes_ = None
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
        self.forest = [{} for _ in range(self.n_estimators)]
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.warm_start = warm_start
        self.label_encoder = None
        self.feature_importance_ = None
        self.per_era_feature_importance_ = None
        self._is_fitted = False
        self._trees_trained = 0  # Track number of trees already trained

    def _validate_hyperparams(self, **kwargs):
        # Validate objective
        if kwargs["objective"] not in ["regression", "multiclass", "binary"]:
            raise ValueError(
                f"objective must be 'regression', 'binary', or 'multiclass', got {kwargs['objective']}."
            )
        
        # Type checks
        int_params = [
            "num_bins",
            "max_depth",
            "n_estimators",
            "threads_per_block",
            "rows_per_thread",
        ]
        float_params = [
            "learning_rate",
            "min_split_gain",
            "L2_reg",
            "colsample_bytree",
            "min_child_weight", 
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

        if not (2 <= kwargs["num_bins"] <= 255):
            raise ValueError("num_bins must be between 2 and 255 inclusive.")
        if kwargs["max_depth"] < 1:
            raise ValueError("max_depth must be at least 1.")
        if not (0.0 < kwargs["learning_rate"] <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0].")
        if kwargs["n_estimators"] <= 0:
            raise ValueError("n_estimators must be positive.")
        if kwargs["min_child_weight"] <= 0:
            raise ValueError("min_child_weight must be positive.")
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
        if kwargs["L2_reg"] < 0:
            raise ValueError("L2_reg must be non-negative.")
        if kwargs["colsample_bytree"] <= 0 or kwargs["colsample_bytree"] > 1:
            raise ValueError(
                f"Invalid colsample_bytree: {kwargs['colsample_bytree']}. Must be a float value > 0 and <= 1."
            )

    def _compute_tree_predictions(self, tree, bin_indices):
        num_samples = bin_indices.size(0)
        predictions = torch.zeros(num_samples, device=self.device, dtype=torch.float32)
        
        def traverse(node, sample_mask):
            """Recursively traverse tree and assign leaf values."""
            if "leaf_value" in node:
                # Leaf node: assign value to all samples in this leaf
                predictions[sample_mask] = node["leaf_value"] * self.learning_rate
            else:
                # Split node: route samples left or right
                feature_idx = node["feature"]
                split_bin = node["bin"]
                
                # Samples go left if bin_value <= split_bin
                go_left = bin_indices[sample_mask, feature_idx] <= split_bin
                
                left_mask = sample_mask.clone()
                left_mask[sample_mask] = go_left
                right_mask = sample_mask.clone()
                right_mask[sample_mask] = ~go_left
                
                traverse(node["left"], left_mask)
                traverse(node["right"], right_mask)
        
        # Start with all samples
        all_samples = torch.ones(num_samples, dtype=torch.bool, device=self.device)
        traverse(tree, all_samples)
        
        return predictions
    
    def _compute_softmax_gradients_hessians(self, y_true_encoded):
        probs = softmax(self.gradients, dim=1)  # [n_samples, n_classes]
        
        n_samples = y_true_encoded.shape[0]
        y_onehot = torch.zeros(n_samples, self.num_classes, device=self.device)
        y_onehot[torch.arange(n_samples), y_true_encoded.long()] = 1.0
        
        gradients = probs - y_onehot
        hessians = probs * (1.0 - probs)
        hessians = torch.clamp(hessians, min=1e-6)
        
        return gradients, hessians

    def validate_fit_params(
        self, X, y, era_id, X_eval, y_eval, eval_every_n_trees, early_stopping_rounds, eval_metric
    ):
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

        eval_args = [X_eval, y_eval, eval_every_n_trees]
        if any(arg is not None for arg in eval_args):
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
                early_stopping_rounds = self.n_estimators + 1

            valid_metrics = ["mse", "corr", "rmsle", "logloss", "accuracy"]
            if eval_metric not in valid_metrics:
                raise ValueError(
                    f"Invalid eval_metric: {eval_metric}. Choose from {valid_metrics}."
                )

        return early_stopping_rounds

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

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
                torch.cuda.manual_seed_all(self.random_state)

        if not self.warm_start or not self._is_fitted:
            self._is_fitted = False
            self._trees_trained = 0
            self.forest = [{} for _ in range(self.n_estimators)] if self.objective == "regression" else []
            self.training_loss = []
            self.eval_loss = []
        else:
            if X.shape[1] != self.num_features:
                raise ValueError(
                    f"X has {X.shape[1]} features, but model was trained with {self.num_features} features."
                )
            if self._trees_trained >= self.n_estimators:
                print(f"Model already has {self._trees_trained} trees (n_estimators={self.n_estimators}). No additional training needed.")
                return self

        if era_id is None:
            era_id = np.ones(X.shape[0], dtype="int32")

        if self.objective == "multiclass" or self.objective == "binary":
            if not self.warm_start or not self._is_fitted:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                self.classes_ = self.label_encoder.classes_
                self.num_classes = len(self.classes_)
            else:
                y_encoded = self.label_encoder.transform(y)
            
            if self.objective == "binary" and self.num_classes != 2:
                raise ValueError(f"binary objective requires exactly 2 classes, got {self.num_classes}")
            if self.objective == "multiclass" and self.num_classes < 2:
                raise ValueError(f"multiclass objective requires at least 2 classes, got {self.num_classes}")
            
            return self._fit_classification(X, y_encoded, era_id, X_eval, y_eval, 
                                           eval_every_n_trees, early_stopping_rounds, eval_metric)
        else:
            return self._fit_regression(X, y, era_id, X_eval, y_eval,
                                       eval_every_n_trees, early_stopping_rounds, eval_metric)

    def _fit_regression(self, X, y, era_id, X_eval, y_eval, eval_every_n_trees, early_stopping_rounds, eval_metric):
        self.bin_indices, self.era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = (
            self.preprocess_gpu_data(X, y, era_id)
        )
        self.num_samples, self.num_features = X.shape
        self.num_eras = len(self.unique_eras)
        self.era_indices = self.era_indices.to(dtype=torch.int32)
        
        if self.warm_start and self._is_fitted and self._trees_trained > 0:
            self.gradients = torch.zeros_like(self.Y_gpu) + self.base_prediction
            for tree in self.forest[:self._trees_trained]:
                if tree:
                    leaf_updates = self._compute_tree_predictions(tree, self.bin_indices)
                    self.gradients += leaf_updates
        else:
            self.gradients = torch.zeros_like(self.Y_gpu)
            self.base_prediction = self.Y_gpu.mean().item()
            self.gradients += self.base_prediction
        
        self.root_node_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
        self.feature_indices = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

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

        with torch.no_grad():
            self.grow_forest()

        del self.bin_indices
        del self.Y_gpu
        gc.collect()

        return self
    
    def _fit_classification(self, X, y_encoded, era_id, X_eval, y_eval,
                            eval_every_n_trees, early_stopping_rounds, eval_metric):
        # ── Train data preprocessing ──
        self.bin_indices, self.era_indices, self.bin_edges, self.unique_eras, _ = (
            self.preprocess_gpu_data(X, y_encoded, era_id)
        )
        self.num_samples, self.num_features = X.shape
        self.num_eras = len(self.unique_eras)
        self.era_indices = self.era_indices.to(dtype=torch.int32)

        # Labels on device (int32)
        self.Y_gpu = torch.from_numpy(y_encoded).to(torch.int32).to(self.device)

        # Cache era_ends once (exclusive ends)
        E = int(self.era_indices.max().item()) + 1
        self._era_ends = torch.cumsum(
            torch.bincount(self.era_indices, minlength=E).to(torch.int32),
            dim=0
        ).to(torch.int32)

        # Class log-priors (for stable init + consistent predict init)
        self.class_log_prior_ = torch.empty(self.num_classes, device=self.device, dtype=torch.float32)
        for k in range(self.num_classes):
            pk = (self.Y_gpu == k).float().mean()
            self.class_log_prior_[k] = torch.log(pk + 1e-10)

        # ── Initialize scores F[i,k] ──
        if self.warm_start and self._is_fitted and self._trees_trained > 0:
            # Start from priors
            self.gradients = self.class_log_prior_.unsqueeze(0).expand(self.num_samples, -1).clone()
            # Add predictions from existing trees
            for round_trees in self.forest[:self._trees_trained]:
                for class_k, tree in enumerate(round_trees):
                    if tree:
                        leaf_updates = self._compute_tree_predictions(tree, self.bin_indices)
                        self.gradients[:, class_k] += leaf_updates
        else:
            # Fresh start = just priors
            self.gradients = self.class_log_prior_.unsqueeze(0).expand(self.num_samples, -1).clone()

        self.root_node_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
        self.feature_indices = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

        # ── Optional Eval Set ──
        if X_eval is not None and y_eval is not None:
            self.bin_indices_eval = self.bin_inference_data(X_eval)
            y_eval_encoded = self.label_encoder.transform(y_eval)
            self.Y_gpu_eval = torch.from_numpy(y_eval_encoded).to(torch.int32).to(self.device)
            self.eval_every_n_trees = eval_every_n_trees
            self.early_stopping_rounds = early_stopping_rounds
            self.eval_metric = eval_metric if eval_metric != "mse" else "logloss"
        else:
            self.bin_indices_eval = None
            self.Y_gpu_eval = None
            self.eval_every_n_trees = None
            self.early_stopping_rounds = None

        # ── Grow the forest (K trees per iteration) ──
        with torch.no_grad():
            self.grow_forest_multiclass()

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
                return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu
            
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
            return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu

    # --- Helpers ---

    def _era_ends_from_indices(self) -> torch.Tensor:
        # Precomputed once in _fit_classification
        return self._era_ends

    def _pack_idx_mat(self, node_idx_by_class):
        """
        node_idx_by_class: list[Tensor[int32 or int64]] length K; each 1-D GPU tensor of row ids.
        Returns:
        idx_mat: [K, Mmax] int32 (CUDA)
        idx_len: [K] int32 (CUDA)
        Mmax:    int
        """
        K = len(node_idx_by_class)
        if K == 0:
            idx_mat = torch.empty((0, 1), device=self.device, dtype=torch.int32)
            idx_len = torch.zeros(0, device=self.device, dtype=torch.int32)
            return idx_mat, idx_len, 1

        # Fast packing using pad_sequence (C++ impl)
        idx_mat = pad_sequence(node_idx_by_class, batch_first=True, padding_value=-1).to(dtype=torch.int32)
        idx_len = torch.tensor([len(t) for t in node_idx_by_class], device=self.device, dtype=torch.int32)
        Mmax = idx_mat.size(1)
        
        return idx_mat, idx_len, Mmax

    def _mc_hist_batched_for_node(
        self,
        *,
        feat_idx: torch.Tensor,           # [k] int32
        grads: torch.Tensor,              # [N, K_total] float32 (full tensor, avoid copy)
        hess: torch.Tensor,               # [N, K_total] float32 (full tensor, avoid copy)
        idx_mat: torch.Tensor,            # [G, Mmax] int32
        idx_len: torch.Tensor,            # [G] int32
        active_classes: torch.Tensor,     # [G] int32 (global class indices for G/H access)
        k_tile_hint: int = 0,
        threads_per_block_hint: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched per-era histograms for G groups (classes or child subsets).
        Optimized to use indirect class access in kernel.
        Returns:
            GH, HH with shape [E, k, G, B] (float32, contiguous)
        """
        GH, HH = node_kernel.h_des_mc(
            self.bin_indices,                              # [N,F] int8
            grads.contiguous(),                            # [N,K_total] f32
            hess.contiguous(),                             # [N,K_total] f32
            idx_mat.contiguous(),                          # [G,Mmax] i32
            idx_len.contiguous(),                          # [G] i32
            feat_idx.contiguous(),                         # [k] i32
            self.era_indices.contiguous(),                 # [N] i32
            active_classes.contiguous().to(torch.int32),   # [G] i32 (active class list)
            int(self.num_bins),
            int(k_tile_hint),
            int(threads_per_block_hint),
        )
        # Expect [E, k, G, B]
        return GH, HH

    def _get_split_scratch(self, k_cur: int):
        need_new = (
            not hasattr(self, "_scratch_gain") or
            self._scratch_gain is None or
            self._scratch_gain.shape[1] != k_cur
        )
        if need_new:
            self._scratch_gain = torch.zeros(
                self.num_eras, k_cur, self.num_bins - 1,
                device=self.device, dtype=torch.float32
            )
            self._scratch_dir = torch.zeros_like(self._scratch_gain)
        else:
            self._scratch_gain.zero_()
            self._scratch_dir.zero_()
        return self._scratch_gain, self._scratch_dir

    def _partition_one_class(self, idx: torch.Tensor, global_feat: int, split_bin: int):
        """Split indices for a single class/node by feature<=bin (all on GPU)."""
        if idx.numel() == 0:
            return idx, idx
        fcol = self.bin_indices[idx, global_feat]  # int8
        mask_left = (fcol <= split_bin)
        left = idx[mask_left]
        right = idx[~mask_left]
        return left, right

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
    
    def compute_histograms_multiclass(self, sample_indices, feature_indices, grad, hess):
        """
        Compute histograms for multiclass - similar to regression but with explicit grad/hess
        """
        grad_hist = torch.zeros(
            (self.num_eras, len(feature_indices), self.num_bins), 
            device=self.device, dtype=torch.float32
        )
        hess_hist = torch.zeros(
            (self.num_eras, len(feature_indices), self.num_bins), 
            device=self.device, dtype=torch.float32
        )

        # We need to create temporary tensors that match what compute_histogram3 expects
        # It expects self.residual, but we have grad/hess per class
        # Save the original residual
        saved_residual = self.residual if hasattr(self, 'residual') else None
        
        # Create a dummy dataset structure for histogram computation
        # The histogram kernel accumulates residuals, so we pass grad directly as residual
        self.residual = grad
        
        node_kernel.compute_histogram3(
            self.bin_indices,
            self.residual,
            sample_indices,
            feature_indices,
            self.era_indices,
            grad_hist,
            hess_hist,
            self.num_bins,
            self.threads_per_block,
            self.rows_per_thread,
        )
        
        # Restore
        if saved_residual is not None:
            self.residual = saved_residual
            
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

    def find_best_split_classification(
            self,
            GH_parent: torch.Tensor,          # [E, k, K, B]
            HH_parent: torch.Tensor,          # [E, k, K, B]
            feat_idx: torch.Tensor,           # [k] int32 (local -> global feature ids)
            node_idx_by_class: list[torch.Tensor],
        ):
            """
            Vectorized 'find_best_split' for multiclass.

            GH_parent / HH_parent : [E, k, K, B]
            feat_idx              : [k] int32
            node_idx_by_class     : list of length K, each a 1D CUDA int32 index tensor
            
            Returns:
                best_local_feat : list[int] length K (0..k-1 or -1)
                best_bin        : list[int] length K (0..B-2 or -1)
                can_split       : list[bool] length K
            """
            device = GH_parent.device
            E, k, K, B = GH_parent.shape
            F = k  # local feature count
            assert E == self.num_eras

            # --- 1) Very cheap sample-count gating per class ---
            class_sizes = torch.tensor(
                [int(idx.numel()) for idx in node_idx_by_class],
                device=device,
                dtype=torch.float32,
            )
            # Require at least 2 samples in the node to even consider splitting
            enough_samples = class_sizes >= 2.0   # [K]

            # --- 2) Flatten (class, feature) into one feature axis for compute_split ---
            # GH_parent: [E, F, K, B] -> [E, K*F, B]
            GH_flat = GH_parent.permute(0, 2, 1, 3).reshape(E, K * F, B).contiguous()
            HH_flat = HH_parent.permute(0, 2, 1, 3).reshape(E, K * F, B).contiguous()

            per_era_gain_flat = torch.zeros(
                E, K * F, B - 1, device=device, dtype=torch.float32
            )
            per_era_dir_flat = torch.zeros_like(per_era_gain_flat)

            node_kernel.compute_split(
                GH_flat,
                HH_flat,
                self.min_split_gain,
                self.min_child_weight,
                self.L2_reg,
                per_era_gain_flat,
                per_era_dir_flat,
                self.threads_per_block,
            )

            # Reshape back to [E, K, F, B-1]
            per_era_gain_cf = per_era_gain_flat.view(E, K, F, B - 1)
            per_era_dir_cf  = per_era_dir_flat.view(E, K, F, B - 1)

            # --- 3) Collapse eras, reproduce regression-era logic per class ---
            if E == 1:
                # era_splitting_criterion = per_era_gain[0, :, :]
                crit = per_era_gain_cf[0]        # [K, F, B-1]
                crit_2d = crit.view(K, -1)       # [K, F*(B-1)]
                mask_2d = crit_2d > self.min_split_gain
            else:
                # directional_agreement = per_era_direction.mean(dim=0).abs()
                dir_agree = per_era_dir_cf.mean(dim=0).abs()   # [K, F, B-1]
                crit      = per_era_gain_cf.mean(dim=0)        # [K, F, B-1]

                dir_2d  = dir_agree.view(K, -1)                # [K, F*(B-1)]
                crit_2d = crit.view(K, -1)                     # [K, F*(B-1)]

                max_dir, _ = dir_2d.max(dim=1, keepdim=True)   # [K, 1]
                mask_2d = (dir_2d == max_dir) & (crit_2d > self.min_split_gain)

            # Apply sample-count gating
            mask_2d = mask_2d & enough_samples.view(K, 1)

            # --- 4) Argmax per class under the mask ---
            best_local_feat = [-1] * K
            best_bin = [-1] * K
            can_split = [False] * K

            if mask_2d.numel() == 0:
                return best_local_feat, best_bin, can_split

            neg_inf = torch.finfo(crit_2d.dtype).min
            crit_masked = torch.where(mask_2d, crit_2d, neg_inf)

            has_candidate = mask_2d.any(dim=1)       # [K]
            best_flat_idx = crit_masked.argmax(dim=1)  # [K]

            # --- 5) Decode (feature, bin) and update feature importance ---
            for c in range(K):
                if not bool(has_candidate[c].item()):
                    continue

                flat = int(best_flat_idx[c].item())
                lf = flat // (B - 1)   # local feature index
                lb = flat % (B - 1)    # bin index

                best_local_feat[c] = lf
                best_bin[c] = lb
                can_split[c] = True

                # Map to global feature id
                global_f = int(feat_idx[lf].item())

                # per-era gains for this (class, feature, bin)
                per_era_gains = per_era_gain_cf[:, c, lf, lb]  # [E]
                for e_idx in range(self.num_eras):
                    self.per_era_feature_importance_[e_idx, global_f] += float(
                        per_era_gains[e_idx].item()
                    )

            return best_local_feat, best_bin, can_split



    def grow_tree(self, gradient_histogram, hessian_histogram, node_indices, depth, class_k=None):
        if depth == self.max_depth:
            leaf_value = self.residual[node_indices].mean()
            if class_k is not None:
                # Multiclass: update specific class column
                self.gradients[node_indices, class_k] += self.learning_rate * leaf_value
            else:
                # Regression: update 1D gradients
                self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": node_indices.numel()}

        parent_size = node_indices.numel()
        local_feature, best_bin = self.find_best_split(
            gradient_histogram, hessian_histogram
        )

        if local_feature == -1:
            leaf_value = self.residual[node_indices].mean()
            if class_k is not None:
                # Multiclass: update specific class column
                self.gradients[node_indices, class_k] += self.learning_rate * leaf_value
            else:
                # Regression: update 1D gradients
                self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": parent_size}
        
        # Track feature importance: accumulate per-era gains for the chosen feature
        global_feature_idx = self.feat_indices_tree[local_feature].item()
        per_era_gains = self.per_era_gain[:, local_feature, best_bin]  # [num_eras]
        
        # OPTIMIZATION: Vectorized GPU accumulation
        self.per_era_feature_importance_[:, global_feature_idx] += per_era_gains
        
        split_mask = self.bin_indices[node_indices, self.feat_indices_tree[local_feature]] <= best_bin
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
            grad_hist_left, hess_hist_left, left_indices, new_depth, class_k
        )
        right_child = self.grow_tree(
            grad_hist_right, hess_hist_right, right_indices, new_depth, class_k
        )

        return {
            "feature": self.feat_indices_tree[local_feature],
            "bin": best_bin,
            "left": left_child,
            "right": right_child,
        }

    # --- Multiclass Recursive Grow with Newton Logic ---

    def _grow_tree_multiclass_round(
        self,
        grads: torch.Tensor,        # [N, K]
        hess: torch.Tensor,         # [N, K]
        feat_idx: torch.Tensor,     # [k]
        node_idx_by_class,          # list[Tensor[int32]] length K
        depth: int,
        seed_hist_by_class: dict | None = None,
    ):
        device = self.device
        K = len(node_idx_by_class)
        k = int(feat_idx.numel())
        empty_idx = torch.empty(0, dtype=torch.int32, device=device)

        # ── Base case: make leaves ──
        if depth == self.max_depth or k == 0:
            trees = []
            for c in range(K):
                idx = node_idx_by_class[c]
                if idx.numel() == 0:
                    trees.append({"leaf_value": 0.0, "samples": 0})
                else:
                    # NEWTON STEP: Leaf = - Sum(G) / (Sum(H) + L2)
                    # NOTE: We expect 'grads' passed in to be Gradients (p-y).
                    # So we take -sum(grads).
                    # If 'grads' passed in were negative gradients, we would take sum(grads).
                    # Let's assume 'grads' here are Gradients (p-y).
                    g_sum = grads[idx, c].sum()
                    h_sum = hess[idx, c].sum()
                    
                    leaf_val = -g_sum / (h_sum + self.L2_reg)
                    
                    self.gradients[idx, c] += self.learning_rate * leaf_val
                    trees.append({"leaf_value": float(leaf_val.item()), "samples": int(idx.numel())})
            return trees

        # ── Build/collect parent histograms ──
        if seed_hist_by_class:
            need_build = []
            GH_parent = torch.zeros((self.num_eras, k, K, self.num_bins),
                                    device=device, dtype=torch.float32)
            HH_parent = torch.zeros_like(GH_parent)
            for c in range(K):
                if node_idx_by_class[c].numel() == 0:
                    continue
                seed = seed_hist_by_class.get(c, None)
                if seed is None:
                    need_build.append(c)
                else:
                    GH_parent[:, :, c, :] = seed[0]
                    HH_parent[:, :, c, :] = seed[1]

            if need_build:
                idx_list = [node_idx_by_class[c] for c in need_build]
                idx_mat, idx_len, _ = self._pack_idx_mat(idx_list)
                
                # Indirect access list
                sel_c = torch.tensor(need_build, device=device, dtype=torch.int32)
                
                # Optimized call: Pass full grads/hess + indices
                GH_miss, HH_miss = self._mc_hist_batched_for_node(
                    feat_idx=feat_idx,
                    grads=grads,
                    hess=hess,
                    idx_mat=idx_mat,
                    idx_len=idx_len,
                    active_classes=sel_c
                )
                
                sel_c_long = sel_c.to(torch.long)
                GH_parent.index_copy_(2, sel_c_long, GH_miss)
                HH_parent.index_copy_(2, sel_c_long, HH_miss)
        else:
            idx_mat, idx_len, _ = self._pack_idx_mat(node_idx_by_class)
            all_classes = torch.arange(K, device=device, dtype=torch.int32)
            
            GH_parent, HH_parent = self._mc_hist_batched_for_node(
                feat_idx=feat_idx,
                grads=grads,
                hess=hess,
                idx_mat=idx_mat,
                idx_len=idx_len,
                active_classes=all_classes
            )

        # ── Pick best split per class ──
        

        self.per_era_gain = torch.zeros(self.num_eras, k, self.num_bins - 1,
                                        device=device, dtype=torch.float32)
        self.per_era_direction = torch.zeros_like(self.per_era_gain)

        best_local_feat, best_bin, can_split = self.find_best_split_classification(
            GH_parent, HH_parent, feat_idx, node_idx_by_class
        )

        if not any(can_split):
            # Fallback to leaves
            return self._grow_tree_multiclass_round(grads, hess, feat_idx, node_idx_by_class, self.max_depth)

        # ── Partition per class ──
        active_classes = [c for c in range(K) if can_split[c]]
        small_is_left = {}
        left_idx_by_c, right_idx_by_c = {}, {}
        
        for c in active_classes:
            lf, lb = best_local_feat[c], best_bin[c]
            
            # Check children weights
            HHc_feat = HH_parent[:, lf, c, :]
            total = HHc_feat.sum()
            left_H = HHc_feat[:, : (lb + 1)].sum()
            right_H = total - left_H
            
            if left_H < self.min_child_weight or right_H < self.min_child_weight:
                can_split[c] = False
                continue

            small_is_left[c] = (left_H <= right_H)
            
            idx = node_idx_by_class[c]
            gl_f = int(feat_idx[lf].item())
            L, R = self._partition_one_class(idx, gl_f, lb)
            left_idx_by_c[c] = L
            right_idx_by_c[c] = R

        # Re-filter
        active_classes = [c for c in active_classes if can_split[c]]

        # ── Build small-child histograms ──
        small_lists = [
            (left_idx_by_c[c] if small_is_left[c] else right_idx_by_c[c])
            for c in active_classes
        ]
        
        if small_lists:
            idx_mat_small, idx_len_small, _ = self._pack_idx_mat(small_lists)
            sel_c_small = torch.tensor(active_classes, device=device, dtype=torch.int32)
            
            GH_small, HH_small = self._mc_hist_batched_for_node(
                feat_idx=feat_idx,
                grads=grads,
                hess=hess,
                idx_mat=idx_mat_small,
                idx_len=idx_len_small,
                active_classes=sel_c_small
            )
            
            sel = sel_c_small.to(torch.long)
            GH_parent_sel = GH_parent.index_select(2, sel)
            HH_parent_sel = HH_parent.index_select(2, sel)
            GH_big = GH_parent_sel.sub_(GH_small)
            HH_big = HH_parent_sel.sub_(HH_small)
        else:
            GH_small, HH_small, GH_big, HH_big = None, None, None, None

        # ── Recurse ──
        seed_left_ALL = {}
        seed_right_ALL = {}

        if GH_small is not None:
            for j, c in enumerate(active_classes):
                GH_s = GH_small[:, :, j, :].contiguous()
                HH_s = HH_small[:, :, j, :].contiguous()
                GH_b = GH_big[:, :, j, :].contiguous()
                HH_b = HH_big[:, :, j, :].contiguous()

                if small_is_left[c]:
                    seed_left_ALL[c] = (GH_s, HH_s)
                    seed_right_ALL[c] = (GH_b, HH_b)
                else:
                    seed_left_ALL[c] = (GH_b, HH_b)
                    seed_right_ALL[c] = (GH_s, HH_s)
        
        # Prepare lists for next level (non-split classes carry empty indices -> become leaves)
        left_nodes_next = [left_idx_by_c.get(c, empty_idx) if can_split[c] else empty_idx for c in range(K)]
        right_nodes_next = [right_idx_by_c.get(c, empty_idx) if can_split[c] else empty_idx for c in range(K)]
        
        all_left = self._grow_tree_multiclass_round(
            grads, hess, feat_idx, left_nodes_next, depth + 1, seed_hist_by_class=seed_left_ALL
        )
        all_right = self._grow_tree_multiclass_round(
            grads, hess, feat_idx, right_nodes_next, depth + 1, seed_hist_by_class=seed_right_ALL
        )
        
        trees = []
        for c in range(K):
            if can_split[c]:
                trees.append({
                    "feature": torch.tensor(int(feat_idx[best_local_feat[c]].item()), dtype=torch.float32),
                    "bin": int(best_bin[c]),
                    "left": all_left[c],
                    "right": all_right[c],
                })
            else:
                # Create leaf for classes that didn't split
                idx = node_idx_by_class[c]
                if idx.numel() == 0:
                    trees.append({"leaf_value": 0.0, "samples": 0})
                else:
                    # Newton leaf
                    g_sum = grads[idx, c].sum()
                    h_sum = hess[idx, c].sum()
                    leaf_val = -g_sum / (h_sum + self.L2_reg)
                    self.gradients[idx, c] += self.learning_rate * leaf_val
                    trees.append({"leaf_value": float(leaf_val.item()), "samples": int(idx.numel())})

        return trees

    def grow_forest(self):
        """Regression forest growing (unchanged)"""
        if not hasattr(self, 'training_loss') or not self.warm_start or not self._is_fitted:
            self.training_loss = []
            self.eval_loss = []
            self.per_era_feature_importance_ = torch.zeros((self.num_eras, self.num_features), device=self.device, dtype=torch.float32)
        
        self.stop = False

        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            k = self.num_features
            
        self.per_era_gain = torch.zeros(self.num_eras, k, self.num_bins-1, device=self.device, dtype=torch.float32)
        self.per_era_direction = torch.zeros(self.num_eras, k, self.num_bins-1, device=self.device, dtype=torch.float32)

        start_iter = self._trees_trained if self.warm_start and self._is_fitted else 0
        
        if len(self.forest) < self.n_estimators:
            self.forest.extend([{} for _ in range(self.n_estimators - len(self.forest))])

        for i in range(start_iter, self.n_estimators):
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
            self._trees_trained = i + 1

            self.compute_eval(i)

            if self.stop:
                break

        self.feature_importance_ = self.per_era_feature_importance_.sum(axis=0).cpu().numpy()
        self.per_era_feature_importance_ = self.per_era_feature_importance_.cpu().numpy()
        self._is_fitted = True
        
        print(f"Finished training forest. Total trees: {self._trees_trained}")

    def grow_forest_multiclass(self):
        """
        Optimized Multiclass (Newton)
        """
        if not hasattr(self, 'training_loss') or not self.warm_start or not self._is_fitted:
            self.training_loss = []
            self.eval_loss = []
            self.per_era_feature_importance_ = torch.zeros((self.num_eras, self.num_features), device=self.device, dtype=torch.float32)
            self.forest = []

        self.stop = False

        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            k = self.num_features

        self.per_era_gain = torch.zeros(self.num_eras, k, self.num_bins - 1, device=self.device, dtype=torch.float32)
        self.per_era_direction = torch.zeros_like(self.per_era_gain)

        start_iter = self._trees_trained if self.warm_start and self._is_fitted else 0

        for i in range(start_iter, self.n_estimators):
            # Newton Boosting: Get Gradients (p-y) and Hessians (p(1-p))
            grads, hess = self._compute_softmax_gradients_hessians(self.Y_gpu)
            
            # Note: We pass 'grads' (p-y) and 'hess' (p(1-p))
            # The histogram will sum them.
            # The gain formula in kernel expects (SumG)^2 / SumH.
            # Since we want to MINIMIZE loss, and gain is Reduction in Loss,
            # G^2/H works for Newton direction regardless of sign.
            # But for LEAF value: -Sum(p-y) / Sum(p(1-p)).
            # To use the same G for histogram and leaf, we use 'grads' directly.
            # BUT wait: Gain calculation uses (G_L^2/H_L + ...).
            # If G is (p-y), then -G is (y-p). Squared it is the same.
            # BUT we must use -G for leaf value if we sum G.
            
            # Optimization: We pass a negative gradient if we want the kernel to calculate Gain correctly?
            # Actually, Gain depends on G^2, so sign doesn't matter for splits.
            # The only place sign matters is leaf value.
            # Leaf value = -Sum(G) / Sum(H).
            # So we can pass G, sum it to S, then Leaf = -S/H.
            
            # NOTE: We pass -grads to the histogram builder in previous versions.
            # Why? Because usually G = dL/dF. For regression (MSE), L = 1/2(y-p)^2 -> dL/dp = -(y-p) = p-y.
            # Negative gradient is (y-p).
            # XGBoost uses g = dL/dpred.
            # Obj approx: g*ft + 1/2*h*ft^2.
            # Opt ft = -g/h.
            # Gain = 1/2 * g^2/h.
            # So yes, using 'grads' (p-y) is correct for XGBoost style.
            # And Leaf = -sum(grads)/sum(hess).
            
            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(self.num_features, device=self.device, dtype=torch.int32)[:k]
            else:
                self.feat_indices_tree = self.feature_indices

            root_idx = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
            node_idx_by_class = [root_idx for _ in range(self.num_classes)]

            # IMPORTANT: Pass 'grads' (p-y) and 'hess' (p(1-p))
            # We previously passed -grads. Let's stick to passing 'grads' and negating at leaf calc.
            # Actually, looking at my `_grow_tree_multiclass_round` above:
            # leaf_val = -g_sum / (h_sum + L2).
            # So passing `grads` is consistent with this formula.
            
            trees_k = self._grow_tree_multiclass_round(
                grads, hess,
                self.feat_indices_tree.to(torch.int32),
                node_idx_by_class, 
                depth=0
            )

            self.forest.append(trees_k)
            self._trees_trained = i + 1

            self.compute_eval_multiclass(i)
            if self.stop:
                break

        self.feature_importance_ = self.per_era_feature_importance_.sum(axis=0).cpu().numpy()
        self.per_era_feature_importance_ = self.per_era_feature_importance_.cpu().numpy()
        self._is_fitted = True
        print(f"Finished training multiclass forest. Total rounds: {self._trees_trained} "
            f"({self._trees_trained * self.num_classes} trees)")

    
    def compute_eval_multiclass(self, i):
        if self.eval_every_n_trees is None:
            return
        
        # Compute training loss
        probs_train = softmax(self.gradients, dim=1)
        train_loss = log_loss_torch(self.Y_gpu, probs_train).item()
        self.training_loss.append(train_loss)

        if i % self.eval_every_n_trees == 0:
            # Get predictions on eval set
            eval_probs = self.predict_proba_binned(self.bin_indices_eval)
            
            if self.eval_metric == "logloss":
                eval_loss = log_loss_torch(self.Y_gpu_eval, eval_probs).item()
            elif self.eval_metric == "accuracy":
                eval_preds = torch.argmax(eval_probs, dim=1)
                eval_loss = 1.0 - accuracy_torch(self.Y_gpu_eval, eval_preds).item()
            else:
                eval_loss = log_loss_torch(self.Y_gpu_eval, eval_probs).item()
            
            self.eval_loss.append(eval_loss)

            if len(self.eval_loss) > self.early_stopping_rounds:
                if self.eval_loss[-(self.early_stopping_rounds+1)] < self.eval_loss[-1]:
                    self.stop = True

            print(
                f"🌲 Round {i+1}/{self.n_estimators} ({self.num_classes} trees) | "
                f"Train LogLoss: {train_loss:.6f} | Eval {self.eval_metric}: {eval_loss:.6f}"
            )

            del eval_probs, eval_loss, train_loss
    
    def compute_histograms_multiclass(self, sample_indices, feature_indices, grad, hess):
        grad_hist = torch.zeros(
            (self.num_eras, len(feature_indices), self.num_bins), 
            device=self.device, dtype=torch.float32
        )
        hess_hist = torch.zeros(
            (self.num_eras, len(feature_indices), self.num_bins), 
            device=self.device, dtype=torch.float32
        )

        saved_residual = self.residual if hasattr(self, 'residual') else None
        self.residual = grad
        
        node_kernel.compute_histogram3(
            self.bin_indices,
            self.residual,
            sample_indices,
            feature_indices,
            self.era_indices,
            grad_hist,
            hess_hist,
            self.num_bins,
            self.threads_per_block,
            self.rows_per_thread,
        )
        
        if saved_residual is not None:
            self.residual = saved_residual
            
        return grad_hist, hess_hist

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
                # print("Detected pre-binned input at predict-time — skipping binning.")
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
        if self.objective == "multiclass" or self.objective == "binary":
            probs = self.predict_proba(X_np)
            class_indices = np.argmax(probs, axis=1)
            return self.label_encoder.inverse_transform(class_indices)
        else:
            bin_indices = self.bin_inference_data(X_np)
            preds = self.predict_binned(bin_indices).cpu().numpy()
            del bin_indices
            return preds
    
    def predict_proba(self, X_np):
        if self.objective not in ["multiclass", "binary"]:
            raise ValueError("predict_proba only available for classification objectives")
        
        bin_indices = self.bin_inference_data(X_np)
        probs = self.predict_proba_binned(bin_indices).cpu().numpy()
        del bin_indices
        return probs
    
    def predict_proba_binned(self, bin_indices):
        num_samples = bin_indices.size(0)
        F = self.class_log_prior_.unsqueeze(0).expand(num_samples, -1).clone()

        for k in range(self.num_classes):
            trees_k = []
            for round_trees in self.forest:
                if round_trees and len(round_trees) > k:
                    trees_k.append(round_trees[k])
            
            if not trees_k:
                continue
                
            tree_tensor = torch.stack([
                self.flatten_tree(t, max_nodes=2 ** (self.max_depth + 1)) 
                for t in trees_k
            ]).to(self.device)

            tree_preds = torch.zeros(num_samples, device=self.device)
            node_kernel.predict_forest(
                bin_indices.contiguous(),
                tree_tensor.contiguous(),
                self.learning_rate,
                tree_preds
            )
            F[:, k] += tree_preds

        probs = softmax(F, dim=1)
        return probs


    def get_feature_importance(self, importance_type='gain', normalize=True):
        if self.feature_importance_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        if importance_type != 'gain':
            raise ValueError(f"importance_type '{importance_type}' not supported. Use 'gain'.")
        
        importance = self.feature_importance_.copy()
        
        if normalize and importance.sum() > 0:
            importance = importance / importance.sum()
        
        return importance
    
    def get_per_era_feature_importance(self, normalize=True):
        if self.per_era_feature_importance_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        importance = self.per_era_feature_importance_.copy()
        
        if normalize:
            for era_idx in range(importance.shape[0]):
                era_sum = importance[era_idx].sum()
                if era_sum > 0:
                    importance[era_idx] /= era_sum
        
        return importance

    def save_model(self, path):
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        state = {
            'objective': self.objective,
            'num_bins': self.num_bins,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'min_child_weight': self.min_child_weight,
            'min_split_gain': self.min_split_gain,
            'L2_reg': self.L2_reg,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
            'warm_start': self.warm_start,
            'forest': self.forest,
            'bin_edges': self.bin_edges,
            'base_prediction': self.base_prediction,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'classes_': self.classes_,
            'label_encoder': self.label_encoder,
            'feature_importance_': self.feature_importance_,
            'per_era_feature_importance_': self.per_era_feature_importance_,
            '_is_fitted': self._is_fitted,
            '_trees_trained': self._trees_trained,
            'training_loss': self.training_loss if hasattr(self, 'training_loss') else [],
            'eval_loss': self.eval_loss if hasattr(self, 'eval_loss') else [],
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.objective = state['objective']
        self.num_bins = state['num_bins']
        self.max_depth = state['max_depth']
        self.learning_rate = state['learning_rate']
        self.n_estimators = state['n_estimators']
        self.min_child_weight = state['min_child_weight']
        self.min_split_gain = state['min_split_gain']
        self.L2_reg = state['L2_reg']
        self.colsample_bytree = state['colsample_bytree']
        self.random_state = state['random_state']
        self.warm_start = state['warm_start']
        self.forest = state['forest']
        self.bin_edges = state['bin_edges']
        self.base_prediction = state['base_prediction']
        self.num_features = state['num_features']
        self.num_classes = state['num_classes']
        self.classes_ = state['classes_']
        self.label_encoder = state['label_encoder']
        self.feature_importance_ = state['feature_importance_']
        self.per_era_feature_importance_ = state['per_era_feature_importance_']
        self._is_fitted = state['_is_fitted']
        self._trees_trained = state['_trees_trained']
        self.training_loss = state.get('training_loss', [])
        self.eval_loss = state.get('eval_loss', [])
        
        print(f"Model loaded from {path} ({self._trees_trained} trees)")
        
        return self

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