import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from warpgbm.cuda import node_kernel
from tqdm import tqdm
from typing import Tuple
from torch import Tensor
import gc

histogram_kernels = {
    "hist1": node_kernel.compute_histogram,
    "hist2": node_kernel.compute_histogram2,
    "hist3": node_kernel.compute_histogram3,
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
        histogram_computer="hist3",
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
            histogram_computer=histogram_computer,
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
        self.compute_histogram = histogram_kernels[histogram_computer]
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
        if kwargs["histogram_computer"] not in histogram_kernels:
            raise ValueError(
                f"Invalid histogram_computer: {kwargs['histogram_computer']}. Choose from {list(histogram_kernels.keys())}."
            )
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

            if eval_metric not in ["mse", "corr"]:
                raise ValueError(
                    f"Invalid eval_metric: {eval_metric}. Choose 'mse' or 'corr'."
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
        self.era_indices = self.era_indices.to(torch.int16)
        self.num_samples, self.num_features = X.shape
        self.gradients = torch.zeros_like(self.Y_gpu)
        self.root_node_indices = torch.arange(self.num_samples, device=self.device)
        self.base_prediction = self.Y_gpu.mean().item()
        self.gradients += self.base_prediction
        self.best_gains = torch.zeros(self.num_features, device=self.device)
        self.best_bins = torch.zeros(
            self.num_features, device=self.device, dtype=torch.int32
        )
        self.feature_indices = torch.arange(self.num_features, device=self.device)

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
            is_integer_type = np.issubdtype(X_np.dtype, np.integer)
            if is_integer_type:
                max_vals = X_np.max(axis=0)
                if np.all(max_vals < self.num_bins):
                    print(
                        "Detected pre-binned integer input — skipping quantile binning."
                    )
                    bin_indices = (
                        torch.from_numpy(X_np)
                        .to(self.device)
                        .contiguous()
                        .to(torch.int8)
                    )

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
                else:
                    print(
                        "Integer input detected, but values exceed num_bins — falling back to quantile binning."
                    )

            bin_indices = torch.empty(
                (self.num_samples, self.num_features), dtype=torch.int8, device="cuda"
            )
            bin_edges = torch.empty(
                (self.num_features, self.num_bins - 1),
                dtype=torch.float32,
                device="cuda",
            )

            X_np = torch.from_numpy(X_np).to(torch.float32).pin_memory()

            for f in range(self.num_features):
                X_f = X_np[:, f].to("cuda", non_blocking=True)
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

    def compute_histograms(self,
                           bin_indices_sub: torch.Tensor,  # [N, F_sub], int8
                           gradients: torch.Tensor,        # [N], float32
                           era_ids: torch.Tensor           # [N], int16
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds a [F_sub, B*E] gradient & hessian histogram in one GPU pass.

        Parameters
        ----------
        bin_indices_sub : (N, F_sub) int8 tensor of binned features
        gradients       : (N,) float32 tensor of residuals
        era_ids         : (N,) int16 tensor of era IDs in [0, E)

        Returns
        -------
        grad_hist : (F_sub, B*E) float32 tensor
        hess_hist : (F_sub, B*E) float32 tensor
        """
        N, F_sub = bin_indices_sub.shape
        B        = self.num_bins
        E        = len(self.unique_eras)

        # allocate output
        grad_hist = torch.zeros((F_sub, B * E),
                                device=self.device, dtype=torch.float32)
        hess_hist = torch.zeros_like(grad_hist)

        if N > 0:
            # kernel expects contiguous inputs
            self.compute_histogram(
                bin_indices_sub.contiguous(),
                era_ids.contiguous(),
                gradients.contiguous(),
                grad_hist,
                hess_hist,
                B,
                E,
                self.threads_per_block,
                self.rows_per_thread
            )

        return grad_hist, hess_hist

  

    def find_best_split(self,
                    grad_hist_flat: torch.Tensor,
                    hess_hist_flat: torch.Tensor
                   ) -> Tuple[int, int]:
        """
        1) If E == 1, call compute_split kernel on [F, B] hists directly.
        2) Else, do the DES directional pass + early exits + tie‐breaker kernel.
        """
        import torch
        from warpgbm.cuda.node_kernel import compute_split

        F, BE = grad_hist_flat.shape
        B     = self.num_bins
        E     = BE // B

        #print("num_eras =", E)

        # --- special case: single era → skip DES, use kernel across all features
        if E == 1:
            G_sum = grad_hist_flat  # [F, B]
            H_sum = hess_hist_flat

            if not hasattr(self, "_best_gains") or self._best_gains.numel() < F:
                self._best_gains = torch.empty((F,), device=G_sum.device, dtype=torch.float32)
                self._best_bins  = torch.empty((F,), device=G_sum.device, dtype=torch.int32)

            threads = min(self.threads_per_block, F)
            compute_split(
                G_sum, H_sum,
                float(self.min_split_gain),
                float(self.min_child_weight),
                1e-6,
                self._best_gains,
                self._best_bins,
                threads
            )

            # if no valid split at all
            if torch.all(self._best_bins == -1):
                return -1, -1

            f = int(self._best_gains.argmax().item())
            b = int(self._best_bins[f].item())
            return f, b

        # --- otherwise: DES with directional pass + tie‐breaker ---
        G  = grad_hist_flat.view(F, E, B).transpose(1, 2)  # [F, B, E]
        H  = hess_hist_flat.view(F, E, B).transpose(1, 2)  # [F, B, E]
        CG = G.cumsum(dim=1)
        CH = H.cumsum(dim=1)

        GL = CG[:, :-1, :]   # [F, B-1, E]
        HL = CH[:, :-1, :]
        TG = CG[:,   -1, :]  # [F, E]
        TH = CH[:,   -1, :]
        GR = TG[:, None, :] - GL
        HR = TH[:, None, :] - HL

        valid = (
            (HL >= self.min_child_weight).all(dim=2)
          & (HR >= self.min_child_weight).all(dim=2)
        )  # [F, B-1]
        if not valid.any():
            return -1, -1

        dirs      = torch.sign(GL/(HL+1e-6) - GR/(HR+1e-6))  # [F, B-1, E]
        dir_score = dirs.sum(dim=2).abs().div(E)              # [F, B-1]
        dir_score = dir_score.masked_fill(~valid, float('-inf'))

        max_dir = dir_score.max()
        if max_dir == -float('inf'):
            return -1, -1

        per_feat_max = dir_score.max(dim=1).values  # [F]
        tie_feats    = (per_feat_max == max_dir).nonzero().view(-1)

        if tie_feats.numel() <= 1:
            if tie_feats.numel() == 0:
              return -1, -1
            f = int(tie_feats.item())
            b = int(dir_score[f].argmax().item())
            return f, b

        # collapse eras → [F, B]
        G_sum = G.sum(dim=2)
        H_sum = H.sum(dim=2)

        G_tie = G_sum[tie_feats]  # [K, B]
        H_tie = H_sum[tie_feats]
        K     = G_tie.size(0)

        if not hasattr(self, "_best_gains") or self._best_gains.numel() < K:
            self._best_gains = torch.empty((K,), device=G_tie.device, dtype=torch.float32)
            self._best_bins  = torch.empty((K,), device=G_tie.device, dtype=torch.int32)
        else:
            self._best_gains = self._best_gains[:K]
            self._best_bins  = self._best_bins[:K]

        threads = min(self.threads_per_block, K)
        compute_split(
            G_tie, H_tie,
            float(self.min_split_gain),
            float(self.min_child_weight),
            1e-6,
            self._best_gains,
            self._best_bins,
            threads
        )

        # if no valid split among tied features
        if torch.all(self._best_bins == -1):
            return -1, -1

        rel     = int(self._best_gains.argmax().item())
        f_final = int(tie_feats[rel].item())
        b_final = int(self._best_bins[rel].item())
        return f_final, b_final






    def grow_tree(self, gradient_histogram, hessian_histogram, node_indices, depth):
        # Terminal condition
        if depth == self.max_depth:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": node_indices.numel()}

        parent_size = node_indices.numel()

        # 1) find best feature & composite bin (in [0, B*E))
        local_feat_idx, comp_bin = self.find_best_split(
            gradient_histogram, hessian_histogram
        )
        if local_feat_idx == -1:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": parent_size}

        # map local index → global feature id
        feat_id = self.feat_indices_tree[local_feat_idx]
        B       = self.num_bins  # original bin count
        # 2) split mask: use composite code = era*B + raw_bin
        codes = self.bin_indices[node_indices, feat_id]
          
    
        split_mask = codes <= comp_bin

        left_indices  = node_indices[split_mask]
        right_indices = node_indices[~split_mask]
        left_size     = left_indices.numel()
        right_size    = right_indices.numel()

        # 3) rebuild the smaller side’s hist, infer the other by subtraction
        if left_size <= right_size:
            gradL, hessL = self.compute_histograms(
                self.bin_indices[left_indices][:, self.feat_indices_tree],
                self.residual[left_indices],
                self.era_indices[left_indices],
            )
            gradR = gradient_histogram - gradL
            hessR = hessian_histogram  - hessL
        else:
            gradR, hessR = self.compute_histograms(
                self.bin_indices[right_indices][:, self.feat_indices_tree],
                self.residual[right_indices],
                self.era_indices[right_indices],
            )
            gradL = gradient_histogram - gradR
            hessL = hessian_histogram  - hessR

        # 4) recurse
        left_child  = self.grow_tree(gradL, hessL, left_indices,  depth + 1)
        right_child = self.grow_tree(gradR, hessR, right_indices, depth + 1)

        return {
            "feature": feat_id,
            "bin":     comp_bin,  # composite in [0, B*E)
            "left":    left_child,
            "right":   right_child,
        }

    
    def get_eval_metric(self, y_true, y_pred):
        if self.eval_metric == "mse":
            return ((y_true - y_pred) ** 2).mean().item()
        elif self.eval_metric == "corr":
            return 1 - torch.corrcoef(torch.vstack([y_true, y_pred]))[0, 1].item()
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
        self.eval_loss = []
        self.stop = False

        # precompute number of eras
        E = int(self.era_indices.max().item()) + 1

        # adjust colsample logic
        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices

        for i in range(self.n_estimators):
            # 1) residual
            self.residual = self.Y_gpu - self.gradients

            # 2) feature subsample if needed
            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(
                    self.num_features, device=self.device
                )[:k]

            # 3) era-aware histogram via composite keys
            #    bin_sub: [N, F_sub]
            bin_sub = self.bin_indices[:, self.feat_indices_tree]

            # compute_histograms now takes era_ids and returns [F_sub, B*E]
            self.root_gradient_histogram, self.root_hessian_histogram = (
                self.compute_histograms(bin_sub, self.residual, self.era_indices)
            )

            # 4) grow tree as before (best_bins now in [0, B*E))
            tree = self.grow_tree(
                self.root_gradient_histogram,
                self.root_hessian_histogram,
                self.root_node_indices,
                depth=0,
            )
            self.forest[i] = tree

            # 5) eval / early stop
            self.compute_eval(i)
            if self.stop:
                break

        print("Finished training forest.")


    def bin_data_with_existing_edges(self, X_np):
        X_tensor = torch.from_numpy(X_np).type(torch.float32).pin_memory()
        num_samples = X_tensor.size(0)
        bin_indices = torch.zeros(
            (num_samples, self.num_features), dtype=torch.int8, device=self.device
        )
        with torch.no_grad():
            for f in range(self.num_features):
                X_f = X_tensor[:, f].to(self.device, non_blocking=True)
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
            bin_indices = (
                torch.from_numpy(X_np).to(self.device).contiguous().to(torch.int8)
            )
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