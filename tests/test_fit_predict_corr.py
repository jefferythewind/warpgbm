import numpy as np
from warpgbm import WarpGBM
from sklearn.datasets import make_regression

import numpy as np
import time
from warpgbm import WarpGBM
from sklearn.datasets import make_regression

def test_fit_predictpytee_correlation():
    np.random.seed(42)
    N = 100_000
    F = 1000
    X, y = make_regression(n_samples=N, n_features=F, noise=0.1, random_state=42)
    era = np.zeros(N, dtype=np.int32)
    corrs = []

    for hist_type in ['hist1', 'hist2', 'hist3']:
        print(f"\nTesting histogram method: {hist_type}")

        model = WarpGBM(
            max_depth=10,
            num_bins=10,
            n_estimators=10,
            learning_rate=1,
            verbosity=False,
            histogram_computer=hist_type,
            threads_per_block=64,
            rows_per_thread=4
        )

        start_fit = time.time()
        model.fit(X, y, era_id=era)
        fit_time = time.time() - start_fit
        print(f"  Fit time:     {fit_time:.3f} seconds")

        start_pred = time.time()
        preds = model.predict(X)
        pred_time = time.time() - start_pred
        print(f"  Predict time: {pred_time:.3f} seconds")

        corr = np.corrcoef(preds, y)[0, 1]
        print(f"  Correlation:  {corr:.4f}")
        corrs.append(corr)

    assert (np.array(corrs) > 0.95).all(), f"In-sample correlation too low: {corrs}"
