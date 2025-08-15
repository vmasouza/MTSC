import numpy as np
import pandas as pd
import time

from tslearn.barycenters import \
    euclidean_barycenter, \
    dtw_barycenter_averaging, \
    softdtw_barycenter

from collections import Counter
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from aeon.classification.convolution_based import RocketClassifier 

from aeon.datasets.tsc_datasets import multivariate
from aeon.datasets import load_classification


def fusion_dimensions(data, operation, quantile_value=0.25):
    if operation == 'sum':
        fused = np.sum(data, axis=1)

    elif operation == 'mean':
        fused = np.mean(data, axis=1)

    elif operation == 'median':
        fused = np.median(data, axis=1)

    elif operation == 'variance':
        fused = np.var(data, axis=1)

    elif operation == 'quantile':
        fused = np.quantile(data, quantile_value, axis=1)

    elif operation == 'max':
        fused = np.max(data, axis=1)

    elif operation == 'geometric_mean':
        fused = np.exp(np.mean(np.log(np.abs(data) + 1e-9), axis=1)) * np.sign(np.prod(data, axis=1))

    elif operation == 'euclidean_barycenter':
        result = [euclidean_barycenter(data[i]) for i in range(data.shape[0])]
        fused = np.squeeze(np.array(result), axis=-1)

    elif operation == 'dtw_barycenter':
        result = [dtw_barycenter_averaging(data[i], max_iter=50, tol=1e-3)
                  for i in range(data.shape[0])]
        fused = np.squeeze(np.array(result), axis=-1)

    elif operation == 'softdtw_barycenter':
        result = [softdtw_barycenter(data[i], gamma=1., max_iter=50, tol=1e-3)
                  for i in range(data.shape[0])]
        fused = np.squeeze(np.array(result), axis=-1)

    else:
        raise ValueError(f"Unknown operation: {operation}")

    #fused = zscore_norm(fused, axis=1)
    return fused


if __name__ == "__main__":

    
    fusion_operations = ["sum", "mean", "median", "variance", "quantile", "max", "geometric_mean", "euclidean_barycenter", "dtw_barycenter", "softdtw_barycenter"]
    results = []
    for dataset in multivariate:
        # loading dataset
        X_train, y_train = load_classification(dataset, split="Train")
        X_test, y_test = load_classification(dataset, split="Test")
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test  = le.transform(y_test)

        for operation in fusion_operations:
            start_time = time.time()
            X_train_fused = fusion_dimensions(X_train, operation)
            X_test_fused = fusion_dimensions(X_test, operation)
            clf = RocketClassifier(n_kernels=1000)
            clf.fit(X_train_fused, y_train)
            y_pred = clf.predict(X_test_fused)

            end_time = time.time()

            acc = accuracy_score(y_test, y_pred)
            elapsed = end_time - start_time        

            results.append({
                "Dataset": dataset,
                "Fusion": operation,
                "Accuracy": acc,
                "Time in seconds": elapsed
            })
    df_results = pd.DataFrame(results)
    df_results.to_csv("fusion.csv", index=False)