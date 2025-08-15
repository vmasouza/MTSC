import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from aeon.classification.convolution_based import RocketClassifier 

from aeon.datasets.tsc_datasets import multivariate
from aeon.datasets import load_classification


if __name__ == "__main__":

    results = pd.DataFrame(columns=['Dataset', 'Accuracy', 'Time in seconds'])
    results['Dataset'] = multivariate

    for i, dataset in enumerate(multivariate):
        # loading dataset
        X_train, y_train = load_classification(dataset, split="Train")
        X_test, y_test = load_classification(dataset, split="Test")
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test  = le.transform(y_test)

        start_time = time.time()
        # concatenating channels
        n_samples, n_channels, series_length = X_train.shape
        X_train_flat = X_train.reshape(n_samples, n_channels * series_length)
        X_test_flat  = X_test.reshape(X_test.shape[0], n_channels * series_length)

        # train classifier
        clf = RocketClassifier(n_kernels=1000)
        clf.fit(X_train_flat, y_train)

        # test phase
        y_pred = clf.predict(X_test_flat)
        end_time = time.time()

        acc = accuracy_score(y_test, y_pred)
        elapsed = end_time - start_time

        results.at[i, 'Accuracy'] = acc
        results.at[i, 'Time in seconds'] = elapsed
    results.to_csv("concatenation.csv", index=False)


