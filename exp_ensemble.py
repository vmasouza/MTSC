import numpy as np
import pandas as pd
import time

from collections import Counter
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from aeon.classification.convolution_based import RocketClassifier 

from aeon.datasets.tsc_datasets import multivariate
from aeon.datasets import load_classification


if __name__ == "__main__":

    results = pd.DataFrame(columns=['Dataset', 'Accuracy', 'Time in seconds'])
    results['Dataset'] = multivariate

    for idx, dataset in enumerate(multivariate):
        # loading dataset
        X_train, y_train = load_classification(dataset, split="Train")
        X_test, y_test = load_classification(dataset, split="Test")
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test  = le.transform(y_test)

        
        n_channels = X_train.shape[1]
        n_samples_test = X_test.shape[0]
        
        # train a classifier for each channel
        start_time = time.time()
        classifiers = [RocketClassifier(n_kernels=1000) for _ in range(n_channels)]
        for ch in range(n_channels):
            classifiers[ch].fit(X_train[:, ch, :], y_train)

        # obtain predictions of each classifier trained by channel
        predictions = np.zeros((n_samples_test, n_channels), dtype=int)
        for ch in range(n_channels):
            predictions[:, ch] = classifiers[ch].predict(X_test[:, ch, :])

        # majority voting
        y_pred = []
        for i in range(n_samples_test):
            counts = Counter(predictions[i])
            y_pred.append(counts.most_common(1)[0][0])
        y_pred = np.array(y_pred)
        end_time = time.time()
        
        elapsed = end_time - start_time
        acc = accuracy_score(y_test, y_pred)

        results.at[idx, 'Accuracy'] = acc
        results.at[idx, 'Time in seconds'] = elapsed
    results.to_csv("ensemble.csv", index=False)


