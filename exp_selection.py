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


        # max number of folds according to the number of classes in training
        _, counts = np.unique(y_train, return_counts=True)
        max_possible_folds = counts.min()
        n_splits = min(10, max_possible_folds)

        #print(f"Using {n_splits} folds")

        n_channels = X_train.shape[1]
        cv_scores = []

        # train a classifier for each channel
        start_time = time.time()
        for ch in range(n_channels):
            clf = RocketClassifier(n_kernels=1000)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X_train[:, ch, :], y_train, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            cv_scores.append(mean_score)
            #print(f"Channel {ch}: mean CV = {mean_score:.4f}")

        # choose only the channel that had the highest result in CV (train)
        best_channel = np.argmax(cv_scores)
        #print(f"Best channel: {best_channel} (mean CV = {cv_scores[best_channel]:.4f})")

        best_clf = RocketClassifier(n_kernels=1000)
        best_clf.fit(X_train[:, best_channel, :], y_train)
        y_pred = best_clf.predict(X_test[:, best_channel, :])
        end_time = time.time()

        acc = accuracy_score(y_test, y_pred)
        elapsed = end_time - start_time        

        results.at[idx, 'Accuracy'] = acc
        results.at[idx, 'Time in seconds'] = elapsed
    results.to_csv("selection.csv", index=False)


