## profiling dataset by searching for negative influence
import numpy as np
import pandas as pd
from src.influence import logistic_influence, LogisticInfluence


class InfluenceOutlierDetector:
    def __init__(self, model, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.influencer = LogisticInfluence(model, X_train, y_train)

    def detect(self, X_test, y_test, strategy='avg', top_k=5):
        if strategy == 'avg':
            influences = self.influencer.average_influence(X_test, y_test)
        else:
            raise ValueError("Unknown strategy")

        print(influences.shape)
        top_k_idx = np.argsort(influences)[-top_k:]
        return top_k_idx, influences[top_k_idx]