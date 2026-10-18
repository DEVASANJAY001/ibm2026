"""
ml/bundle.py
────────────
Container bundle for trained classifier and target class names.
"""

import numpy as np


class FertilizerModelBundle:
    def __init__(self, pipeline, classes):
        self.pipeline = pipeline
        self.classes_ = np.array(classes)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
