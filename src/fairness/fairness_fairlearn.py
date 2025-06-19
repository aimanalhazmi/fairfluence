from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, recall_score


class FairnessEvaluator:
    def __init__(self, y_true, y_pred, sensitive_features):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sensitive_features = sensitive_features
        self.frame = None

    def evaluate(self):
        self.frame = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "recall": recall_score,
                "demographic_parity_diff": demographic_parity_difference,
                "equalized_odds_diff": equalized_odds_difference
            },
            y_true=self.y_true,
            y_pred=self.y_pred,
            sensitive_features=self.sensitive_features
        )
        return self.frame

    def report(self):
        if self.frame is None:
            raise ValueError("You must call .evaluate() before .report()")
        print("=== Fairness metrics by group ===")
        print(self.frame.by_group)
        print("\n=== Overall disparities ===")
        print("Demographic Parity Difference:", self.frame.overall["demographic_parity_diff"])
        print("Equalized Odds Difference:", self.frame.overall["equalized_odds_diff"])