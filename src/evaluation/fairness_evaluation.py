import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)
from scipy.stats import spearmanr

# Temporary, to import influence 
import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)
from influence.logistic_influence import LogisticInfluence

# Load dataset, this we would later do with our own module
df = pd.read_csv("/Users/lulo/Documents/Facultad/TU Berlin/EDML/project/ml-data-profiler/src/unfair_data.csv")  
X = df[[f"feature_{i}" for i in range(5)]].values
y = df["target"].values
sensitive = df["A"].values # Does it have to be hardcoded?

# Split into train/test, also keeping indices and sensitive atribute
all_indices = np.arange(len(y))
X_train, X_test, y_train, y_test, s_train, s_test, train_idx, test_idx = train_test_split(
    X, y, sensitive, all_indices,
    test_size=0.2,
    stratify=y,
    random_state=0
)
n_train = X_train.shape[0]

# Train the logistic regression model
model = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# Compute baseline fairness (DPD & EOD) on the test set using fairlearn
y_pred = model.predict(X_test)
dpd_orig = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=s_test)
eod_orig = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=s_test)
print(f"Baseline DPD: {dpd_orig:.4f}")
print(f"Baseline EOD: {eod_orig:.4f}")

# Using the Influence Module, calculate per-example average influence
influencer = LogisticInfluence(model, X_train, y_train)
avg_inf = influencer.average_influence(X_test, y_test)

# print(avg_inf)

# Construct DataFrame for training set with each point avg_inf
df_train = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(X_train.shape[1])])
df_train["y"] = y_train
df_train["sensitive"] = s_train
df_train["avg_inf"] = avg_inf

# Group by sensitive attribute and compute mean/std of influence
group_stats = df_train.groupby("sensitive")["avg_inf"].agg(mean="mean", std="std", count="count")
print("\nInfluence by group:")
print(group_stats)

# Identify group with the most harmful points on average (most disadvantaged group)
disadvantaged = group_stats["mean"].idxmax()
print(f"\nDisadvantaged group (most harmful avg_inf): {disadvantaged}")

# TO DO: --- VALIDATE AGAINST GROUND-TRUTH FLIPS ---
# Note: since your unfair dataset had multiple manipulations (DPD and EOD),
# you may need to adapt the ground-truth flip logic accordingly.
# If you did record which samples were flipped, you can follow the same structure as before.

# Remove top-k harmful points from disadvantaged group
k = max(1, int(0.05 * group_stats.loc[disadvantaged, "count"]))  # 5% of that group
to_remove = (df_train[df_train["sensitive"] == disadvantaged].nlargest(k, "avg_inf").index)

#print(df_train[df_train["sensitive"] == disadvantaged].nsmallest(k, "avg_inf"))

mask = ~df_train.index.isin(to_remove)
X_train2 = X_train[mask]
y_train2 = y_train[mask]
s_train2 = s_train[mask]

# Retrain the model without those points
model2 = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000)
model2.fit(X_train2, y_train2)

# Calculate fairness again on the same test set
y_pred2 = model2.predict(X_test)
dpd_new = demographic_parity_difference(y_true=y_test, y_pred=y_pred2, sensitive_features=s_test)
eod_new = equalized_odds_difference(y_true=y_test, y_pred=y_pred2, sensitive_features=s_test)

print(f"\nAfter removing top {k} harmful points from group {disadvantaged}:")
print(f"  New DPD: {dpd_new:.4f}")
print(f"  New EOD: {eod_new:.4f}")




