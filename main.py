import pandas as pd
import numpy as np
from scipy.stats import logistic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.fairness.fairness_influence import evaluate_patterns
from src.influence import LogisticInfluence
from src.profiling.empty_detection import empty_detection
from src.profiling.influence_detection import InfluenceOutlierDetector


def pattern_to_readable(pattern, columns):
    readable = []
    for col_idx, val in pattern.items():
        col_name = columns[col_idx] if isinstance(col_idx, int) else col_idx
        if isinstance(val, pd.Interval):
            readable.append(f"{col_name} ∈ {val}")
        else:
            readable.append(f"{col_name} = {val}")
    return readable


# === 1. read data ===
df = pd.read_csv('adult.csv')

# === 2. Data cleaning ===
df = df.replace('?', np.nan).dropna()

# select attribute
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']


# labeling data
y = (df['income'] == '>50K').astype(int).values

# scaling
scaler = StandardScaler()
X_numerical = scaler.fit_transform(df[numerical_cols])
X_index = df.index
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X_numerical, y, X_index, test_size=0.2, random_state=42)


# === 3. Missing Value detection ===
print("\n==== Missing Value Detection ====")
mv_report = empty_detection().report(df)
print(mv_report)


# === 5. influence Function detection ===
print("\n==== influence Function Detection ====")
model = LogisticRegression(max_iter=500).fit(X_train, y_train)
infl_detector = InfluenceOutlierDetector(model, X_train, y_train)
infl = infl_detector.detect(X_test[:5], y_test[:5])

# === 6. influence Function on fairness ===
logistic_influence = LogisticInfluence(model, X_train, y_train)
X_train_infl = logistic_influence.average_influence(X_test[:5], y_test[:5])

X_train_raw = df.loc[train_index].copy().reset_index(drop=True)
print(X_train_raw)
X_train_raw['influence'] = X_train_infl
X_train_raw = X_train_raw.drop(columns=["income"])

top_patterns = evaluate_patterns(X_train_raw, min_support=0.05, top_k=10)
for i, p in enumerate(top_patterns):

    print(f"Pattern {i + 1}:")
    readable = pattern_to_readable(p['pattern'], df.columns)
    for cond in readable:
        print("  -", cond)
    print(
        f"  Support: {p['support']:.2%}, Responsibility: {p['responsibility']:.3f}, Interestingness: {p['interestingness']:.3f}")