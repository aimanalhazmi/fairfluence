import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

# Temporary, to import influence 
import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from influence.logistic_influence import LogisticInfluence
from ingestion.ingestion import IngestorFactory

from pyod.models.knn import KNN
from pyod.models.kpca import KPCA

K_PERCENTAGE = 0.05 # How many points we want to select
N_SAMPLE = 1000 # Data points in the mock dataset
MISLABELING = 0.02 # Mislabeled data, theoretically highly negative inflkuence
MAX_ITER = 100 # Model iterations

link = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
ingestor_factory = IngestorFactory(link, 0)
ingestor = ingestor_factory.create()
dataframe = ingestor.load_data()

# Mock dataset with 2 classes, 2% of mislabeled data
X, y = make_classification(
    n_samples=N_SAMPLE,
    n_features=4,         
    n_informative=2,
    n_redundant=1,
    n_repeated=0,
    n_clusters_per_class=1,
    class_sep=0.9,        
    flip_y=MISLABELING,          
    random_state=912
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

n_train = X_train.shape[0]
k = int(K_PERCENTAGE * n_train) 

# Train baseline model
baseline = LogisticRegression(max_iter=MAX_ITER)
baseline.fit(X_train, y_train)
print("Baseline test accuracy:", baseline.score(X_test, y_test))

# Get k outliers with KNN
knn = KNN()
knn.fit(X_train)
knn_scores = knn.decision_scores_ 
knn_outliers = np.argsort(knn_scores)[-k:]  # indices of top k


# Get k outliers with KernelPCA
kpca = KPCA()
kpca.fit(X_train)
kpca_scores = kpca.decision_scores_
kpca_outliers = np.argsort(kpca_scores)[-k:]

# Get k negative influent points
model = LogisticRegression(max_iter=MAX_ITER).fit(X_train, y_train)
influencer = LogisticInfluence(model, X_train, y_train)
avg_inf = influencer.average_influence(X_test, y_test)
neg_influencers = np.argsort(avg_inf)[-k:]


# Get k random points
rng = np.random.RandomState(912)
random_idxs = rng.choice(n_train, size=k, replace=False)

# Small XAI, get PCA on train data, then plot the classes marking the selected points by each procedure.
X2 = PCA(n_components=2, random_state=912).fit_transform(X_train)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X2[:, 0], X2[:, 1],
    c=y_train, cmap='coolwarm', alpha=0.6
)

plt.scatter(X2[neg_influencers, 0], X2[neg_influencers, 1], facecolors='none', edgecolors='red', s=100, label='Negative influence points')
plt.scatter(X2[knn_outliers, 0], X2[knn_outliers, 1],facecolors='none', edgecolors='blue', s=100, label='KNN Outliers')
plt.scatter(X2[kpca_outliers, 0], X2[kpca_outliers, 1],facecolors='none', edgecolors='green', s=100, label='KPCA Outliers')
plt.scatter(X2[random_idxs, 0], X2[random_idxs, 1],facecolors='none', edgecolors='black', s=100, label='Random K Points')

handles, labels = scatter.legend_elements(prop="colors")
plt.legend(
    handles + [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='none', markeredgecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='none', markeredgecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='none', markeredgecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='none', markeredgecolor='black', markersize=10)
    ],
    labels + ['Neg Influence', 'KNN Outliers', 'KPCA Outliers', 'Random k'],
    title="Classes / Flags"
)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Training Data: Classes, Negative Influent Points & KNN Outliers')

plt.show()

# Retrain the model without the k points, and recalculate accuracy
def retrain_and_score(removal_idxs, label):
    mask = np.ones(n_train, dtype=bool)
    mask[removal_idxs] = False
    X_sub, y_sub = X_train[mask], y_train[mask]
    clf = LogisticRegression(max_iter=MAX_ITER).fit(X_sub, y_sub)
    acc = clf.score(X_test, y_test)
    print(f"{label} removal: test acc = {acc:.4f}")
    return acc

retrain_and_score(knn_outliers, "KNN outliers")
retrain_and_score(kpca_outliers, "KPCA outliers")
retrain_and_score(neg_influencers,"Influence")
retrain_and_score(random_idxs, "Random")

# Experiment sweeping k from 0.5% to 20% and seeing how removing this many points affects the loss. So, like
# we remove k points with each method and recalculate loss, varying k.
def retrain_and_logloss(removal_idxs, label):
    mask = np.ones(n_train, dtype=bool)
    mask[removal_idxs] = False
    X_sub, y_sub = X_train[mask], y_train[mask]
    clf = LogisticRegression(max_iter=MAX_ITER).fit(X_sub, y_sub)
    probs = clf.predict_proba(X_test)
    loss = log_loss(y_test, probs)
    return loss

steps = np.linspace(0.005, 0.2, 100)
results = {'influence': [], 'knn': [], 'random': []}

for frac in steps:
    k = max(1, int(frac * n_train))
    idx_inf = np.argsort(avg_inf)[-k:]
    idx_knn = np.argsort(KNN().fit(X_train).decision_scores_)[-k:]
    idx_rnd = rng.choice(n_train, size=k, replace=False)

    results['influence'].append(retrain_and_logloss(idx_inf, f"Inf {frac:.3f}"))
    results['knn'].append(retrain_and_logloss(idx_knn, f"KNN {frac:.3f}"))
    results['random'].append(retrain_and_logloss(idx_rnd, "Rnd {frac:.3f}"))

plt.figure(figsize=(8, 5))
plt.plot(steps, results['influence'], marker='o', label='Influence')
plt.plot(steps, results['knn'], marker='s', label='KNN')
plt.plot(steps, results['random'], marker='^', label='Random')
plt.xlabel('Removal fraction k')
plt.ylabel('Test Log Loss')
plt.title('Effect of Removing k Points on Test Log Loss')
plt.legend()
plt.show()
