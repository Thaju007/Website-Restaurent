Bro can you upload this on your github and share me on teams or something
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# -----------------------------
# STEP 1: Load & Drop Redundant Columns
# -----------------------------
df = pd.read_csv("customer_data.csv")

drop_cols = ["arguspermid", "comsigext_id"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# -----------------------------
# STEP 2: Filter Variables
# -----------------------------

# 2A: Drop zero or near-zero variance
nzv = df.var()
df = df.loc[:, nzv > 0.001]

# 2B: Drop sparse binary columns (>98% zeros)
sparse_cols = [c for c in df.columns 
               if df[c].value_counts(normalize=True).iloc[0] > 0.98]
df = df.drop(columns=sparse_cols)

# 2C: Drop duplicate columns
df = df.T.drop_duplicates().T

# 2D: Drop highly correlated variables
corr = df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_correlated = [col for col in upper.columns if any(upper[col] > 0.85)]
df = df.drop(columns=drop_correlated)

print("Remaining columns after filtering:", df.shape)

# -----------------------------
# STEP 3: Feature Selection (Top 10 Features)
# -----------------------------

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(df)

# 3A: PCA for unsupervised variance importances
pca = PCA(n_components=20)
pca.fit(X)
loadings = pd.DataFrame(np.abs(pca.components_), columns=df.columns)

pca_importance = loadings.sum().sort_values(ascending=False)

# 3B: MiniBatch KMeans for initial cluster labels
kmeans_init = MiniBatchKMeans(n_clusters=8, batch_size=5000)
clusters_init = kmeans_init.fit_predict(X)

# 3C: RandomForest to rank features
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X, clusters_init)

rf_importance = pd.Series(rf.feature_importances_, index=df.columns)

# Combine PCA + RF importance
combined_importance = (
    0.6 * rf_importance.rank(ascending=False) +
    0.4 * pca_importance.rank(ascending=False)
).sort_values()

top_features = combined_importance.head(10).index.tolist()
print("Top 10 selected features:")
print(top_features)

X_sel = df[top_features]
X_sel_scaled = scaler.fit_transform(X_sel)

# -----------------------------
# STEP 4: Pick Best K Using Metrics
# -----------------------------

results = []
for k in range(3, 15):
    km = MiniBatchKMeans(n_clusters=k, batch_size=5000)
    labels = km.fit_predict(X_sel_scaled)
    
    sil = silhouette_score(X_sel_scaled, labels)
    dbi = davies_bouldin_score(X_sel_scaled, labels)
    ch = calinski_harabasz_score(X_sel_scaled, labels)
    
    results.append((k, sil, dbi, ch))

results_df = pd.DataFrame(results, columns=["k", "silhouette", "davies_bouldin", "calinski_harabasz"])
print(results_df)

# Pick best K (maximize silhouette, CH; minimize DBI)
best_k = results_df.sort_values(
    by=["silhouette", "calinski_harabasz", "davies_bouldin"],
    ascending=[False, False, True]
).iloc[0]["k"]

print(f"Best number of clusters: {best_k}")

# Final Model
final_kmeans = MiniBatchKMeans(n_clusters=int(best_k), batch_size=5000)
final_labels = final_kmeans.fit_predict(X_sel_scaled)

df["cluster"] = final_labels

# -----------------------------
# STEP 5: PRIZM-Style Cluster Naming
# -----------------------------
cluster_summary = df.groupby("cluster")[top_features].mean()

def name_cluster(values):
    name_parts = []
    
    if values.max() > values.mean() + values.std():
        strong_feature = values.idxmax()
        name_parts.append(strong_feature.replace("_", " ").title())
    
    if len(name_parts) == 0:
        return "General Segment"
    return " | ".join(name_parts)

cluster_names = cluster_summary.apply(name_cluster, axis=1)
cluster_summary["segment_name"] = cluster_names

print(cluster_summary)
