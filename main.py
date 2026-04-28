import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (JSON properly)
df = pd.read_json(r'C:/Users/noree/Desktop/unsupervised learning/ecommerce-data-metadata.json')

print("Shape:", df.shape)
print(df.head())

# If JSON is nested → flatten
if isinstance(df.iloc[0], dict):
    df = pd.json_normalize(df)

# Clean data
df = df.dropna()
df = df.drop_duplicates()

# Select numeric features
features = df.select_dtypes(include=[np.number])

# Check if empty
if features.empty:
    raise ValueError("No numeric columns found!")

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

print("Data Preprocessed ✅")

# -------- KMeans --------
inertia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Elbow Plot
plt.plot(K, inertia, 'bx-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

df['Cluster'] = clusters

# Visualization
if features.shape[1] >= 2:
    sns.scatterplot(x=features.iloc[:,0], y=features.iloc[:,1], hue=clusters)
    plt.title("Customer Segments")
    plt.show()

print(df.groupby("Cluster").mean())

# -------- Anomaly Detection --------
kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
kde.fit(scaled_data)

scores = kde.score_samples(scaled_data)

threshold = np.percentile(scores, 5)
anomalies = scores < threshold

df['Anomaly'] = anomalies

plt.scatter(features.iloc[:,0], features.iloc[:,1], c=anomalies)
plt.title("Anomaly Detection")
plt.show()

print("Anomalies Found:", df['Anomaly'].sum())

# -------- PCA --------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

print("Explained Variance:", pca.explained_variance_ratio_)

plt.scatter(pca_data[:,0], pca_data[:,1], c=clusters)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization")
plt.show()

# -------- Recommendation System --------

# Make sure columns exist first
if {'CustomerID','Spending_Score','Annual_Income'}.issubset(df.columns):

    user_item = df.pivot_table(index='CustomerID',
                               columns='Spending_Score',
                               values='Annual_Income',
                               fill_value=0)

    similarity = cosine_similarity(user_item)

    def recommend(user_id, top_n=3):
        user_index = user_item.index.tolist().index(user_id)
        similar_users = similarity[user_index]

        similar_indices = np.argsort(similar_users)[::-1][1:top_n+1]
        return user_item.index[similar_indices]

    print("Recommendations:", recommend(user_item.index[0]))

else:
    print("Recommendation system skipped (columns not found)")