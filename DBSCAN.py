import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("path/to/datasert.csv/here")

selected_features = [' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', ' Fwd Packet Length Mean', ' Bwd Packet Length Mean', ' Flow IAT Mean']
df = df[selected_features].dropna()

df = df.head(10000)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

dbscan = DBSCAN(eps=1, min_samples=4)
df['Cluster'] = dbscan.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Step 6: Visualize clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='PCA1', y='PCA2', 
    hue='Cluster', 
    palette='viridis', 
    data=df, 
    legend='full', 
    s=60
)
plt.title('DBSCAN Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()
