from collections import defaultdict
from ipywidgets import interactive
import hdbscan
import folium
import re
import matplotlib
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier


cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
        '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
        '#000075', '#808080']*10


df = pd.read_csv('taxi_data.csv')
df.head()

df.duplicated(subset=['LON', 'LAT']).values.any()

df.isna().values.any()

print(f'Before (Nulls and Duplicates) \t:\tdf.shape = {df.shape}')
df.dropna(inplace=True)
df.drop_duplicates(subset=['LON','LAT'],keep ='first', inplace=True)
print(f'After (Nulls and Duplicates) \t:\tdf.shape = {df.shape}')

df.head()

X = np.array(df[['LON', 'LAT']], dtype='float64')

plt.scatter(X[:,0], X[:,1], alpha=0.2, s=50)

m = folium.Map(location=[df.LAT.mean(), df.LON.mean()], zoom_start=9,
               tiles='Stamen Toner')

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row.LAT, row.LON],
        radius=5,
        popup=re.sub(r'[^a-zA-Z ]+', '', row.NAME),
        color='#1787FE',
        fill=True,
        fill_colour='#1787FE'
    ).add_to(m)


X_blobs,_=make_blobs(n_samples=1000,centers=10,
                    n_features=2,cluster_std=0.5,random_state=4)
plt.scatter(X_blobs[:,0],X_blobs[:,1],alpha=0.2)

class_predictions = np.load('Data/sample_clusters.npy')
unique_clusters = np.unique(class_predictions)
for unique_clusters in unique_clusters:
    X=X_blobs[class_predictions==unique_clusters]
    plt.scatter(X[:,0],X[:,1],alpha=0.2,c=cols[unique_clusters])

silhouette_score(X_blobs, class_predictions)
class_predictions = np.load('Data/sample_clusters_improved.npy')
unique_clusters = np.unique(class_predictions)
for unique_cluster in unique_clusters:
    X = X_blobs[class_predictions==unique_cluster]
    plt.scatter(X[:,0], X[:,1], alpha=0.2, c=cols[unique_cluster])

silhouette_score(X_blobs, class_predictions)

X_blobs, _ = make_blobs(n_samples=1000, centers=50, n_features=2, cluster_std=1, random_state=4)

data = defaultdict(dict)
for x in range(1,21):
    model = KMeans(n_clusters=3, random_state=17, max_iter=x, n_init=1).fit(X_blobs)
    data[x]['class_predictions'] = model.predict(X_blobs)
    data[x]['centroids'] = model.cluster_centers_
    data[x]['unique_classes'] = np.unique(class_predictions)

X=np.array(df[['LON','LAT']],dtype='float64')
k=70
model = KMeans(n_clusters=k,random_state=17).fit(X)
class_predictions=model.predict(X)
df[f'CLUSTER_kmeans{k}'] = class_predictions
df.head()

def create_map(df, cluster_column):
    m = folium.Map(location=[df.LAT.mean(), df.LON.mean()], zoom_start=9, tiles='Stamen Toner')

    for _, row in df.iterrows():

        if row[cluster_column] == -1:
            cluster_colour = '#000000'
        else:
            cluster_colour = cols[row[cluster_column]]

        folium.CircleMarker(
            location= [row['LAT'], row['LON']],
            radius=5,
            popup= row[cluster_column],
            color=cluster_colour,
            fill=True,
            fill_color=cluster_colour
        ).add_to(m)

    return m

m = create_map(df, 'CLUSTER_kmeans70')
print(f'K={k}')
print(f'Silhouette Score: {silhouette_score(X, class_predictions)}')

m.save('kmeans_70.html')


best_silhouette, best_k = -1, 0

for k in tqdm(range(2, 100)):
    model = KMeans(n_clusters=k, random_state=1).fit(X)
    class_predictions = model.predict(X)

    curr_silhouette = silhouette_score(X, class_predictions)
    if curr_silhouette > best_silhouette:
        best_k = k
        best_silhouette = curr_silhouette

print(f'K={best_k}')
print(f'Silhouette Score: {best_silhouette}')


dummy = np.array([-1, -1, -1, 2, 3, 4, 5, -1])

new = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(dummy)])

model = DBSCAN(eps=0.01, min_samples=5).fit(X)
class_predictions = model.labels_

df['CLUSTERS_DBSCAN'] = class_predictions

m = create_map(df, 'CLUSTERS_DBSCAN')


print(f'Number of clusters found: {len(np.unique(class_predictions))}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = 0
no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')

model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2,
                        cluster_selection_epsilon=0.01)


class_predictions = model.fit_predict(X)
df['CLUSTER_HDBSCAN'] = class_predictions

m = create_map(df, 'CLUSTER_HDBSCAN')

print(f'Number of clusters found: {len(np.unique(class_predictions))-1}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')

m

classifier=KNeighborsClassifier(n_neighbors=1)
df_train=df[df.CLUSTER_HDBSCAN!=-1]
df_predict=df[df.CLUSTER_HDBSCAN ==-1]
x_train = np.array(df_train[['LON','LAT']],dtype='float64')
y_train = np.array(df_train['CLUSTER_HDBSCAN'])
X_predict = np.array(df_predict[['LON','LAT']],dtype='float64')
classifier.fit(x_train,y_train)
predictions = classifier.predict(X_predict)
df['CLUSTER_hybrid'] = df['CLUSTER_HDBSCAN']
df.loc[df.CLUSTER_HDBSCAN==-1,'CLUSTER_hybrid'] = predictions
m= create_map(df,'CLUSTER_hybrid')
m

class_predictions=df.CLUSTER_hybrid
print(f'Number of clusters found: {len(np.unique(class_predictions))}')
print(f'Silhouette: {silhouette_score(X, class_predictions)}')
m.save('hybrid.html')

df['CLUSTER_hybrid'].value_counts().plot.hist(bins=70,
                                              alpha=0.4,label='Hybrid')
df['CLUSTER_kmeans70'].value_counts().plot.hist(bins=70,
                                alpha=0.4,label='K-Means(70)')
plt.legend()
plt.xlabel('Cluster Sizes')
