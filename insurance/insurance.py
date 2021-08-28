import pandas as pd
# =============================================================================
# Import data into python in a data frame format
# =============================================================================
import matplotlib.pylab as plt
# =============================================================================
# data visuvalization library
# =============================================================================

df1 = pd.read_csv("/content/drive/MyDrive/AutoInsurance.csv")
df1.head()
df1.info()
# removing unwanted columns
df=df1.copy()
df2=df.drop(['Marital Status','Customer','State','Education','Renew Offer Type','Policy','Policy Type'],axis=1)
df_cat=[col for col in df2.columns if df2[col].dtype=="O"]
df_cat=df2[df_cat].drop(['Effective To Date'],axis=1)
# encoding
df_cat=pd.get_dummies(df_cat,drop_first=True)
num=df._get_numeric_data()
#scaling using Normalization
from sklearn.preprocessing import MinMaxScaler
MS=MinMaxScaler()
num=pd.DataFrame(MS.fit_transform(num),columns=num.columns)

Final=pd.concat([df_cat,num],axis=1)

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as dendo
z = linkage(Final, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
#from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(Final) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df1['clust'] = cluster_labels # creating a new column and assigning it to new column 

#df1 = df.iloc[:, [10,0,1,2,3,4,5,6,7,8,9]]
#df1.head()

# Aggregate mean of each cluster
df1.iloc[ :, 0:].groupby(df1.clust).mean()

# creating a csv file 
df1.to_csv("insurance.xlsx", encoding = "utf-8")

import os
os.getcwd()


