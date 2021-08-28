import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
df=pd.ExcelFile('F:/vrenv/Assignments/Hierarchical_clust_clustering/assignment/solution/New folder/EastWestAirlines.xlsx')
df1=pd.read_excel(df,'data')
df2=df1.drop(['ID#','Award?'],axis=1) # This Data is Not important. Because these columns are not putting any effect on our dataset.
#df2.info() # there is no null value

#  Normailzation to scale values between 0 to 1.
norm=MinMaxScaler()
df2_norm=pd.DataFrame(norm.fit_transform(df2),columns=df2.columns)

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as dendo
import matplotlib.pyplot as plt
linkages=['complete','single','ward','average','centroid'] 
for i in linkages:
  link = linkage(df2_norm, method = i, metric = "euclidean")
  plt.figure(figsize=(10,8))
  plt.title('Dendogram')
  plt.xlabel('Index')
  plt.ylabel('Distance')
  dendo.dendrogram(link,leaf_font_size=10,leaf_rotation=0) # Using Dendogram for this dataset its not possible to find best number of cluster So i did some research so found one more way that was Seoulette Score.
  plt.show()


# If dendogram is not working in cluster selection then we can also try silhouette score.

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
clusters=[3,4,5,6,7]
linkages=['complete','single','ward','average','single'] 
for i in clusters:
  for j in linkages:
    ag1=AgglomerativeClustering(n_clusters=i,affinity='euclidean',linkage=j)
    ag_fit=ag1.fit_predict(df2_norm)
    sl=silhouette_score(df2_norm,ag_fit)
    print(f'{i} clusters gives silhouette_score {sl} linkage is {j}'.format(i,sl,j) ) # As close silhouette score will be to +1 as much good it would be.


ag=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='average') # here i am using average linkage with 3 clusters based on high silhouette score.
result=ag.fit(df2_norm)
df1['clusters']=pd.Series(result.labels_)
df1['clusters'].value_counts()
df1.iloc[:,1:].groupby(df2_norm.clusters).mean()
df1.to_csv('airline.csv',encoding='utf-8')

# after getting the clusters using Agglomerative Clustering only 1 cluster is containing 98 percent of data. So we should not use Hierarchical Clustering here.
