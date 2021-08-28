import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv('/content/drive/MyDrive/crime_data.csv')
df1.info() # there is no null value
df2=df1.iloc[:,1:]
#  standardscaler to scale values.
st=StandardScaler()
sfi=st.fit(df2)
df2_norm=sfi.transform(df2)
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
  dendo.dendrogram(link,leaf_font_size=10,leaf_rotation=0) # Using Dendogram 3 and 4 cluster were looking good.
  print("Linkage Name-->",i) # linkage name of every Dendogram
  plt.show()

# As you can see linkage ward and average are giving us better image of dendogram. but we will also try silhouette score.
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
clusters=[3,4,5,6,7]
linkages=['complete','single','ward','average','single'] 
for i in clusters:
  for j in linkages:
    ag1=AgglomerativeClustering(n_clusters=i, affinity='euclidean',linkage=j)
    agpred=ag1.fit_predict(df2_norm)
    s1=silhouette_score(df2_norm,agpred)
    print(f'{i} clusters gives silhouette_score {s1} linkage is {j}'.format(i,s1,j) ) # As close silhouette score to +1 as much good it would be


ag=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete') # here i am using complete linkage with 3 clusters based on high silhouette score.
result=ag.fit(df2_norm)
df['clusters']=pd.Series(result.labels_)
df['clusters'].value_counts()

