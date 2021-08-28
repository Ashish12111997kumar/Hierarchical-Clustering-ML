
#Q3.)Perform clustering analysis on the telecom data set. The data is a mixture of
# both categorical and numerical data. It consists the number of customers who churn. 
import pandas as pd
# =============================================================================
# Import data into python in a data frame format
# =============================================================================
import matplotlib.pylab as plt
# =============================================================================
# data visuvalization library
# =============================================================================

df1 = pd.read_excel("/content/drive/MyDrive/Telco_customer_churn.xlsx")

df1.describe()
df1.info()
df1.columns
#data preprocessing
# drop the unwanted columns
df = df1.drop(['Customer ID', 'Count', 'Quarter', 'Referred a Friend','Offer', 'Phone Service', 'Multiple Lines','Internet Service', 'Internet Type',
       'Online Security', 'Online Backup', 'Device Protection Plan','Premium Tech Support', 'Streaming TV', 'Streaming Movies',
       'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing','Payment Method'], axis=1)


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df)
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df['clust'] = cluster_labels # creating a new column and assigning it to new column 

df1 = df.iloc[:, [10,0,1,2,3,4,5,6,7,8,9]]
df1.head()

# Aggregate mean of each cluster
df1.iloc[ :, 0:].groupby(df1.clust).mean()

# creating a csv file 
df1.to_csv("Telco_customer_churn.xlsx", encoding = "utf-8")

import os
os.getcwd()

