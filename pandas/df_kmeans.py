import pandas as pd #для обработки и анализа данных
import matplotlib.pyplot as plt #для визуализации 
from sklearn.cluster import KMeans #машинное обучение

data1 = pd.read_excel(r'C:\Users\Wotan\Desktop\pandas\excel.xlsx') #путь к файлу
df = pd.DataFrame(data1, columns = ['brinnel', 'density'])  #наименование столбцов

# print (df)

#df = pd.DataFrame(cars,columns=['brinnel','density'])
  
kmeans = KMeans(n_clusters=3).fit(df) 
centroids = kmeans.cluster_centers_ 
print(centroids)

plt.scatter(df['brinnel'], df['density'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()