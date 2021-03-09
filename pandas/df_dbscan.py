import pandas as pd #для обработки и анализа данных
import matplotlib.pyplot as plt #для визуализации 
from sklearn.cluster import DBSCAN #машинное обучение

data1 = pd.read_excel(r'C:\Users\Wotan\Desktop\pandas\excel.xlsx') #путь к файлу
df = pd.DataFrame(data1, columns = ['brinnel', 'density'])  #наименование столбцов

# print (df)

#df = pd.DataFrame(cars,columns=['brinnel','density'])
  
dbscan = DBSCAN(eps=2, min_samples=5).fit(df) 


plt.scatter(df['brinnel'], df['density'], c= dbscan.labels_.astype(float), s=35, alpha=0.5)
plt.show()