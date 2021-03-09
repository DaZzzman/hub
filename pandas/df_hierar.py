import pandas as pd #для обработки и анализа данных
import matplotlib.pyplot as plt #для визуализации 
from sklearn.manifold import TSNE #машинное обучение
from sklearn import datasets
from scipy.cluster.hierarchy import linkage, dendrogram

data1 = pd.read_excel(r'C:\Users\Wotan\Desktop\pandas\excel.xlsx') #путь к файлу
df = pd.DataFrame(data1, columns = ['brinnel', 'density'])  #наименование столбцов



# Реализация иерархической кластеризации при помощи функции linkage
mergings = linkage(df, method='complete')

# Строим дендрограмму, указав параметры удобные для отображения
dendrogram(mergings,
        #    labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
           )

plt.show()