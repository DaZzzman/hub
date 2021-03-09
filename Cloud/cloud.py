from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np 

text = open('text.txt', 'r').read()
stopwords = set(STOPWORDS)

custom_mask = np.array(Image.open('cloud1.png')) 
wc = WordCloud(background_color = 'white',
               stopwords = stopwords,
               mask = custom_mask,
               contour_width = 2,
               contour_color = 'black')

wc.generate(text)
image_colors = ImageColorGenerator(custom_mask)
wc.recolor(color_func = image_colors)


wc.to_file('wordcloud.png')

