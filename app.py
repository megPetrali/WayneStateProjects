# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv('dice.csv')

df = df.drop(columns='Unnamed: 0', axis=1)

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 

stopwords = set(STOPWORDS)

file_lines = df['title'].values.flatten()

words =''
for line in file_lines:
  tokens = line.split()
  for token in tokens:
    words = words + ' ' + token

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(words.lower()) 


# plot the WordCloud                        
plt.figure(figsize = (8, 8)) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
wordcloud.to_file('assets/wc.png')

app.layout = html.Div(children=[
    html.H1(children='DSE 6000 Final Project'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    html.Div(
        html.Img(src='assets/wc.png')
    ),

    html.Div(children='''
        This is a wordcloud.
    ''')
])

if __name__ == '__main__':
    app.run_server(debug=True)
