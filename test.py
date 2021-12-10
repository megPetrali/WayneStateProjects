import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
all_data = pd.read_csv('simplyhired.csv')

all_data['description'] = all_data['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))

file_lines = all_data['description'].values.flatten()
description_words = []
for line in file_lines:
    tokens = line.split()
    for token in tokens:
        description_words.append(token)

description_df = pd.DataFrame(description_words)
description_df = description_df[0].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))

description_df = description_df.value_counts()
description_df = description_df.reset_index()
description_df.columns = ['words','count']

description_df = description_df[description_df['count'] != 'data']

description_scatterplot = px.scatter(description_df.nlargest(50,columns=['count']), x='words', y='count', size='count', color='count',
                    hover_data=['count'])