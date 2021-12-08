# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


indeed = pd.read_csv('indeed_jobs.csv')
indeed = indeed.drop(columns='Unnamed: 0', axis=1)

# Drop duplicates
indeed.drop_duplicates(inplace=True)

# Remove non-ascii characters
indeed.location.replace({r'[^\x00-\x7F]+':' '}, regex=True, inplace=True)

# Clean location; specify remote
indeed['location_type'] = np.where(indeed['jobtype'].str.find('Remote') >= 0, 'Remote', 'Physical Location')
indeed['location_type'] = np.where(indeed['location'] == "United States", 'Remote', indeed['location_type'])

# Clean location; city-state for non-remote
indeed['location_city'] = np.where(indeed['location_type'] == "Physical Location", indeed['location'].str.split(', ', 1).str[0], indeed['location_type'])
indeed['location_state'] = np.where(indeed['location_type'] == "Physical Location", indeed['location'].str.split(', ', 1).str[1], indeed['location_type'])
# Clean location: city-state for non-remote: split on first instance of + or ' ' in order to limit column to state code
indeed['location_state'] = np.where(indeed['location_type'] == "Physical Location", indeed['location_state'].str.split(' ', 1).str[0], indeed['location_state'])
indeed['location_state'] = np.where(indeed['location_type'] == "Physical Location", indeed['location_state'].str.split('+', 1).str[0], indeed['location_state'])

# Salary Cleaning
# hourly or yearly
indeed['salary_type'] = np.where(indeed['salary'].str.find("hour") >= 0, 'Hourly', indeed['salary'])
indeed['salary_type'] = np.where(indeed['salary'].str.find("year") >= 0, 'Yearly', indeed['salary_type'])

# remove "a year" "an hour"
indeed['salary'] = indeed['salary'].str.split(' a').str[0]

# remove "Up to "
indeed['salary'] = indeed['salary'].str.replace("Up to ","")

# remove other characters
indeed['salary'] = indeed['salary'].str.replace("$","", regex=False)
indeed['salary'] = indeed['salary'].str.replace(",","", regex=False)

# split into min and max
indeed[['salary_min','salary_max']] = indeed['salary'].str.split(" - ", expand=True)

#convert hourly number to yearly number
indeed['salary_min'] = np.where(indeed['salary_type'] == 'Hourly', indeed['salary_min'].astype(float)*40*50, indeed['salary_min'].astype(float))
indeed['salary_max'] = np.where(indeed['salary_type'] == 'Hourly', indeed['salary_max'].astype(float)*40*50, indeed['salary_max'].astype(float))

# get final average or min salary
indeed['salary'] = np.where(indeed['salary_max'].isna(), indeed['salary_min'], (indeed['salary_max'] + indeed['salary_min']) / 2)

indeed.drop(['salary_min', 'salary_max'], axis=1, inplace=True)

#############################################################################################################################################################

simply = pd.read_csv('simplyhired.csv')
simply = simply.drop(columns='Unnamed: 0', axis=1)
# Replace non-ascii chars
simply.location.replace({r'[^\x00-\x7F]+':' '}, regex=True, inplace=True)

# Drop duplicates
simply.drop_duplicates(inplace=True)

# location type
simply['location_type'] = np.where(simply['location'] == 'Remote', 'Remote', 'Physical Location')

# Adjust location columns
simply[['location_city','location_state']] = simply['location'].str.split(", ",expand=True)
simply['location_state'] = simply['location_state'].str.split(' +').str[0]
simply['location_state'] = np.where(simply['location_city'] == 'Remote', 'Remote', simply['location_state'])

# Salary Cleaning
# hourly or yearly
simply['salary_type'] = np.where(simply['salary'].str.find("hour") >= 0, 'Hourly', simply['salary'])
simply['salary_type'] = np.where(simply['salary'].str.find("year") >= 0, 'Yearly', simply['salary_type'])

# remove "a year" "an hour"
simply['salary'] = simply['salary'].str.split(' a').str[0]

# remove other characters
simply['salary'] = simply['salary'].str.replace("$","", regex=False)
simply['salary'] = simply['salary'].str.replace(",","", regex=False)

# split into min and max
simply[['salary_min','salary_max']] = simply['salary'].str.split(" - ", expand=True)

#convert hourly number to yearly number
simply['salary_min'] = np.where(simply['salary_type'] == 'Hourly', simply['salary_min'].astype(float)*40*50, simply['salary_min'].astype(float))
simply['salary_max'] = np.where(simply['salary_type'] == 'Hourly', simply['salary_max'].astype(float)*40*50, simply['salary_max'].astype(float))

# get final average or min salary
simply['salary'] = np.where(simply['salary_max'].isna(), simply['salary_min'], (simply['salary_max'] + simply['salary_min']) / 2)

simply.drop(['salary_min', 'salary_max'], axis=1, inplace=True)



from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 


stopwords = set(STOPWORDS)
def create_wordcloud(file_lines, file_name):
    # file_lines = dice['title'].values.flatten()

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
    plt.title("SimplyHired Wordcloud", fontsize=13)
    wordcloud.to_file('assets/' + file_name )


create_wordcloud(simply['title'].values.flatten(), 'simplyhired_jobtitle_wc.png')

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

wordcloud_trends = [
    '''In this wordcloud, we can see that Data Scientist is the most frequent string in the job title field, indicating that this is a job with many
        opportunities under it. If someone is looking for a position with the specific title of Data Scientist, then this is a good dataset for them to look in!'''
    ,
    '''We can also see prominent job titles of Data Analyst, Statistical Programmer, and Data Science. Data Analyst is definitely a job that many people specifically look for,
        and this dataset gives many opportunities for them. Data Science is probably a job asset that Data Scientists might look for, reinforcing our previous point that this
        dataset contains many opportunities for Data Scientists. 
        '''
    ,
    '''Machine Learning, the next largest new string, is a much different string from those previously mentioned. That it is in such high
        demand here is very insightful. Job seekers looking for a job in data and excel at programming and machine learning techniques might want to take this to mean they should
        look more into job opportunities like these.
        '''
    ,
    '''We also see some terms that are very similar to job titles already mentioned, like Analyst Data, Data Science, and multiple that contain the word Analyst,
        including healthcare, QA, and Technology. This indicates that Analysts of all types are in high demand. This also indicates to us that when applying for jobs, having an idea of what kind of analyst
        you want to be can give you an advantage in honing in on a job of interest and one that will be a good fit.
        '''
    ,
    '''Finally, we see a multitude of levels represented in this data. We can see that there are roles open for Junior-level, Senior Level, and even
        Managers. This dataset contains jobs for everyone, regardless of how far they are in their career. There does seem to be slightly more jobs for Senior levels
        than there are for the others, making this source an expecially good one for those who are already on the senior leve and for people looking to move up in their
        career.
        '''
]

app.layout = html.Div(children=[
    html.H1(children='DSE 6000 Final Project'),

    html.H3(children='''
        Rachel Balon, Krishna Manjeera Chittoor, Joseph Felice
    '''),

    html.Br(),

    html.Div(children='''
        The following is a wordcloud of the job titles in the SimplyHired dataframe. We wanted to see what the most 
        frequent terms are listed in the Job Title fields. This would give us an idea of what the most common job titles are, 
        which in turn would give us an idea of what the most common choices are for people looking to get into Data Science careers.
    '''),

    html.Div(
        html.Img(
            src='https://drive.google.com/uc?export=download&id=1Ko3b73pnRSi3eKbNDai7kQqKxUOCFEWI', 
            style={
                'height':'70%', 
                'width':'70%',
                'border-style':'solid',
                'border-width':'2px'
            }
        )
    ),

    html.Div(
        className="wordcloud_trends",
        children=[
            html.Ul(id='wc_trend_list', children=[html.Li(i) for i in wordcloud_trends])
        ],
    ),


])

if __name__ == '__main__':
    app.run_server(debug=True)
