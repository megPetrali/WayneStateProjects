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


dice = pd.read_csv('dice.csv', skipinitialspace = True)
dice = dice.drop(columns='Unnamed: 0', axis=1)

# Clean location
dice[['location_name','location_city']] = dice['location'].str.split("  ",expand=True)
dice['location_city'] = dice['location_city'].str.replace(", USA", "")

dice[['location_city','location_state']] = dice['location_city'].str.split(", ",expand=True)
dice.location_state.fillna(value='Remote', inplace=True)

dice = dice.drop(columns='location_name',axis=1)

#############################################################################################################################################################

flex = pd.read_csv('flexjobs.csv')
flex = flex.drop(columns='Unnamed: 0', axis=1)

# clean HTML; remove html tags and clean html-encoded characters (also strips newlines and leading/trailing whitespace)
flex.description = flex.description.str.strip().str.replace(r'<[^<>]*>', '', regex=True)
import html as python_html
flex.description = flex.description.map(python_html.unescape)

# remove salary, since this column is empty
flex.drop(columns='salary', axis=1, inplace=True)

# Clean location; remote locations and US National are remote
flex['location_type'] = np.where(flex['jobtype'] == ('Option for Remote Job'), "Option for Remote", '0')
flex['location_type'] = np.where(flex['jobtype'].str.find('Remote') >= 0, "Remote", flex['location_type'])
flex['location_type'] = np.where(flex['location'] == "Work from Anywhere", "Remote", flex['location_type'])
flex['location_type'] = np.where(flex['location'] == "US National", "Remote", flex['location_type'])
flex['location_type'] = flex['location_type'].str.replace('0', "Physical Location")

# Clean location; separate cities and states/countries, and multiple location in-person jobs
flex['location_city'] = np.where(flex['location_type'] == "Physical Location", flex['location'].str.split(', ', 1).str[0], flex['location_type'])
flex['location_state_country'] = np.where(flex['location_type'] == "Physical Location", flex['location'].str.split(', ', 1).str[1], flex['location_type'])
flex['location_city'] = np.where(flex['location_state_country'].str.count(',') >= 2, 'Multiple Locations', flex['location_city'])
flex['location_state_country'] = np.where(flex['location_state_country'].str.count(',') >= 2, 'Multiple Locations', flex['location_state_country'])

#############################################################################################################################################################

indeed = pd.read_csv('indeed_jobs.csv')
indeed = indeed.drop(columns='Unnamed: 0', axis=1)

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
    wordcloud.to_file('assets/' + file_name)

create_wordcloud(dice['title'].values.flatten(), 'dice_jobtitle_wc.png')

app.layout = html.Div(children=[
    html.H1(children='DSE 6000 Final Project'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    html.Div(
        html.Img(src='assets/dice_jobtitle_wc.png')
    ),

    html.Div(children='''
        This is a wordcloud.
    ''')
])

if __name__ == '__main__':
    app.run_server(debug=True)
