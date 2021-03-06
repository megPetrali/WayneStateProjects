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
# hourly or yearly or monthly
indeed['salary_type'] = np.where(indeed['salary'].str.find("hour") >= 0, 'Hourly', indeed['salary'])
indeed['salary_type'] = np.where(indeed['salary'].str.find("year") >= 0, 'Yearly', indeed['salary_type'])
indeed['salary_type'] = np.where(indeed['salary'].str.find("month") >= 0, 'Monthly', indeed['salary_type'])

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

#convert monthly number to yearly number
indeed['salary_min'] = np.where(indeed['salary_type'] == 'Monthly', indeed['salary_min'].astype(float)*12, indeed['salary_min'].astype(float))
indeed['salary_max'] = np.where(indeed['salary_type'] == 'Monthly', indeed['salary_max'].astype(float)*12, indeed['salary_max'].astype(float))

# get final average or min salary
indeed['salary'] = np.where(indeed['salary_max'].isna(), indeed['salary_min'], (indeed['salary_max'] + indeed['salary_min']) / 2)

indeed.drop(['salary_min', 'salary_max'], axis=1, inplace=True)

indeed.drop('jobtype', axis=1, inplace=True)

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
# hourly or yearly or daily
simply['salary_type'] = np.where(simply['salary'].str.find("hour") >= 0, 'Hourly', simply['salary'])
simply['salary_type'] = np.where(simply['salary'].str.find("year") >= 0, 'Yearly', simply['salary_type'])
simply['salary_type'] = np.where(simply['salary'].str.find("a day") >= 0, 'Daily', simply['salary_type'])

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

#convert daily to yearly number
simply['salary_min'] = np.where(simply['salary_type'] == 'Daily', simply['salary_min'].astype(float)*50, simply['salary_min'].astype(float))
simply['salary_max'] = np.where(simply['salary_type'] == 'Daily', simply['salary_max'].astype(float)*50, simply['salary_max'].astype(float))

# get final average or min salary
simply['salary'] = np.where(simply['salary_max'].isna(), simply['salary_min'], (simply['salary_max'] + simply['salary_min']) / 2)

simply.drop(['salary_min', 'salary_max'], axis=1, inplace=True)
simply.drop('company', axis=1, inplace=True)

all_data = pd.concat([indeed, simply])

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 


stopwords = set(STOPWORDS)
def create_wordcloud(file_lines, file_name, title):
    # file_lines = dice['title'].values.flatten()

    words =''
    for line in file_lines:
        tokens = line.split()
        for token in tokens:
            words = words + ' ' + token

    wordcloud = WordCloud(width = 1400, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10)
    wordcloud.generate(words.lower()) 

    # plot the WordCloud            
    plt.title(title, fontsize=13)
    plt.imshow(wordcloud)
    plt.axis("off") 
    # plt.tight_layout(pad = 0)     
    wordcloud.to_file('assets/' + file_name )


all_data['lower_title'] = all_data['title'].str.lower()
all_data['simplified_title'] = np.where(all_data['lower_title'].str.find("manager") >= 0, "Manager",
                            np.where(all_data['lower_title'].str.find("director") >= 0, "Director",
                            np.where(all_data['lower_title'].str.find("vp ") >= 0, "Vice President",
                            np.where(all_data['lower_title'].str.find("vice president") >= 0, "Vice President",
                            np.where(all_data['lower_title'].str.find("data scientist") >= 0, "Scientist",
                            np.where(all_data['lower_title'].str.find("analyst") >= 0, "Analyst",
                            np.where(all_data['lower_title'].str.find("statisti") >= 0, "Statistician",
                            np.where(all_data['lower_title'].str.find("engineer") >= 0, "Engineer", 
                            np.where(all_data['lower_title'].str.find("data science") >= 0, "Other - DS", "Other")))))))))
all_data.drop(columns='lower_title', axis=0, inplace=True)

salary_job_title_plot = px.box(all_data, 
                                x="simplified_title", 
                                y="salary", 
                                title="Salaries by Job Titles", 
                                labels={'simplified_title': 'Job Title', 'salary': 'Salary'},
                                category_orders={"simplified_title": ["Analyst","Engineer","Scientist","Statistician","Other - DS","Other"]}
                                )



create_wordcloud(simply['title'].values.flatten(), 'simplyhired_jobtitle_wc.png', 'Simply Hired Job Titles')

create_wordcloud(indeed['title'].values.flatten(), 'indeed_jobtitle_wc.png', 'Indeed Job Titles')

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
    '''We can also see prominent job titles containing Data Analyst, Machine Learning, and Data Science. Data Analyst is definitely a job that many people specifically look for,
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
        including healthcare, QA, and engineering. This indicates that Analysts of all types are in high demand. This also indicates to us that when applying for jobs, having an idea of what kind of analyst
        you want to be can give you an advantage in honing in on a job of interest and one that will be a good fit.
        '''
    ,
    '''Finally, we see a multitude of levels represented in this data. We can see that there are roles open for Junior-level, Senior Level, and even
        Managers. This dataset contains jobs for everyone, regardless of how far they are in their career. There does seem to be slightly more jobs for Junior levels
        than there are for the others, making this source an especially good one for entry level employees.
        '''
]

salary_jobtitle_boxplot_trends = [
    '''From this boxplot we can see that Vice Presidents make on average and overall the most money, followed by Directors. Surprisingly, Managers and Engineers make very similar distributions of salary. This could be because some senior engineers take on as many duties as managers, however they keep Engineer in their job title because they are not technically in charge of a team.'''
    ,
    '''Analysts make the minimum Median salary, but Statisticians make the minimum salary (out of the classified groups) this tells us that, on average, analysts make the least amount of money, but this makes sense because this is a pretty basic job title which probably on average pays more. Statisticians making the minimum salary tells us that these job seekers have to be very cognizant of what they are worth and what wage they should be asking for.'''
    ,
    '''Engineers make the maximum salary in the dataset. This is higher than all of the salaries for Vice President, Director, and Managers. This tells us that job titles that contain the word Engineer have the opportunity for the highest salaries'''
    ,
    '''There are a couple outliers in some of the groups. The "Other Data Science" jobs has the second highest salary in the dataset. This job is for a Data Science & AI Architect at IBM. This might be an indication that AI experts are in high demand. We can see whether AI is an important part of many jobs in that query visual.'''
]

#remote vs in person comparision
avg_salaries_locationtype = all_data.dropna().groupby(['location_type', 'location_state'] , as_index=False ).mean()
print(avg_salaries_locationtype.head(50))


import plotly.graph_objects as go
from dash.dependencies import Input, Output
app.config.suppress_callback_exceptions = True

# @app.callback(Output('remote_graph', 'figure'),
#     [Input('radio-items', 'value')])
# def make_bar_chart(value):
#     trace = px.bar(avg_salaries_locationtype, x='salary', y='location_type' , 
#                               text = 'salary' , orientation= 'h' 
#                               , title = 'Salary by Type of Location'
#                               , labels ={ 'salary':'Salary ($)', 'location_type':'Location Type'})
#     # layout = #define layout
#     figure = go.Figure(data=[trace])    
#     figure.update_layout(transition_duration=500)
#     # figure.update_traces(texttemplate='%{text:$.4s}', textposition='outside')
#     return figure

@app.callback(
    Output('remote_graph', 'figure'),
    Input('radio-items', 'value'))
def update_figure(value):
    filtered_df = avg_salaries_locationtype[(avg_salaries_locationtype.location_state == value) | (avg_salaries_locationtype.location_state == 'Remote')]
    fig = px.bar(filtered_df, x='salary', y='location_type' , 
                              text = 'salary' , orientation= 'h' ,
                              title = 'Salary by Type of Location'
                              , labels ={ 'salary':'Salary ($)', 'location_type':'Location Type'})

    fig.update_layout(transition_duration=500)
    fig.update_layout(
        autosize=True,
        margin=dict(l=200, r=80, t=50, b=200)
    )

    fig.update_traces(texttemplate='%{text:$.4s}', textposition='outside')
    return fig

#create plotly salary map by state
from urllib.request import urlopen
import json
import pandas as pd
import plotly.express as px


state_salary_avg = all_data.groupby('location_state' , as_index=False).mean()
fig_salarymap = px.choropleth(state_salary_avg,  locations='location_state', 
                             color = 'salary' ,
                             color_continuous_scale="Greens", 
                             range_color=(0, 150000),
                             locationmode = 'USA-states',                            
                             scope="usa") 
                            

fig_salarymap.update_layout(
    title_text = 'Salary by State' ,
    margin={"r":0, "t":50, "l":0, "b":0} 
)

#plotly map for job volume by state
state_totals = all_data.groupby('location_state' , as_index=False).size()
state_totals.head()


fig_JobCountMap = px.choropleth(state_totals,  locations='location_state', 
                             color = 'size',
                             color_continuous_scale="blues",
                             range_color=(0,240),
                             locationmode = 'USA-states',                            
                             scope="usa",
                             title = 'Job Count by State') 
                            

fig_JobCountMap.update_layout(
    title_text = 'Salary by State' ,
    margin={"r":0, "t":50, "l":0, "b":0} 
)

all_data['description'] = all_data['description'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (STOPWORDS)]))
all_data['description'] = all_data['description'].str.replace('[^\w\s]','')

file_lines = all_data['description'].str.lower().values.flatten()
description_words = []
for line in file_lines:
    tokens = str(line).split()
    for token in tokens:
        description_words.append(token)

description_df = pd.DataFrame(description_words)
description_df = description_df[0].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (STOPWORDS)]))

description_df = description_df.value_counts()
description_df = description_df.reset_index()
description_df.columns = ['words','count']

description_df = description_df[description_df['count'] != 'data']
description_df = description_df[description_df['words'] != '']


description_df = description_df[description_df['words'] != 'data']
description_scatterplot = px.scatter(description_df.nlargest(50,columns=['count']), x='words', y='count', size='count', color='count',
                    hover_data=['count'], title='Most Frequent Words in Job Descriptions')
description_scatterplot.update_layout(
    margin=dict(l=20, r=20, t=50, b=20),
)

salary_location_trendlist = [
    '''With the data we used from Simply Hired and Indeed, remote jobs are shown to have a higher salary.''',
    '''There is an appoximate 5K difference in avarage salary for remote jobs versus in person.'''
    


]

jobcount_map_trendlist = [
    '''Most data related jobs are located in California, with second highest being New York.''',
    '''California's job volume for data related jobs is significantly more than other states at around double of the amount of jobs in other states
Our datasets included little data on the northwest(Montana etc.)'''


]

salary_map_trendlist = [
    '''Rhode Island and Connecticut are shown as the state with the highest salary but our final dataset only included one salary in each of these states, so more data would be needed on those states to make a full conclusion. Next, Virginia has the highest salary, likely due to large metropolitan areas such as near Washington, D.C.''',
    '''Here in Mighigan the average salary for a data related job is 70k. Michigan ranks about average compared to the other states.'''

]



# Pyspark
# import os
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["SPARK_HOME"] = "/spark/spark-3.2.0-bin-hadoop3.2"

# import findspark
# findspark.init()
# from pyspark.sql import SparkSession, Row
# from pyspark.sql.functions import concat, col, lit,when,count,isnull,max,min,to_date
# from pyspark import SparkContext,SparkConf

# # get a spark session. 
# spark = SparkSession.builder.master("local[*]").getOrCreate()

#from pyspark.sql.functions import split, explode, desc
#spark_simply= spark.read.csv("simplyhired.csv",header='true', 
#                      inferSchema='true')
#dfWords1 = spark_simply.select(explode(split('description', '\\s+')).alias('word')) \
#                     .groupBy('word').count().orderBy(desc('word'))
#dfWords2 = spark_simply.select(explode(split('title', '\\s+')).alias('word')) \
#                     .groupBy('word').count().orderBy(desc('word'))

#spark_simply=spark.createDataFrame(simply)
#spark_simply.createOrReplaceTempView('simply')

#statesal=spark.sql("""
#SELECT location_state, salary from simply
#Where location_state != '%Remote%'
#order by location_state;
# """)

#df6=statesal.na.drop("any")
#statedf=df6.orderBy('salary')
#statedf.toPandas()

# query for avg state salary
#avgst= statedf.groupby('location_state').avg()
#avgstate=avgst.select('location_state',round('avg(salary)',0)).withColumnRenamed('round(avg(salary), 0)', 'salary').orderBy('location_state')
#avgstate.show()

description_counts = pd.read_csv('simplywordcount.csv').drop(columns='Unnamed: 0', axis=1)
title_counts = pd.read_csv('titlecount.csv').drop(columns='Unnamed: 0', axis=1)

#export avg salary df to csv

#b.to_csv('AvgStateSalary.csv')


app.layout = html.Div(children=[
    html.H1(children='DSE 6000 Final Project'),

    html.H3(children='''
        Rachel Balon, Megan Petralia, Joseph Felice
    '''),

    html.Br(),

    html.H4(children='''
        The following is a wordcloud of the job titles in the SimplyHired dataframe. We wanted to see what the most 
        frequent terms are listed in the Job Title fields. This would give us an idea of what the most common job titles are, 
        which in turn would give us an idea of what the most common choices are for people looking to get into Data Science careers.
    '''),

    html.Div(children=[
        html.H2(children='''Simply Hired Wordcloud''', style={'padding':'0px','textAlign': 'center'}),
        html.Img(
            src='https://drive.google.com/uc?export=download&id=1Ko3b73pnRSi3eKbNDai7kQqKxUOCFEWI', 
            style={
                'height':'405px', 
                'width':'720px',
                'borderStyle':'solid',
                'borderWidth':'2px',
                'padding':'0px',
                'margin-top':'0px',
                'textAlign': 'center'
            }
        )
    ], style={'textAlign': 'center'}),

    html.Div(
        className="wordcloud_trends",
        children=[
            html.Ul(id='wc_trend_list', children=[html.Li(i) for i in wordcloud_trends])
        ],
    ),

    html.H4(children='''
        We can compare this wordcloud to the one for the indeed data to see if there are any differences between frequent job title contents:
    '''),

    html.Div(children=[
        html.H2(children='''Indeed Wordcloud''', style={'padding':'0px','textAlign': 'center'}),
        html.Img(
            src='https://drive.google.com/uc?export=download&id=1dUJV5U-98fiC1Nfn1iMcbh5Grd48WAJE', 
            style={
                'height':'337.5px', 
                'width':'600px',
                'borderStyle':'solid',
                'borderWidth':'2px',
                'padding':'0px',
                'margin-top':'0px',
                'margin-left':'auto',
                'margin-right':'auto'
            }
        )
    ], style={'textAlign': 'center'}),

    html.Div(
        className="wordcloud_trends_indeed",
        children=[
            html.Ul(id='wc_trend_list_indeed', children=[html.Li(i) for i in [
                '''We can see here that Data Scientist is still the most frequent string in a job title. Data Science and Analytics are also still
                highly prevalent in the data.''',
                '''However, we can see that the indeed jobs have more healthcare-related jobs (by seeing the words health, registered nurse, and even some specific
                health fields like oncology and respiratory therapist. These jobs are interesting to be included in the data, and when we look at the descriptions
                we can see that they actually have minimal data manipulation needs and more data collection needs. These jobs also seem to have higher expectations
                (for example, you should be an RN to apply for a registered nurse) which are not normally needed in the data field. That makes the simply dataset
                better for filtering out those types of jobs.''']])
        ],
    ),

    # html.Div(children=generate_table(description_counts)),

    html.Br(),

    html.H4(children='''
        Next, we will look at the job titles at a higher level by categorizing them, which makes it easier to compare salaries by job type. We
        want to see which job titles pay best and which ones have the potentially highest pay.
    '''),
    
    dcc.Graph(
        id='salary_title_graph',
        figure=salary_job_title_plot
    ),

    html.Div(
        className="salary_jobtitle_trends",
        children=[
            html.Ul(id='salary_jobtitle_trend_list', children=[html.Li(i) for i in salary_jobtitle_boxplot_trends])
        ],
    ),

    html.H4(children='''
            Below is a bar chart used to compare the average salary for remote positions versus in-person.
    '''),

    html.Div(children=[
        dcc.Graph(
            id='remote_graph',
        ),
        dcc.Dropdown(
            id='radio-items',
            options = [
                {'label': 'Rhode Island', 'value': 'RI'},
                {'label': 'Connecticut', 'value': 'CT'},
                {'label': 'District of Columbia', 'value': 'DC'},
                {'label': 'Alaska', 'value': 'AK'},
                {'label': 'Washington', 'value': 'WA'},
                {'label': 'Colorado', 'value': 'CO'},
                {'label': 'California', 'value': 'CA'},
                {'label': 'Illinois', 'value': 'IL'},
                {'label': 'Texas', 'value': 'TX'},
                {'label': 'New York', 'value': 'NY'},
                {'label': 'Virginia', 'value': 'VA'},
                {'label': 'Missouri', 'value': 'MO'},
                {'label': 'Tennessee', 'value': 'TN'},
                {'label': 'New Jersey', 'value': 'NJ'},
                {'label': 'Georgia', 'value': 'GA'},
                {'label': 'Arizona', 'value': 'AZ'},
                {'label': 'Maryland', 'value': 'MD'},
                {'label': 'Florida', 'value': 'FL'},
                {'label': 'Michigan', 'value': 'MI'},
                {'label': 'Ohio', 'value': 'OH'},
                {'label': 'Pennsylvania', 'value': 'PA'},
                {'label': 'Delaware', 'value': 'DE'},
                {'label': 'Utah', 'value': 'UT'},
                {'label': 'South Carolina', 'value': 'SC'},
                {'label': 'North Carolina', 'value': 'NC'},
                {'label': 'Wisconsin', 'value': 'WI'},
                {'label': 'Kentucky', 'value': 'KY'},
                {'label': 'Massachusetts', 'value': 'MA'},
                {'label': 'Indiana', 'value': 'IN'},
                {'label': 'Hawaii', 'value': 'HI'},
                {'label': 'Louisiana', 'value': 'LA'},
                {'label': 'Arkansas', 'value': 'AR'},
                {'label': 'New Mexico', 'value': 'NM'}
            ],
            value = "MD"
            )
            ],
        style={"width": "50%"},
    ),

    html.Div(
        className="salary_location_trends",
        children=[
            html.Ul(id='salary_location_trend_list', children=[html.Li(i) for i in salary_location_trendlist])
        ],
    ),

    html.H4(children='''
        Below is a map of the United States showing average salary per state.
    '''),
    
    dcc.Graph(
        id='salary_map',
        figure=fig_salarymap,
        style={
                'height':'40%', 
                'width':'40%',
                'borderStyle':'solid',
                'borderWidth':'2px',
                'margin-left':'auto',
                'margin-right':'auto'}
    ),

    html.Div(
        className="salary_map_trends",
        children=[
            html.Ul(id='salary_map_trend_list', children=[html.Li(i) for i in salary_map_trendlist])
        ],
    ),
 
    html.H4(children='''
        Below is a map of the United States showing the volume of data related job listing per state.
    '''),    
    
    dcc.Graph(
        id='jobcount_map',
        figure=fig_JobCountMap,
        style={
                'height':'40%', 
                'width':'40%',
                'borderStyle':'solid',
                'borderWidth':'2px',
                'margin-left':'auto',
                'margin-right':'auto'}
    ),

    html.Div(
        className="jobcount_map_trends",
        children=[
            html.Ul(id='jobcount_map_trend_list', children=[html.Li(i) for i in jobcount_map_trendlist])
        ],
    ),
    
    html.H4(
        '''The next visual is a look at what is mentioned in the job descriptions. We wanted to look at whether specific skills or cetain aspects of jobs in this dataset jump out.'''
    ),

    dcc.Graph(
        id='desc_scatter',
        figure=description_scatterplot,
        style={
            'height':'40%',
            'width':'40%',
            'borderStyle':'solid',
            'borderWidth':'2px',
            'margin-left':'auto',
            'margin-right':'auto'
        }
    ),

    html.Div(
        className="description_trends",
        children=[
            html.Ul(id='description_trend_list', children=[html.Li(i) for i in [
                '''Experience is highly sought-after''',
                '''Most Jobs deal with ???Analysis??? and ???Analytics???''',
                '''???Years??? came up as a top word, which can be observed as employers looking for years of job experience''',
                '''???Machine???, ???Learning??? and ???Engineering??? also had high totals, which can indicate employers have a strong need for individuals with AI/ML data engineering skills.''',
                '''???Team??? showed to also have high counts. This is indicating employers are looking for candidates with abilities to work with a team, which is a necessity in the workplace.''',
                '''Lastly, ???Business??? was also a top word counted. Having strong knowledge of ???business??? practices is a necessity to being a successfully data scientist/analyst'''
            ]])
        ],
    ),

    html.Div('''
        In conclusion, we found many insights in this dataset. Some of the most common contents of job titles in the dataset are Data Analyst, Machine Learning, and Data Science, with Data Scientist being the most common. That makes these data especially helpful to people looking for jobs of these types, and Data Science a skill that is necessary. Machine learning tells us that there are also many opportunities on the more technical and programmatic side of this discipline. From the multitude of job titles we see, being able to specify what type of analyst you want to be (qa, technology, health) might make it easier to narrow down to jobs of interest. This dataset contains jobs for many different levels of the job selection process, including Junior, Senior, and Managers. Comparing the two data sources shows us that there are very similar job offerings available in these datasets. However, the Indeed data contains more Health-related fields, like Nurses and Healthcare statisticians. This is either a good thing or a bad thing depending on what you are looking for. Some of those nursing jobs do not seem to have much to do with data analytics and more with data collection, not to mention also requiring a distinct schooling that not everyone searching for Data-oriented jobs may have. In terms of salary, Vice Presidents make on average and overall the most money, followed by Directors, but surprisingly Managers and Engineers make very similar distributions of salary. On the other side, analysts make the minimum Median salary, but Statisticians make the minimum salary which tells us that, on average, analysts make the least amount of money, but this makes sense because this is a pretty basic job title which probably on average pays more. Engineers make the maximum salary in the dataset. This is higher than all of the salaries for Vice President, Director, and Managers. Remote jobs are shown to have a higher salary on average, but that varies by the physical location. There is an approximate 5K difference in average salary for remote jobs versus in person. Rhode Island and Connecticut are shown as the state with the highest salary but our final dataset only included one salary in each of these states, so more data would be needed on those states to make a full conclusion. Next, Virginia has the highest salary, likely due to large metropolitan areas such as near Washington, D.C. Here in Michigan, the average salary for a data related job is 70k. Michigan ranks about average compared to the other states. Most data related jobs listed on Simply Hired and Indeed are located in California, with second highest being New York. California's job volume for data related jobs is significantly more than other states at around double of the amount of jobs in other states. Our datasets included little data on the northwest(Montana etc.). Finally, job descriptions can also tell us much about what Data-related jobs can require. Experience is highly sought-after, as are Machine Learning skills. Team-based jobs are also very common. Finally, having some knowledge of business practices is in high demand, even for a technical job in Data Science and Analytics.
    ''', style={'padding':'20px'})

])



if __name__ == '__main__':
    app.run_server(debug=True)
