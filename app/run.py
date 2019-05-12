#*****************************************************************************************************
#Importing Libraries
#*****************************************************************************************************
import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine 
import plotly.figure_factory as ff
from plotly.graph_objs import Histogram

app = Flask(__name__)

#*****************************************************************************************************
#Functions
#*****************************************************************************************************
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def TopicsMessages(df):
   
    """Function that creates a dataframe with the messages topics sources
    
    Args:
    df - Dataframe containing the messages characteristics
    
    Returns:
    df2 - Dataframe containing the messages topic sources
    TotalMessages - Total number of messages per topic
    """

    Topics = []
    Genres = []
    Counts = []
    TotalMessages = []
    for column in df.columns[4:]:
        Vals = df[df[column] == 1].groupby('genre').count()[column].tolist()
        Topics+= [column,column,column]
        Genres+=['direct', 'news', 'social']
        TotalMessages.append(sum(Vals))
        for n in range(len(Vals)):
            Counts.append(Vals[n])
        if len(Vals) <3:
            for n in range(len(Vals),3):
                Counts+=[0]            

    dictGenres = {'TypeMessages':Topics, 'Genres':Genres, 'Counts':Counts}
    df2 = pd.DataFrame(dictGenres)
    
    return df2, TotalMessages

def MessagesCharac(df):
   
    """Function that generates a vector with the messages words lengths and
       creates their distribution
    
    Args:
    df - Dataframe containing the messages characteristics
    
    Returns:
    MessLens - Numpy array containing the number of words in each message
    TotalMessages - Distribution of the number of words in each message
    """
    #Generating a vector with the message words lengths
    MessLens = []
    for message in df.message:
        MessLens.append(len(message.split()))

    #Removing outliers from the messages words lengths vector 
    MessLens = np.asarray(MessLens)
    q3, q1 = np.percentile(MessLens, [75 ,25])
    IQR = q3 - q1
    fence_low = q1 - 1.5*IQR
    fence_high = q3 + 1.5*IQR
    CountsDist = np.where(np.logical_and(MessLens>fence_low, MessLens<fence_high))
    MessDist = MessLens[CountsDist[0]]
        
    return MessLens, MessDist
    
#*****************************************************************************************************
#Plotting
#*****************************************************************************************************

group_labels = ['distplot']
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)

    #Creating a pandas dataframe with less columns for more comprehensible visualizations
    columns = ['related','request', 'offer', 'aid_related','weather_related','direct_report']
    df2 = df.drop(columns, axis=1)
    
    #Generating a vector with the message words lengths
    MessLens, MessDist = MessagesCharac(df)
    
    #Creating a pandas dataframe with the messages topics sources
    dfGenres, TotalMessages = TopicsMessages(df2)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [       
        {
            'data': [
                Bar(
                    x=df2.columns[4:], 
                    y=TotalMessages,
                    name = 'Direct'
                )
            ],

            'layout': {
                'autosize':True,
                'width':1100,
                'height':700,
                'margin':{
                    'l':60,
                    'r':50,
                    'b':200,
                    't':100,
                    'pad':4
                },
                'title': 'Message Topics',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Topics",
                    'tickangle':-45,
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=MessDist,
                    opacity=0.7
                )
            ],

            'layout': {
                'autosize':True,
                'width':1100,
                'height':700,
                'margin':{
                    'l':60,
                    'r':50,
                    'b':200,
                    't':100,
                    'pad':4
                },
                'title': 'Distribution of the Messages lengths',
                'filename':'basic histogram',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Number of Words",
                    'tickangle':0,
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df2.columns[4:], 
                    y=dfGenres[dfGenres["Genres"] == "direct"]["Counts"], 
                    name = 'Direct'
                ),
                Bar(
                    x=df2.columns[4:], 
                    y=dfGenres[dfGenres["Genres"] == "news"]["Counts"], 
                    name = ' News'
                ),
                Bar(
                    x=df2.columns[4:], 
                    y=dfGenres[dfGenres["Genres"] == "social"]["Counts"], 
                    name = 'Social'
                )
            ],

            'layout': {
                'autosize':True,
                'width':1100,
                'height':700,
                'margin':{
                    'l':60,
                    'r':50,
                    'b':200,
                    't':100,
                    'pad':4
                },
                'title': 'Message Sources',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Topics",
                    'tickangle':-45,
                }
            }
        }        
        
    ]  
       
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

#*****************************************************************************************************
#Main
#*****************************************************************************************************

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()