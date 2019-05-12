#*****************************************************************************************************
#Importing Libraries
#*****************************************************************************************************
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['wordnet'])
import re
import pandas as pd
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import pickle
#*****************************************************************************************************
#Functions
#*****************************************************************************************************
def load_data(database_filepath):

    """Function that loads a sql database
    
    Args:
    database_filepath - Path for the sql database
    
    Returns:
    X - Messages found in the loaded sql database
    Y - Categories in which are classified the loaded messages
    category_names - Name for the categories
    """    
    
    #Reading data
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    #Defining datasets
    X = df.message
    Y = df[df.columns[4:]]
    category_names = df.columns[4:]

    return X, Y, category_names

def tokenize(text):
    
    """Function that separates and classify words in messages for NLP
    
    Args:
    text - Corpus
    
    Returns:
    clean_tokens - Sectioned and classified corpus
    """        
    
    #Creating a tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    #Removing unwanted characters
    tokens = tokenizer.tokenize(text.replace("'s", ' is').replace("'re", ' are'))
    lemmatizer = WordNetLemmatizer()

    #Creating tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    
    """Function that creates a ML model to classify emergency messages, using pipelines
    
    Args:
    None
    
    Returns:
    model - Machine Learning Model
    """        
    
    model = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ]))
        ])),
        ('clf', MultiOutputClassifier(MultinomialNB(alpha=.01)))
    ])

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    
    """Function that test the performance of the used ML model
    
    Args:
    model - Machine Learning Model to test
    X_test - Messages to test the Model
    Y_test - Real categories for the X_test messages
    
    Returns:
    None    
    """
    
    y_test_pred = model.predict(X_test)

    for n in range(Y_test.shape[1]):
        print(classification_report(Y_test.values[:,n], y_test_pred[:,n]))

def save_model(model, model_filepath):
    
    """Function that saves the trained ML model using pickle
    
    Args:
    model - Machine Learning Model to save
    model_filepath - Path and name to save the model    
    
    Returns:
    None
    """     
    
    #Saving model
    pickle.dump(model, open(model_filepath, 'wb'))
#*****************************************************************************************************
#Main
#*****************************************************************************************************
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()