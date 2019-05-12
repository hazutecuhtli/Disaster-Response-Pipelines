#*****************************************************************************************************
#Importing Libraries
#*****************************************************************************************************
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
#*****************************************************************************************************
#Functions
#*****************************************************************************************************
def load_data(messages_filepath, categories_filepath):

    """Function that loads csv data related with emergency messages responses
    
    Args:
    messages_filepath - Path for the csv that contains information about emergency messages
    categories_filepath - Path for the csv files that contains the classification of the emergency messages
    
    Returns:
    df - Dataframe containin emergency messages and their categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)

    return df

def clean_data(df):

    """Function used to clean the pandas dataframe containing emergency messages
    
    Args:
    df - Dataframe containing emergency messages and their categories
    
    Returns:
    df - Dataframe cleaned, containing emergency messages and their categories
    """    
    # Creating a dataframe with the df categories
    categories = df.categories.str.split(';', expand=True)

    # Creating the columns names
    row = categories.iloc[0].tolist()
    category_colnames = [column.split('-')[0] for column in row]
    categories.columns = category_colnames

    # Converting the content of the dataframe to integers
    for column in categories:
        categories[column] = categories.apply(lambda x : x[column].split('-')[1], axis=1)
        categories[column] = categories[column].astype(int)

    # Changing categories in the df dataframe
    df.drop('categories', axis=1, inplace=True)
    df = df.join(categories)

    # Removing duplicates in the df dataframe
    df.drop(df[df.duplicated()].index, axis=0, inplace=True)

    return df

def save_data(df, database_filename):

    """Function used to saved a dataframe as a sql database
    
    Args:
    df - Pandas dataframe
    database_filename - Path and name of the sql database to save
    
    Returns:
    None
    """     
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False)
#*****************************************************************************************************
#Main
#*****************************************************************************************************
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()