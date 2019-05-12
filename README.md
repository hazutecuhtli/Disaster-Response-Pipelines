# Disaster Response Pipeline Project

### Summary:

This project uses data from Figure Eight about messages obtained from different types of emergencies. The project objective is to classify those messages accordingly to their different categories, which represent different topics with which the messages can be related.

### Content:

The project is composed of the following folders and main files:

- app folder:<br>
  - run.py: File used to visualize the obtained results and to classify similar messages to detect futures responses to emergencies
  
- data folder:<br>
  - process_data.py: File that contains a script to process the emergency response messages, by loading, merging and cleaning the messages.
  
- models folder:<br>
  - train_classifier.py: Script file that creates a ML model to classify the response to emergency messages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
