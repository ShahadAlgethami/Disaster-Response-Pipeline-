

# Disaster Response Pipeline Project

I apply data engneering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python app/run.py`
3. Go to http://0.0.0.0:3001/

### Files

1. ETL Pipeline Preparation.ipynb is a Description for workspace/data/process_data.py
2. ML Pipeline Preparation.ipynb is a Description for workspace/model/train_classifier.py
3. workspace/data/process_data.py A data cleaning pipeline that:
   * Loads the messages and categories datasets
   * Merges the two datasets
   * Cleans the data
   * Stores it in a SQLite database
4. workspace/model/train_classifier.py A machine learning pipeline that:
   * Loads data from the SQLite database
   * Splits the dataset into training and test sets
   * Builds a text processing and machine learning pipeline
   * Trains and tunes a model using GridSearchCV
   * Outputs results on the test set
   * Exports the final model as a pickle file
