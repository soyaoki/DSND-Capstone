import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(filepath):
    '''
    Load data from input path.

    Args:
        database_filepath (str): The file path to load.

    Returns:
        X (pd.Series): A pandas Series containing the messages.
        Y (pd.DataFrame): A pandas DataFrame containing the categories.
    '''
    
    # load datasets
    df = pd.read_csv(filepath)

    return df


def clean_data(df):
    '''
    Clean and preprocess the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        df (pd.DataFrame): The cleaned DataFrame.
    '''
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Save the clean DataFrame as a SQLite database file.

    Args:
        df (pd.DataFrame): The cleaned DataFrame to be saved.
        database_filename (str): The file path of the SQLite database.

    Returns:
        None
    '''
    # save the clean dataframe as a database file
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DetectAIEssays', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 3:

        dataset_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    DATASET: {}\n'
              .format(dataset_filepath))
        df = load_data(dataset_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of datasets as the first argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the second argument. \n\nExample: python process_data.py '\
              'dataset_essays.csv DetectAIEssays.db')


if __name__ == '__main__':
    main()