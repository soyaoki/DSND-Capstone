import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from sqlalchemy import create_engine

from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

from lightgbm import LGBMClassifier
from sklearn.manifold import TSNE

def load_data(database_filepath):
    '''
    Load data from a SQLite database.

    Args:
        database_filepath (str): The file path of the SQLite database.

    Returns:
        X (pd.Series): A pandas Series containing the messages.
        Y (pd.DataFrame): A pandas DataFrame containing the categories.
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name="DetectAIEssays", con=engine)
    X = df["text"]
    Y = df["generated"]
    return X, Y

def tokenize(text):    
    '''
    Tokenize the input text.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        clean_tokens (list): A list of cleaned and lemmatized tokens.
    '''
    # get list of all urls using regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # convert to lowercase
    text = text.lower()
    
    # remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Custom transformer for identifying whether the starting word in a sentence is a verb.

    Attributes:
        None

    Methods:
        starting_verb(text): 
            Tokenizes sentences, tags part of speech for the first word, 
            and checks if it is a verb or 'RT' for retweet.
            Returns 1 if true, 0 otherwise.

        fit(x, y=None): 
            Placeholder method required for compatibility with scikit-learn pipelines.
            Returns the instance itself.

        transform(X): 
            Applies the starting_verb function to all values in X.
            Returns a DataFrame with the results, replacing NaN values with 0.

    Usage:
        starting_verb_extractor = StartingVerbExtractor()
        result_df = starting_verb_extractor.transform(text_data)
    '''
    def starting_verb(self, text):
        '''
        Tokenizes sentences, tags part of speech for the first word, 
        and checks if it is a verb or 'RT' for retweet.
        
        Args:
            text (str): Input text containing one or more sentences.

        Returns:
            int: 1 if the starting word is a verb or 'RT', 0 otherwise.
        '''
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            if len(pos_tags) > 0:
                # index pos_tags to get the first word and part of speech tag
                first_word, first_tag = pos_tags[0]
            
                # return true if the first word is an appropriate verb or RT for retweet
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
                
            return 0

    def fit(self, x, y=None):
        '''
        Placeholder method required for compatibility with scikit-learn pipelines.

        Args:
            x: Input features.
            y: Target variable (not used in this implementation).

        Returns:
            StartingVerbExtractor: The instance itself.
        '''
        return self

    def transform(self, X):
        '''
        Applies the starting_verb function to all values in X.

        Args:
            X: Input text data.

        Returns:
            pd.DataFrame: DataFrame with the results, replacing NaN values with 0.
        '''
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)

        return pd.DataFrame(X_tagged).replace(np.nan, 0)

class TextStatsExtractor(BaseEstimator, TransformerMixin):
    '''
    Custom transformer for extracting basic text statistics from input text data.

    Attributes:
        None

    Methods:
        fit(x, y=None): 
            Placeholder method required for compatibility with scikit-learn pipelines.
            Returns the instance itself.

        transform(posts): 
            Applies the get_text_stats function to each element in the input text data.
            Returns a DataFrame with the extracted text statistics.

    Usage:
        text_stats_extractor = TextStatsExtractor()
        stats_df = text_stats_extractor.transform(text_data)
    '''
    def get_text_stats(self, text):
        '''
        Extracts basic text statistics from the input text.

        Args:
            text (str): Input text.

        Returns:
            pd.Series: Series containing word count, sentence count, and newline count.
        '''
        word_count = np.log(len(re.findall(r'\w+', str(text))))
        sentence_count = np.log(len(re.findall(r'[.!?]+', str(text))))
        newline_count = np.log(str(text).count('\n'))
        return pd.Series([word_count, sentence_count, newline_count], index=['N_words_log', 'N_sentences_log', 'N_newlines'])
    
    def fit(self, x, y=None):
        '''
        Placeholder method required for compatibility with scikit-learn pipelines.

        Args:
            x: Input features.
            y: Target variable (not used in this implementation).

        Returns:
            TextStatsExtractor: The instance itself.
        '''
        return self

    def transform(self, posts):
        '''
        Applies the get_text_stats function to each element in the input text data.

        Args:
            posts: Input text data.

        Returns:
            pd.DataFrame: DataFrame with the extracted text statistics.
        '''
        return pd.DataFrame(posts).apply(self.get_text_stats, axis=1)

def build_model():
    '''
    Build and return a machine learning model pipeline.

    Args:
        None

    Returns:
        cv (GridSearchCV): A grid search object for model tuning.
    '''
    # build a pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor()),
            ('text_stats', TextStatsExtractor())
        ])),
        ('clf', LGBMClassifier(random_state=42))
    ])
    
    # specify parameters for grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_features': [5, 100], 
        # 'clf__estimator__n_estimators': [50, 200],
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=2)
    print(cv)

    return cv

def evaluate_model(model, X_test, Y_test):
    '''
    Evaluate the performance of the model on the test data.

    Args:
        model: The trained machine learning model.
        X_test (pd.Series): Test features.
        Y_test (pd.DataFrame): True labels for the test data.

    Returns:
        None
    '''
    # predict test labels
    Y_test_pred = model.predict(X_test)
    Y_test_pred_proba = model.predict_proba(X_test)

    # print classification report on test data
    print("AUC: {}".format(roc_auc_score(Y_test, Y_test_pred_proba[:,1])))

def save_model(model, model_filepath):
    '''
    Save the trained model to a file using pickle.

    Args:
        model: The trained machine learning model.
        model_filepath (str): The file path to save the model.

    Returns:
        None
    '''
    # save the model
    with open(model_filepath, mode="wb") as f:
        pickle.dump(model, f)

def save_data(model, model_filepath, X_train, X_test, Y_train, Y_test):
    '''
    Save processed train and test data along with t-SNE-transformed features for visualization.

    Args:
        model (object): Trained machine learning model.
        model_filepath (str): Filepath for saving the data.
        X_train (pd.DataFrame): Training data features.
        X_test (pd.DataFrame): Testing data features.
        Y_train (pd.Series): Training data target variable.
        Y_test (pd.Series): Testing data target variable.

    Returns:
        None
    '''
    # save train data
    # df_train = pd.concat([X_train.reset_index(drop=True),Y_train.reset_index(drop=True)], axis=1)
    # df_train.to_csv("/".join(model_filepath.split("/")[:-1]) + "/train_data.csv")

    # save test data with t-SNE
    df_features = pd.DataFrame(model.best_estimator_.named_steps['features'].transform(X_test).toarray())
    X_reduced = TSNE(n_components=2, random_state=42).fit_transform(df_features)
    df_test_features_with_tsne = pd.concat([pd.concat([pd.concat([X_test.reset_index(drop=True),Y_test.reset_index(drop=True)], axis=1), df_features], axis=1), pd.DataFrame(X_reduced, columns=['Feature_tsne_1', 'Feature_tsne_2'])],axis=1)
    df_test_features_with_tsne.to_csv("/".join(model_filepath.split("/")[:-1]) + "/test_data.csv")

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print("")
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print("")
        
        print('Building model...')
        model = build_model()
        print("")

        print('Training model...')
        model.fit(X_train, Y_train)
        print("")

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)
        print("")

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print("")
        
        print('Saving data...\n')
        save_data(model, model_filepath, X_train, X_test, Y_train, Y_test)
        print("")

        print('Trained model saved!')
        print("")

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()