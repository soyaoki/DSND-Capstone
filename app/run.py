import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
import joblib
from sqlalchemy import create_engine

import sys
import os
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1])+"/models/")
print(sys.path)
from train_classifier import tokenize, StartingVerbExtractor, TextStatsExtractor

import random

def get_random_text_samples(df, condition_column='generated', condition_values=[0, 1], num_samples_per_condition=1):
    random_samples = []
    for value in condition_values:
        condition_indices = df[df[condition_column] == value].index.tolist()
        if condition_indices:
            random_indices = random.sample(condition_indices, min(num_samples_per_condition, len(condition_indices)))
            random_samples.extend(df.loc[random_indices, "text"].tolist())
    return random_samples

app = Flask(__name__)

'''
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
'''

# load data
engine = create_engine('sqlite:///../data/DetectAIEssays.db')
df = pd.read_sql_table('DetectAIEssays', engine)
df_test = pd.read_csv('../models/test_data.csv')

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    dataset_attribute_counts = [(df['generated']==0).sum()/len(df['generated']), (df['generated']==1).sum()/len(df['generated'])]
    dataset_attribute_names = ["Human", "LLM"]
    
    feature_tsne_1 = df_test['Feature_tsne_1']
    feature_tsne_2 = df_test['Feature_tsne_2']
    label = df_test['generated']
    text = df_test['text']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # graph1: Distribution of Message Genres
        {
            'data': [
                Bar(
                    x=dataset_attribute_names,
                    y=dataset_attribute_counts,
                    marker=dict(color=['#28a745', '#343a40'])
                )
            ],

            'layout': {
                'title': 'Distribution of Dataset',
                'yaxis': {
                    'title': "Rate[-]"
                },
                'xaxis': {
                    'title': "Essays written by"
                }
            }
        },
        
        # graph2: Distribution of Message Categories
        {
            'data': [
                Scatter(
                    x=feature_tsne_1,
                    y=feature_tsne_2,
                    mode='markers',
                    text=text,
                    marker=dict(size=16,
                                color=label,
                                colorscale=[[0, '#28a745'], [1, '#343a40']],
                    ),
                )
            ],

            'layout': {
                'title': 'Distribution of Features of Essays Written by Human and LLM',
                'yaxis': {
                    'title': "Feature 2 by t-SNE"
                },
                'xaxis': {
                    'title': "Feature 1 by t-SNE"
                },
                'height': 1000,
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # get random text samples
    random_texts = get_random_text_samples(df)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, random_texts=random_texts)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([str(query)])
    classification_proba = model.predict_proba([str(query)])
    classification_results = dict(zip(["Human", "LLM"], classification_proba[0]))

    print("---------------------------------------")
    print("query : {}".format(query))
    print("classification_proba : {}".format(classification_proba))
    print("classification_results : {}".format(classification_results))
    print("---------------------------------------")

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()