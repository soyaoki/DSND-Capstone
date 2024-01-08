# Capstone Project ~ Who Wrote This Essay, Human or LLM? ~

# 1. Project Overview

This repository contains a Flask web app for essay classification, utilizing data from [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview). The project involves building a machine learning pipeline to classify essays, for protecting plagiarism by LLM. The web app allows users to input new essays and receive classification results whether human or LLM wrote the essay, accompanied by visualizations for enhanced data interpretation.

__Key Features__:
- Flask web app for essay classification.
- Machine learning pipeline for the classification.
- User-friendly interface for everyone to input essay and obtain classification results.
- Visualizations to provide insights into the data.

Below are a few screenshots of the web app.

## __Main Page__
![](/main-page-1.png)
![](/main-page-2.png)
## __Essays Classification Result Page__
![](/classification-result-page-Human.png)
![](/classification-result-page-LLM.png)

# 2. Files in the repository

- `Jupyter-Notebook`
    - `llm-detect-ai-generated-text.ipynb` : A notebook for understanding data and designing ETL step and ML step. 
- `app`
    - `templates`
        -  `master.html`  : HTML file of main page of the web app.
        -  `go.html` : HTML file of classification result page of the web app.
    - `run.py` : A python script for Flask web app using SQLite data and a pre-trained classifier for essay classification.
- `data`
    - `dataset_essays.csv` : Data for training included essays written by human and LLM.
    - `process_data.py` : A python script to prepare clean data for ML. It's used at ETL(Extract, Transfor, Load) step.
- `models`
    - `train_classifier.py` : A python script to train an essay classifier. It's used at ML(Machine Learning) step.

# 3. Instructions:
1. Install requirements.
   
   `pip install -r requirements.txt`
   
3. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
      
        `python data/process_data.py data/dataset_essays.csv data/DetectAIEssays.db`
      
    - To run ML pipeline that trains classifier and saves
      
        `python models/train_classifier.py data/DetectAIEssays.db models/classifier.pkl`

4. Go to `app` directory: `cd app`

5. Run your web app: `python run.py`

6. Click [here](http://0.0.0.0:3000/) to open the web app. If you're working at Udacity's workspace, click the `PREVIEW` button to open the homepage.

# 4. Libraries used

`numpy`, `pandas`, `sklearn`, etc.

For more detail, see [requirements.txt](/requirements.txt)

# 5. Blog post

https://soyaoki.github.io/2024/01/08/who-wrote-this-essay-human-or-llm.html

# 6. Necessary acknowledgments

[LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview)

