# Hate Speech Detection

This project focuses on detecting hate speech in social media, specifically within tweets. It aims to classify tweets into three categories: "Hate Speech", "Offensive Language", and "None".

## Project Structure

The project includes several key files that are essential for model training, evaluation, and deployment:

- `25k_dataset.csv`, `hatespeech_train.csv`, `prepared_hatespeech_train.csv`, `test_dataset.csv`, `train_dataset.csv`, `twits_25k_balanced.csv`, `twits_25k_preprocessed.csv`: Datasets used for model training and evaluation.
- `app.py`: The main Flask application file to deploy the model on a web interface.
- `bayesnaiveclass.ipynb`, `modelling.ipynb`, `notebook1c20a1fb3f.ipynb`, `palestine_hatespeech.ipynb`: Jupyter notebooks containing data analysis, model training, and evaluation.
- `naive_bayes_model.pkl`, `one_vs_rest_classifier.pkl`, `vectorizer.pkl`: Trained models and vectorizer saved for use in the Flask application.
- `index.html`, `result.html`: HTML templates for the web application's user interface.

## Installation

To run this project on your local machine, you will need Python and the following libraries:

- Flask
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter (to run the notebooks)

You can install these dependencies using `pip`:

```bash
pip install flask pandas scikit-learn matplotlib jupyter
