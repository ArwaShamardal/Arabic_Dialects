from flask import Flask, request, jsonify, render_template

from scripts.data_fetching import DBHandler
from scripts.data_preprocessing import Preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from scripts.ml_model import MLModel
# from scripts.dl_model import NLPModel
import os


# model_ml = MLModel.load_model('./models/logistic_regression_model.pkl')
# model_dl = NLPModel.load_model('./models/dl_model.pkl')


app = Flask(__name__)


@app.route("/")
def home():
    model_names = []
    for file in os.listdir('./models'):
        if file.endswith('.pkl'):
            name = file.split('.')[0]
            model_names.append(name)

    return render_template("home.html", models=model_names)


if __name__ == '__main__':
    app.run(debug=True)
