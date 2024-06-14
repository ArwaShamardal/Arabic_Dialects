from flask import Flask, request, jsonify, render_template

from scripts.data_fetching import DBHandler
from scripts.data_preprocessing import Preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from scripts.ml_model import MLModel
from scripts.dl_model import NLPModel
import os

import joblib
# TODO:: Add models here and add them to the models dictionary

model_logistic = MLModel.load_model('./models/logistic_regression_model.pkl')
dl_model= joblib.load('./models/dl_model.pkl')
# model_dl = NLPModel.load_model('./models/dl_model.pkl')

models_dict = {
    "Logistic Regression": model_logistic,
    "Deep Learning": dl_model
}

dialects = {
    "EG": "Egyptian Arabic",
    "LY": "Libyan Arabic",
    "LB": "Lebanese Arabic",
    "SD": "Sudanese Arabic",
    "MA": "Moroccan Arabic"
}


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def home():
    f_model_names = list(models_dict.keys())
    if request.method == 'POST':
        f_text = request.form['text']
        if f_text == "":
            return render_template("home.html", model_names=f_model_names, result="")
        model = request.form['model']
        model = models_dict[model]
        prediction = model.predict([f_text])
        f_result = dialects[prediction[0]]
        return render_template("home.html", model_names=f_model_names, result=f_result, text=f_text)

    return render_template("home.html", model_names=f_model_names, result="")


if __name__ == '__main__':
    app.run(debug=True)
