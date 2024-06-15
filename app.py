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

model_logistic = MLModel.load_model('./models/lg_model.pkl')
model_obj = NLPModel.load_model(
    './models/dl_model.keras', './models/tokenizer.pkl', './models/label_encoder.pkl')
model_dl = model_obj.model
tokenizer = model_obj.tokenizer


models_dict = {
    "Logistic Regression": model_logistic,
    "Deep Learning": model_obj
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
        f_text = Preprocess(f_text).preprocess()
        print(f_text)
        f_text = f_text['text'][0]
        print(f_text)

        f_model_name = request.form['model']
        f_model = models_dict[f_model_name]
        prediction = f_model.predict([f_text])
        f_result = dialects[prediction[0]]
        return render_template("home.html", model_names=f_model_names, result=f_result, text=f_text, model=f_model_name)

    return render_template("home.html", model_names=f_model_names, result="")


if __name__ == '__main__':
    app.run(debug=True)
