from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from tensorflow import keras
import numpy as np

app = Flask('Review classifier')


model = keras.models.load_model('./model/model')


def classify(review):
    answer = model.predict([review])
    label = 'positive' if answer[0].item() > 0.5 else 'negative'
    score = round(answer[1].item())
    return label, score


class ReviewForm(Form):
    review = TextAreaField('', [validators.DataRequired(),
                                validators.length(min=50)])


@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('index.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    label, score = classify(request.form['review'])
    return render_template('results.html', label=label, score=score)


if __name__ == '__main__':
    app.run(host="0.0.0.0:5000", debug=False)
