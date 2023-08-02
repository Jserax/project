import numpy as np
import pandas as pd
import joblib
import glob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')
stop_words.remove('not')
lemmatizer = WordNetLemmatizer()


def process_review(review, stopword=False, lemma=False) -> str:
    mapp = joblib.load('./mapp.joblib')
    review = review.lower()
    review = re.sub(re.compile('<.*?>'), '', review)
    review = ' '.join([mapp[w] if w in mapp else w for w in review.split(" ")])
    review = re.sub('[^A-Za-z0-9]+', ' ', review)
    if stopword:
        review = ' '.join([w for w in review.split(' ')
                           if w not in stop_words])
    if lemma:
        review = ' '.join([lemmatizer.lemmatize(w) for w in review.split(' ')])
    return review


def get_data(path) -> tuple[list, list]:
    review = []
    score = []
    files = glob.glob(f'{path}/*.txt')
    for file in files:
        score.append(int(file.split('_')[-1].split('.')[0]))
        with open(file, "r", encoding='utf-8') as f:
            review.append(f.read())
    return review, score


def get_dataframe(path, stopword=False, lemma=False) -> pd.DataFrame:
    review, score = get_data(f'{path}/pos/')
    pos = pd.DataFrame({'review': review, 'label': 1, 'score': score})
    review, score = get_data(f'{path}/neg/')
    neg = pd.DataFrame({'review': review, 'label': 0, 'score': score})
    df = pd.concat([pos, neg], ignore_index=True)
    df.review = df.review.apply(lambda x: process_review(x,
                                                         stopword=stopword,
                                                         lemma=lemma))
    return df
