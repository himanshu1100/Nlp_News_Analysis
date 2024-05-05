import nltk
import re
import bs4 as bs
import urllib.request
import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pickle
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import string
# Create your views here.
modelClassification = None
modelSentimentAnalysis = None
modelFakeDetection = None
le = []


def index(request):
    load_model("noting")
    if request.method == 'POST':
        data_str = request.body.decode('utf-8')
        news = json.loads(data_str)
        news = news.get('news')
        category = Classification(news)
        sentiment = Sentiment(news)
        summary = Summerization(news)
        # newsType = Fake(news)
        response_data = {
            'category': str(le[category[0]]),
            'summary': summary,
            'fake_real': 'newsType',
            'sentiment': sentiment,
        }

        return JsonResponse(response_data)

    return render(request, "classifier/index.html")


def load_model(request):

    from tensorflow.keras.models import load_model
    LoadData()
    global modelClassification, modelSentimentAnalysis, modelFakeDetection
    modelSentimentAnalysis = load_model(
        'models\RNN_sentiment_analysis_model.h5')
    modelClassification = joblib.load(
        'models\MultinomialNB_classification.pkl')
    with open('models\FakeLR.pkl', 'rb') as file:
        modelFakeDetection = pickle.load(file)


def decode_array(array):
    try:
        decoded_array = np.argmax(array, axis=1)
        return decoded_array
    except:
        raise


def Classification(input):
    return modelClassification.predict([input])


def Sentiment(input):
    with open('models/tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    

    new_text_seq = tokenizer.texts_to_sequences([input])
    # Use the max_len determined during training
    new_text_padded = pad_sequences(new_text_seq, padding='post', maxlen=35)
    predictions = modelSentimentAnalysis.predict(new_text_padded)
    predicted_class_index = predictions.argmax(axis=-1)
    if predicted_class_index[0] == 0:
        return ("Postive Sentiment")
    elif predicted_class_index[0] == 1:
        return ("Negative Sentiment")
    else:
        return ("Neutral Sentiment")


def LoadData():
    global le
    with open("models\Mapping_data.txt", "r") as file:
        le = []
        for line in file:
            string_before_arrow = line.split("-->")[0].strip()
            le.append(string_before_arrow)

def Summerization(article_text):


    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)
    print(article_text)
    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    sentence_list = nltk.sent_tokenize(article_text)
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
        maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
        sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    import heapq
    summary_sentences = heapq.nlargest(
        7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return(summary)


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    # This is where i remove the "()" from the text column. You can do in whatever way you want 
    # The key is to remove the "(Reuters)" string as it is present in all text of True.csv.
    # The Model during the training part can memorize it and perfrom great in training and badly when other testing input is given.
    text = re.sub('[()]','',text)
    text = re.sub('\\W',' ',text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def Fake(news):

    vectorization = TfidfVectorizer()
    with open('models\Vectorizer.pkl', 'rb') as file:
        vectorization = pickle.load(file)
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = modelFakeDetection.predict(new_xv_test)
    return output_label(pred_LR[0])