import pandas as pd

df = pd.read_csv("Tweets.csv")
df = df[['airline_sentiment', 'text']]  
print(df['airline_sentiment'].value_counts())

import re
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)                     # remove mentions
    text = re.sub(r'#', '', text)                        # remove hashtag symbol
    text = contractions.fix(text)                        # expand contractions
    text = re.sub(r"[^\w\s]", '', text)                  # remove punctuation
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t.isalpha()]
    return tokens

#Load Pre-trained Word2Vec

from gensim.models import KeyedVectors

w2v_path = 'GoogleNews-vectors-negative300.bin'
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

#Vectorize Tweets

import numpy as np

def tweet_to_vector(tokens, model, vector_size=300):
    vectors = [model[word] for word in tokens if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Apply preprocessing and vectorization
df['Tokens'] = df['text'].apply(preprocess_tweet)
df['Vector'] = df['Tokens'].apply(lambda x: tweet_to_vector(x, w2v))

#Train-Test-Split

from sklearn.model_selection import train_test_split

X = np.vstack(df['Vector'].values)
y = df['airline_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Multiclass Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

#Predict Function
def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess_tweet(tweet)
    vector = tweet_to_vector(tokens, w2v_model).reshape(1, -1)
    prediction = model.predict(vector)
    return prediction[0]
