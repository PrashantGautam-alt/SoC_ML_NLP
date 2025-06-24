import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors
import string



#Preprocessing

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

#Loading DATASET

df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['Label', 'Message']
df['Tokens'] = df['Message'].apply(preprocess)


#Load Pretrained Word2Vec Model

w2v_path = 'GoogleNews-vectors-negative300.bin'
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

#Into vectors

def message_to_vector(tokens, model, vector_size=300):
    vectors = [model[word] for word in tokens if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

df['Vector'] = df['Tokens'].apply(lambda x: message_to_vector(x, w2v))

X = np.vstack(df['Vector'].values)
y = df['Label'].map({'ham': 0, 'spam': 1})  # Convert to binary labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


def predict_message_class(model, w2v_model, message):
    tokens = preprocess(message)
    vector = message_to_vector(tokens, w2v_model).reshape(1, -1)
    prediction = model.predict(vector)
    return 'spam' if prediction[0] == 1 else 'ham'


msg = "Congratulations! You've won a free iPhone!"
print(predict_message_class(clf, w2v, msg))  # Likely to return 'spam'

msg2 = "Hey, are we still meeting tomorrow?"
print(predict_message_class(clf, w2v, msg2))  # Likely to return 'ham'
