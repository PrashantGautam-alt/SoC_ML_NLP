import math
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Corpus
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

def tokenize(doc):
    return doc.lower().split()

docs_token = [tokenize(doc) for doc in corpus]
vocab = sorted(set(word for doc in docs_token for word in doc))


#document frequency

df = defaultdict(int)

for word in vocab:
    for doc in docs_token:
        if word in doc:
            df[word] += 1

#Compute IDF

N = len(corpus)

idf = {word: math.log(N/df[word]) for word in vocab}


#Compute TF and TF-IDF

tfidf = []

for doc in docs_token:
    tfidf_doc = {}
    total_words = len(doc)
    word_counts = defaultdict(int)

    for word in doc:
        word_counts[word] += 1

    for word in vocab:
        tf = word_counts[word] / total_words
        tfidf_doc[word] = tf*idf[word]

    tfidf.append(tfidf_doc)