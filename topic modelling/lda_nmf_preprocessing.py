import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return text

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens, extra_stopwords=None):
    stopword_set = set(stop_words)
    if extra_stopwords:
        stopword_set.update(extra_stopwords)
    return [w for w in tokens if w not in stopword_set and len(w) > 2]

def lemmatize(tokens):
    return [lemmatizer.lemmatize(w) for w in tokens]

def preprocess_text(text, extra_stopwords=None):
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = lemmatize(tokens)
    tokens = remove_stopwords(tokens, extra_stopwords=extra_stopwords)
    return tokens