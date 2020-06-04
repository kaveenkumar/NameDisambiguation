import json
import re
import unidecode
import pandas as pd
from textblob import TextBlob
from textblob import Word
import nltk.corpus
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Simplify words in string by transforming into singular forms
def lemma(str):
    result = []
    for word in TextBlob(str).split():
        # Using two different libraries for ensuring large coverage
        lemmatised = Word(word).lemmatize()
        WordNetLemmatizer().lemmatize(lemmatised)
        result.append(lemmatised)
    return " ".join(result)

# Create a list of n-sized lists with neighboring words in string
def nGram(str,n):
    wordList = str.split()
    ngram = []
    for word in range(len(wordList) - n + 1):
        ngram.append(wordList[word:word + n])
    return ngram

# Reduce string length and increase string meaningfulness

def preprocessLongString(str):
    if(type(str) is str):
        # Remove punctuation and special chars (keep dashes)
        str = re.sub(r"[.?!,;:/\\–\'\"()\[\]{}&|´`*+~#_°^$€<>%]", "", str)

        # Replace non-Unicode chars
        str = unidecode.unidecode(str)

        # Lowercase all
        str = str.lower()

        # Plural -> Singular (lemmatisation)
        str = lemma(str)

        try:
            # Tweak str to pandas dataframe for following operations
            dict = json.loads('{"0":"' + str + '"}', strict=False)
            strDF = pd.DataFrame([dict])
            strDF = strDF["0"]

            # Remove stop words (e.g. "is", "a", "the")
            stop = nltk.corpus.stopwords.words("english")
            strDF = strDF.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

            # Don't remove rare words (e.g. n=1) because in precise scientific wording, they can still be important

            # Remove single char words
            strDF = strDF.apply(lambda x: " ".join(x for x in x.split() if len(x) > 1))

            # Don't do spelling correction because it takes too long
            # It can be assumed that scientific papers are mostly error-free

            # Remove suffixes (stemming)
            st = PorterStemmer()
            strDF = strDF.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

            # Reduce all whitespaces, tabs, etc. to single spaces
            " ".join(strDF.values[0].split())

            return strDF.values[0]
        except:
            return str
    else:
        return ""