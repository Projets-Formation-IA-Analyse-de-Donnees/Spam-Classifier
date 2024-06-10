# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:52:28 2024

@author: naouf
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection', 
                 sep='\t', on_bad_lines='skip', header=None)

df.rename(columns={0: 'classification', 1: 'sms'}, inplace=True)
df['sms'] = df['sms'].str.lower()
df['sms'] = df['sms'].replace(to_replace='[^\w\s]', value=' ', regex=True)

stop_words = set(stopwords.words('english'))
df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['sms']), axis=1)
df['tokenized_sents2'] = df['tokenized_sents'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

lemmatizer = WordNetLemmatizer()
df['lem_text'] = df['tokenized_sents2'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
df['phrase'] = df['lem_text'].apply(lambda x: ' '.join(map(str, x)))

pipeline = Pipeline(steps=[
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

y = df['classification']
X_train, X_test, y_train, y_test = train_test_split(df['phrase'], y, test_size=0.2, random_state=50, stratify=y)

pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)

score = accuracy_score(y_test, prediction)
print("MultinomialNB:", round(score, 5))
print(confusion_matrix(y_test, prediction))

from sklearn.metrics import  accuracy_score
MultinomialNB: 0.97582