# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB  
import re 
from sklearn.metrics import accuracy_score,recall_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer

#importation du document sous forme de csv
df = pd.read_csv('https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection', 
                 sep='\t',on_bad_lines='skip', header=None)




#renomage des columns
df.rename(columns={0:'classification',1:'sms'}, inplace=True)

df.head()


#supprimer les doublons
df=df.drop_duplicates()



#encodage de la colomn classification,1 our spam ,0 pour ham
label_encod = LabelEncoder()
#label_encod.fit(df['classification'])
#df['classification']=label_encod.fit_transform(df['classification'])


#definir la longueur des messages
def longueur(sms):
    df['longueur'] = df['sms'].str.len()
    df["sms"].apply(longueur)
    return 



#creation d'une liste de mots a rechercher
def mot_a_rechercher(sms):
    motsarechercher=['URGENT!', 'Quiz!', 'YOU!', 'Txt:', 'now!', 'Call ', 'Win', 'WINNER', '!!']
    df["mots"]=df["sms"].str.contains('|'.join(motsarechercher),case=False)
    return



#print(df["mots"].value_counts())
#print()



#creation d'une liste de mots a rechercher en rapport avec les liens
motsarechercherliens=['https', 'http','www','click here']


df["liens"]=df["sms"].str.contains('|'.join(motsarechercherliens),case=False)
#print(df["liens"].value_counts())
#print()



#creation d'une liste de mots a rechercher en rapport avec l'argent'
motargent = ['£', '€', '\$']

df["argent"]=df["sms"].str.contains('|'.join(motargent),case=False)
#print(df["argent"].value_counts())
#print()





#création d'une fonction qui recherche un numero de telephon dans une string
def verifier_numero_telephone(sms):
    #pattern des numeros de telephon a rechercher
    pattern = r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}"
    
    numeros = re.findall(pattern, sms)
    
    return bool(numeros)
#application de la fonction sur le dataframe et creation d'une colomne numero de telephon
#affiche 1 si un numero est presen , 0 si non
df['telephon'] = df['sms'].apply(verifier_numero_telephone)
#print(df['telephon'].value_counts())
#print()





#création d'une fonction qui recherche les adresses email dans une string
def verifier_email(sms):
    #pattern des numeros de telephon a rechercher
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email = re.search(pattern, sms)
    
    email = re.findall(pattern, sms)
    
    return bool(email)
#application de la fonction sur le dataframe et creation d'une colomne email
#affiche 1 si un numero est presen , 0 si non
df['email'] = df['sms'].apply(verifier_email)


def mot_maj_posible (string) :
    match = re.findall ( "[A-Z]{3}", string)
    if match:
        return True
    return False

df['maj'] = df.apply(lambda row: mot_maj_posible(row["sms"]) , axis=1)






y= df['classification']
X = df.drop(['classification','sms'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=99, stratify=df['classification'])




model = CategoricalNB()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

score = recall_score(y_test, prediction, pos_label='spam')
print("MultinomialNB:", round(score, 5))
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))


