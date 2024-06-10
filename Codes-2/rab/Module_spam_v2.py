import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

# Classe definissant les chemins du labyrinthe
class model_IA :
    score = None
    confusion_matrix = None
    classification_report = None 
    N = None 
    train_score = None
    val_score = None 
    x_train = None 
    y_train = None 
    x_test = None
    y_test = None 
    y_pred = None

    # Constructeur, avec en argumant le point d'apparition de la tuille
    def __init__(self, df_init, model_parametre_init, k_value_init ):
        self.df= df_init
        self.df.rename(columns={0:'classification'}, inplace=True)
        self.df.rename(columns={1:'sms'}, inplace=True)

        self.model_parametre = model_parametre_init
        #self.model_parametre.VarianceThreshold(threshold=0)

        self.k_value = k_value_init

        self.model_pipeline = make_pipeline( SelectKBest(f_classif, k=self.k_value), self.model_parametre )
        self.model_evaluation(self.model_pipeline, self.df)

    ####
    # fonction :
    ####
    def traitement_na_duplic (self, df) :
        """
        entrée : un data frame
        sortie : 2 data frame = 'principal' et 'na'
        ---------------------------
        je regarde dans de df d'entré et regarde si il y à des na
        si oui colone 1  => j'ajout la ligne a un df de sorti 'na' et suprime du df de sortie principal
        si oui colone 2  => je suprime la ligne du df de sortie principal
        """
        df_na = None
        df = df.drop_duplicates()
        
        if df['classification'].isna().any() == True :
            df_na = df.loc[df['classification'].isna()]
            
        if df['sms'].isna().any() == True :
            df.drop(df[df['sms'].isna() == True].index, inplace=True)
        
        return df, df_na

    def mot_cle_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        j'ai une liste de mots clés
        je crée le pattern des mots clés
        je recherche dans la colonne 'sms' si je trouve le patern  
        """
        mot_cles = ['URGENT!', 'Quiz!', 'YOU!', 'Txt:', 'now!', 'Call ', 'Win', 'WINNER', '!!', 'For sale', 'FREE!', 'PRIVATE!', 'Account', 'Latest News!']
        pattern = re.compile(r"(?=("+'|'.join(mot_cles)+r"))", re.IGNORECASE)
        match = re.findall(pattern, sms)
        return bool(match)

    def argent_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        j'ai une liste de mots clés
        je crée le pattern des mots clés
        je recherche dans la colonne 'sms' si je trouve le patern  
        """
        mot_cles = ['£', '€', '\$']
        pattern = re.compile(r"(?=("+'|'.join(mot_cles)+r"))", re.IGNORECASE)
        match = re.findall(pattern, sms)
        return bool(match)

    def telephone_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        crée le pattern des numero de tel
        recherche dans une chane de carractère si je trouve le patern    
        """
        pattern = re.compile(r"(\+\d{1,3})?\s?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,4}")
        match = re.search(pattern, sms)
        if match:
            return True
        return False

    def email_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        je crée le pattern des e-mail
        je recherche dans la colonne 'sms' si je trouve le patern    
        """
        pattern = r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"
        match = re.findall(pattern, sms)
        return bool(match)

    def lien_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        j'ai une liste de mots clés
        je crée le pattern des mots clés
        je recherche dans la colonne 'sms' si je trouve le patern  
        """
        mot_cles = ['http', 'https', 'www.', 'click here']
        pattern = re.compile(r"(?=("+'|'.join(mot_cles)+r"))", re.IGNORECASE)
        match = re.findall(pattern, sms)
        return bool(match)

    def mot_maj_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        je crée le pattern des majuscules
        je recherche dans la colonne 'sms' si je trouve le patern  
        """
        pattern = "[A-Z]{3}"
        match = re.findall(pattern, sms)
        return bool(match)

    def long_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : int
        ---------------------
        je mesure la taille de chaque ligne de la colonne 'sms'
        """
        return int(len(sms))

    def encodage_df (self, df) :
        """
        entrée : un data frame
        sortie : un data frame
        ---------------------------
        je lance l'encodage de la colonne 'classification'
        je crée la colonne 'mot_cles' grace à la fonction 'mot_cle_posible'
        je crée la colonne 'argent' grace à la fonction 'argent_posible'
        je crée la colonne 'telephone' grace à la fonction 'telephone_posible'
        je crée la colonne 'email' grace à la fonction 'email_posible'
        je crée la colonne 'lien' grace à la fonction 'lien_posible'
        je crée la colonne 'maj' grace à la fonction 'mot_maj_posible'
        je crée la colonne 'long' grace à la fonction 'long_posible'
        """
        label_encod = LabelEncoder()
        df['classification'] = label_encod.fit_transform(df['classification'])
        
        df['mot_cles'] = df['sms'].apply(self.mot_cle_posible)
        df['argent'] = df['sms'].apply(self.argent_posible)
        df['telephone'] = df['sms'].apply(self.telephone_posible)
        df['email'] = df['sms'].apply(self.email_posible)
        df['lien'] = df['sms'].apply(self.lien_posible)
        df['maj'] = df['sms'].apply(self.mot_maj_posible)
        df['long'] = df['sms'].apply(self.long_posible)
        
        df['mot_cles'] = label_encod.fit_transform(df['mot_cles'])
        df['argent'] = label_encod.fit_transform(df['argent'])
        df['telephone'] = label_encod.fit_transform(df['telephone'])
        df['email'] = label_encod.fit_transform(df['email'])
        df['maj'] = label_encod.fit_transform(df['maj'])
        df['lien'] = label_encod.fit_transform(df['lien'])

        return df

    def train_et_test (self, df) :
        """
        entrée : un data frame, pourcentage de valeur à metre dans df_test
        sortie : 2 data frame = 'traine' et 'test' et leur x et y respectif
        ---------------------------
        avec train_test_split, je crée sépare en 2 le df : trainSet et  testSet
        je crée le y_train
        je crée le y_test
        je crée le x_train
        je crée le x_test
        """
        trainSet, testSet = train_test_split(df, test_size=0.2, random_state=0, stratify=df['classification'])

        trainSet.drop('sms', axis=1, inplace=True)
        testSet.drop('sms', axis=1, inplace=True)

        y_train = trainSet['classification']

        y_test = testSet['classification']

        x_train = trainSet[['mot_cles', 'argent', 'telephone', 'email', 'lien', 'maj', 'long']]

        x_test = testSet[['mot_cles', 'argent', 'telephone', 'email', 'lien', 'maj', 'long']]

        return trainSet, x_train, y_train, testSet, x_test, y_test

    def preprocessing_df (self, df):
        """
        entrée : un data frame
        sortie : 4 data frame et les fitures et tagets
        ---------------------------
        je lance le netoyage des données grace à la fonction taitement_na_duplic
        je lance l'encodage des données grace à la fonction encodage_df
        je lance la création des fiture et targets grace à la fonction train_et_test
        """
        df_na = None
        df_p, df_na = self.traitement_na_duplic(df)
        df_e = self.encodage_df(df_p)
        trainSet, x_train, y_train, testSet, x_test, y_test = self.train_et_test(df_e)
        
        return df_e, df_na, trainSet, x_train, y_train, testSet, x_test, y_test

    def model_evaluation (self, model, df) :
        """
        entrée : un data frame, pourcentage de valeur à metre dans df_test
        sortie : 2 data frame = 'traine' et 'test'
        ---------------------------
        avec train_test_split, je crée sépare en 2 le df
        pourcent est le pourcentage de valeur à metre dans df_test
        """
        df_f, df_na, trainSet, x_train, y_train, testSet, x_test, y_test = self.preprocessing_df(df)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print ('score = ', score)
        print ('confusion_matrix = \n', confusion_matrix(y_test, y_pred))
        print ('classification_report = \n', classification_report(y_test, y_pred))

        N, train_score, val_score = learning_curve(model, x_train, y_train,
                                                cv = 4, scoring='f1',
                                                train_sizes=np.linspace(0.1, 1, 10))
        
        plt.figure(figsize=(12,8))
        plt.plot(N, train_score.mean(axis=1), label='train score')
        plt.plot(N, val_score.mean(axis=1), label='validation score')
        plt.legend()

        self.score = score
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
        self.N = N
        self.train_score = train_score
        self.val_score = val_score
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = y_pred