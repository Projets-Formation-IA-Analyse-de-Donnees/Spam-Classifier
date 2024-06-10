import pandas as pd
import Module_spam_v3
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM
from sklearn.model_selection import GridSearchCV

# Classe definissant la recherche du meilleur modèle
class model_best :

    # Constructeur, avec aucun argument
    def __init__(self):
        #ouverture des fichier
        self.df_json = pd.read_json('C:/Users/sandy/Documents/devIA/brief/SPAM/rendu/model_parametre.json', encoding = "utf-8")
        self.path_csv = "C:/Users/sandy/Documents/devIA/brief/SPAM/rendu/best_model.csv"

        # liste des modèles
        liste_model = self.df_json['model'].values.tolist()
        self.liste_model = list(set(liste_model))

        #fonction qui initialise les parametre d'un modèle (par default : le modèle est BernoulliNB() )
        self.init_model()
        
        # initialise le data frame des meilleurs modèles
        self.df_best_model = pd.DataFrame(data = {'model': [],
                               'Best_Parameters' : [],
                               'Best_Accuracy' : []})
        
        # fonction qui cherche pour tous les modèles de la listes, les paramètres idéals
        self.grid_serch_best_model()
        

    def init_model (self) :
        pd.set_option('mode.chained_assignment', None)
        df = pd.read_csv('https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection', 
                        sep='\t',on_bad_lines='skip', header=None)

        # appelle de la classe model avec df_spam et model_commande
        model = BernoulliNB()
        model_spam = Module_spam_v3.model_IA(df, model, test=False)
        self.df_prepro = model_spam.df
        self.x_train = model_spam.x_train
        self.y_train = model_spam.y_train
        self.x_test = model_spam.x_test
        self.y_test = model_spam.y_test

    def grid_serch_best_model (self) :
        #pour tous les modèles de la liste :
        for count in range(0,len(self.liste_model)) :

            # choisi le modèle
            if count == 0 :
                model = BernoulliNB()
                choix_model = 'BernoulliNB'
            if count == 1 :
                model = CategoricalNB()
                choix_model = 'CategoricalNB'
            if count == 2 :
                model = ComplementNB()
                choix_model = 'ComplementNB'
            if count == 3 :
                model = GaussianNB()
                choix_model = 'GaussianNB'
            if count == 4 :
                model = MultinomialNB()
                choix_model = 'MultinomialNB'
            if count == 5 :
                model = SVC()
                choix_model = 'SVC'
            if count == 6 :
                model = LinearSVC()
                choix_model ='LinearSVC'
            if count == 7 :
                model = NuSVC()
                choix_model = 'NuSVC'
            if count == 8 :
                model = NuSVR()
                choix_model = 'NuSVR'
            if count == 9 :
                model = OneClassSVM()
                choix_model = 'OneClassSVM'
            if count == 10 :
                model = SVR()
                choix_model = 'SVR'
            if count == 11 :
                model = LinearSVR()
                choix_model = 'LinearSVR'
            
            # initialise la liste des parametres de grid_serch pour le modèle
            df_choix = []
            params_str = []
            df_choix = self.df_json.loc[self.df_json['model']==choix_model]
            params_str = df_choix.iloc[0]["param_grid"]
            
            # transforme la liste de parametre (str) en dictionnaire
            params = {}
            for element in params_str :
                value_type = params_str[element].split(",")
                
                value_type_type = []
                for i in value_type :
                    print("i =", i)
                    if i == 'None' :
                        i_type = None
                    if i == 'True' :
                        i_type = True
                    if i == 'False' :
                        i_type = False
                    if i != 'None' and i != 'True' and i != 'False' :
                        try :
                            i_type = int(i)
                        except :
                            try :
                                i_type = float(i)
                            except :
                                i_type = str(i)
                                    
                    print('i_type =',type(i_type))
                    value_type_type.append(i_type)
                
                params[element] = value_type_type
            
            # test : lance la recherche des parametres idéals
            # si succés : l'ajoute à df_best_model l'enregistre dans le fichier best_model
            # si echec : l'ajoute à df_best_model l'enregistre dans le fichier best_model avec comme paramete "error" et comme accuracy 0
            try :
                model_grid = GridSearchCV(model, param_grid=params, n_jobs=-1, cv=5)#, verbose=5
                model_grid.fit(self.x_train,self.y_train)
                
                Best_Parameters = model_grid.best_params_
                Best_Accuracy = model_grid.best_score_
                
                self.df_best_model.loc[count] = [model, Best_Parameters, Best_Accuracy]
                count += 1
                self.df_best_model.to_csv(self.path_csv, index=False)
            
            except :
                
                self.df_best_model.loc[count] = [model, "error", 0]
                count += 1
                self.df_best_model.to_csv(self.path_csv, index=False)

        