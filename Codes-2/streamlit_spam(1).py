import streamlit as st
import pandas as pd
import seaborn as sns
import ast
from Module_spam_v3 import model_IA
from Module_best_model import model_best
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM
from sklearn.metrics import * #plot_confusion_matrix
import matplotlib.pyplot as plt

####
# traitement des boutons et variables de session
####

# traitement du bouton de validité du modèle
def set_state(i):
    st.session_state.stage = i

if 'stage' not in st.session_state:
    st.session_state.stage = 0


# traitement du bouton de validité du fichier csv
def click_Rechercher():
    st.session_state.selected = True

if 'selected' not in st.session_state:
    st.session_state.selected = False


# traitement du bouton de validité du fichier csv
def click_button():
    st.session_state.clicked = True

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

####
# fonction
####
   
def model_selection (choix_model_select, dict_parametre_select) :

    if choix_model_select == 'BernoulliNB' or choix_model_select == 'BernoulliNB()' :
        model_select = BernoulliNB(**dict_parametre_select)
    if choix_model_select == 'CategoricalNB' or choix_model_select == 'CategoricalNB()' :
        model_select = CategoricalNB(**dict_parametre_select)
    if choix_model_select == 'ComplementNB' or choix_model_select == 'ComplementNB()' :
        model_select = ComplementNB(**dict_parametre_select)
    if choix_model_select == 'GaussianNB' or choix_model_select == 'GaussianNB()' :
        model_select = GaussianNB(**dict_parametre_select)
    if choix_model_select == 'MultinomialNB' or choix_model_select == 'MultinomialNB()' :
        model_select = MultinomialNB(**dict_parametre_select)
    if choix_model_select == 'SVC' or choix_model_select == 'SVC()' :
        model_select = SVC(**dict_parametre_select)
    if choix_model_select == 'SVR' or choix_model_select == 'SVR()' :
        model_select = SVR(**dict_parametre_select)
    if choix_model_select == 'LinearSVC' or choix_model_select == 'LinearSVC()' :
        model_select = LinearSVC(**dict_parametre_select)
    if choix_model_select == 'LinearSVR' or choix_model_select == 'LinearSVR()' :
        model_select = LinearSVR(**dict_parametre_select)
    if choix_model_select == 'NuSVC' or choix_model_select == 'NuSVC()' :
        model_select = NuSVC(**dict_parametre_select)
    if choix_model_select == 'NuSVR' or choix_model_select == 'NuSVR()' :
        model_select = NuSVR(**dict_parametre_select)
    if choix_model_select == 'OneClassSVM' or choix_model_select == 'OneClassSVM()' :
        model_select = OneClassSVM(**dict_parametre_select)

    return model_select

def print_graph (confusion_matrix_result, y_test, y_pred, N_value, train_score) :

    col1, col2 = st.columns([2, 3])

    with col1 :
        st.write('score = ', score)
        #st.write('confusion_matrix = \n', )

        plt.figure(figsize=(3, 2))
        sns.heatmap(confusion_matrix_result, annot=True, fmt="d", cmap="Blues", xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'] )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        st.pyplot(plt)
    with col2 :
        classification_report_result = classification_report( y_test, y_pred, output_dict = True )
        df_classification_report = pd.DataFrame(classification_report_result)
        df_classification_report = df_classification_report.transpose()
        df_classification_report = df_classification_report.drop(index = "accuracy")
        st.dataframe(df_classification_report)
        
    figure2 = plt.figure(figsize=(6,4))
    plt.plot(N_value, train_score.mean(axis=1), label='train score')
    plt.plot(N_value, val_score.mean(axis=1), label='validation score')
    plt.legend()
    st.pyplot(figure2)


####
# ouverture des fichiers
####

# ouverture du fichier model_parametre.json
df_json = pd.read_json("C:/Users/naouf/Documents/Naoufel/projet/SpamClassifier/regrouppement/model_parametre.json", encoding = "utf-8", dtype=False)
print("df = ", df_json)

# ouverture du fichier SMSSpamCollection
df_spam = pd.read_csv('https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection', 
                 sep='\t',on_bad_lines='skip', header=None)

# ouverture du fichier best_model.csv
df_best_model = pd.read_csv("C:/Users/naouf/Documents/Naoufel/projet/SpamClassifier/regrouppement/best_model_-_Copie.csv", sep=',',on_bad_lines='skip')

####
# slidbar :
# model à choisir parmi une liste des models 
# paramtètres à configurer selon le model choisi
# envoi des choix au programme python
# possible ajout d'un nouveau fichier .csv
####

#affichage du titre
st.sidebar.title ("Configuration du programme :")

# affichage choix model
st.sidebar.header("Choix du modèle :")

# Création de la liste de model
liste_model = df_json['model'].values.tolist()
liste_model = list(set(liste_model))

# affichage des différents model dans bar déroulant
choix_model = st.sidebar.selectbox('Choisis le modèle', liste_model)

# bouton pour selectionner model
st.sidebar.button('Validez le modèle', on_click=set_state, args=[1])

if st.session_state.stage >= 1 and st.session_state.stage < 3 :
    # action du bouton : quand clické : apparition des paramettes

    # affichage choix des paramettre
    st.sidebar.header("paramètre du modèle :")

    # Création de la liste des paramettre
    df_choix = df_json.loc[df_json['model']==choix_model]
    list_dict_parametre = df_choix.iloc[0]["parametre"]
    
    dict_parametre = {}
    for dict in list_dict_parametre :
        # affichage des différents paramettres avec input
        valeur = st.sidebar.text_input(f"choisi pour {dict['name']}")
        print(f'{dict["name"]} à pour valeur : !{valeur}!')
        # sauvegarde la valeur de l'input dans variable
        if dict['type'] == 'int' :
            try :
                val_type = int(valeur)
            except :
                print (f'erreur sur le type de {dict["name"]}')
                st.sidebar.error(f'Erreur sur le type de {dict["name"]}')
        if dict['type'] == 'int or array-like' :
            try :
                val_type = int(valeur)
            except :
                try :
                    val_type = f'"{valeur}"'
                except :
                    print (f'erreur sur le type de {dict["name"]}')
                    st.sidebar.error(f'Erreur sur le type de {dict["name"]}')
        if dict['type'] == 'float' :
            try :
                val_type = float(valeur)
            except :
                print (f'erreur sur le type de {dict["name"]}')
                st.sidebar.error(f'Erreur sur le type de {dict["name"]}')
        if dict['type'] == 'float or array-like' or dict['type'] == 'float or str' :
            try :
                val_type = float(valeur)
            except :
                try :
                    val_type = f'"{valeur}"'
                except :
                    print (f'erreur sur le type de {dict["name"]}')
                    st.sidebar.error(f'Erreur sur le type de {dict["name"]}')
        if dict['type'] == 'str' :
            try :
                val_type = str(valeur)
            except :
                print (f'erreur sur le type de {dict["name"]}')
                st.sidebar.error(f'Erreur sur le type de {dict["name"]}')
        if dict['type'] == 'bool' :
            try :
                valeur = valeur.lower().capitalize()
                val_type = bool(valeur)
            except :
                print (f'erreur sur le type de {dict["name"]}')
                st.sidebar.error(f'Erreur sur le type de {dict["name"]}')
        if dict['type'] != 'bool' and dict['type'] != 'str' and dict['type'] != 'float' and dict['type'] != 'int' :
            try :
                val_type = f'"{valeur}"'
            except :
                print (f'erreur sur le type de {dict["name"]}')
                st.sidebar.error(f'Erreur sur le type de {dict["name"]}')
        
        if valeur == None or valeur == "" :
            if dict["default_val"] == "TRUE" :
                print("TRUE identidier")
                dict_parametre[dict["name"]] = True
            if dict["default_val"] == "FALSE" :
                print("FALSE identidier")
                dict_parametre[dict["name"]] = False
            if dict["default_val"] == "NULL" :
                dict_parametre[dict["name"]] = None
                print("je passe ici 2")
            if dict["default_val"] != "TRUE" and dict["default_val"] != "FALSE" and dict["default_val"] != "NULL" :
                dict_parametre[dict["name"]] = dict["default_val"]
                print("je passe ici 1")
        else :
            dict_parametre[dict["name"]] = val_type
        
        # affichage de la description du paramètre
        st.sidebar.caption(dict['describe'])

    # bouton pour selectionner les paramètre
    st.sidebar.button('Validez les parametres', on_click=set_state, args=[2])

# affiche le texte de recherche
st.sidebar.header("Rechercher le modèle avec les parametres les plus performants :")

# boutton pour valider la recherche
st.sidebar.button('Rechercher', on_click=click_Rechercher)

# boutton pour selectionner les paramètre
st.sidebar.write('Utiliser le modèle avec les paramètres les plus performants')
st.sidebar.button('Utiliser', on_click=set_state, args=[3])

# affiche choix d'un nouveau fichier csv
st.sidebar.header("Nouveau data frame :")

# champs pour le lien du fichier csv
path_csv_new = st.sidebar.text_input(f"lien du nouveau fichier csv")

# explications
st.sidebar.write(f"Pour pouvoir ouvrir le fichier, il doit être un CSV, les colonnes doivent être séparé par des 'v'.")
st.sidebar.write(f"Il doit également contenir 2 colonnes nommée 0 pour le type 'spam' ou 'ham' et 1 pour les messages.")

# boutton pour valider le fichier csv
st.sidebar.button('Validez le lien et le modèle', on_click=click_button)


####
# page principale :
# affiche les résultats de l'entrainement
# test avec le nouveau fichier csv
####

# affiche le titre de la fenetre principale
st.title ("Résultat du programme")

# si le model et les parametres ont bien été renplis :
if st.session_state.stage >= 2 and st.session_state.stage < 3 :
    st.header("Résultat du modèle choisi :")

    model = model_selection(choix_model, dict_parametre)
    
    # appelle de la classe model avec df_spam et model_commande
    model_spam = model_IA(df_spam, model, test=False)
    score = model_spam.score
    confusion_matrix_result = model_spam.confusion_matrix
    classification_report_r = model_spam.classification_report 
    N_value = model_spam.N
    train_score = model_spam.train_score 
    val_score = model_spam.val_score 
    x_train = model_spam.x_train 
    y_train = model_spam.y_train 
    x_test = model_spam.x_test 
    y_test = model_spam.y_test
    y_pred = model_spam.y_pred

    ####
    #
    # print graphe       
    #
    ####
    
    print_graph (confusion_matrix_result, y_test, y_pred, N_value, train_score)



if st.session_state.stage >= 3:
    st.header("Résultat du modèle idéal :")
    
    best_model = df_best_model.iloc[df_best_model["Best_Accuracy"].idxmax()]

    choix_model = best_model['model']
    dict_param_best = best_model['Best_Parameters']

    dict_parametre = ast.literal_eval(dict_param_best)

    model = model_selection(choix_model, dict_parametre)
    
    # appelle de la classe model avec df_spam et model_commande
    model_spam = model_IA(df_spam, model, test=False)
    score = model_spam.score
    confusion_matrix_result = model_spam.confusion_matrix
    classification_report_r = model_spam.classification_report 
    N_value = model_spam.N
    train_score = model_spam.train_score 
    val_score = model_spam.val_score 
    x_train = model_spam.x_train 
    y_train = model_spam.y_train 
    x_test = model_spam.x_test 
    y_test = model_spam.y_test
    y_pred = model_spam.y_pred

    ####
    #
    # print graphe       
    #
    ####
    
    print_graph (confusion_matrix_result, y_test, y_pred, N_value, train_score)
    

# si le nouveau fichier est bien validé 
if st.session_state.clicked :
    st.header("Résutlat du nouveau data frame :")

    try :
        df_new_fichier = pd.read_csv(path_csv_new, sep=',',on_bad_lines='skip')
        
        if st.session_state.stage >= 2 and st.session_state.stage < 3 :
            st.header("Résultat du modèle choisi avec les nouvelles données :")

            score = model_spam.score
            confusion_matrix_result = model_spam.confusion_matrix
            classification_report_r = model_spam.classification_report 
            N_value = model_spam.N
            train_score = model_spam.train_score 
            val_score = model_spam.val_score 
            x_train = model_spam.x_train 
            y_train = model_spam.y_train 
            x_test = model_spam.x_test 
            y_test = model_spam.y_test
            y_pred = model_spam.y_pred
            
            new_x_test, new_y_test, new_y_pred, new_score, new_confusion_matrix, new_N, new_train_score, new_val_score = model_spam.test_new_csv(df_new_fichier)

            ####
            #
            # print graphe       
            #
            ####
            
            print_graph (new_confusion_matrix, new_y_test, new_y_pred, new_N, new_train_score)

        if st.session_state.stage >= 3:
            st.header("Résultat du modèle idéal avec les nouvelles données :")

            score = model_spam.score
            confusion_matrix_result = model_spam.confusion_matrix
            classification_report_r = model_spam.classification_report 
            N_value = model_spam.N
            train_score = model_spam.train_score 
            val_score = model_spam.val_score 
            x_train = model_spam.x_train 
            y_train = model_spam.y_train 
            x_test = model_spam.x_test 
            y_test = model_spam.y_test
            y_pred = model_spam.y_pred
            
            new_x_test, new_y_test, new_y_pred, new_score, new_confusion_matrix, new_N, new_train_score, new_val_score = model_spam.test_new_csv(df_new_fichier)

            ####
            #
            # print graphe       
            #
            ####

            print_graph (new_confusion_matrix, new_y_test, new_y_pred, new_N, new_train_score)
        
        else :
            st.write("le modèle n'est pas validé")
    
    except :
        st.error("Erreur : le fichier est introuvable ou impossible à lire")

# si la recherche est cliquée 
if st.session_state.selected :
    print("session_state_rechercher chiker")
    
    st.table(df_best_model)

    best_model = df_best_model.iloc[df_best_model["Best_Accuracy"].idxmax()]

    choix_model = best_model['model']
    dict_param_best = best_model['Best_Parameters']

    dict_parametre = ast.literal_eval(dict_param_best)

    model = model_selection(choix_model, dict_parametre)

    st.write('le meilleur modèle est = ')
    st.write(model)
    st.write('ses meilleurs paramètres sont = ')
    st.write(dict_parametre)

    # bouton pour validé le fichier csv
    #st.button('lancer à nouveau la recherche')
    if st.button('lancer à nouveau la recherche') :
        model_best()


####
#
# st.markdown("this is the header")
# st.subheader("this is the subheader")
# st.caption("this is the caption")
# st.code("x=2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''') 
#
# st.checkbox('yes')
# st.button('Click')
# st.radio('Pick your gender',['Male','Female'])
# st.selectbox('Pick your gender',['Male','Female'])
# st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
# st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
# st.slider('Pick a number', 0,50)
#
# st.success("You did it !")
# st.error("Error")
# st.warning("Warning")
# st.info("It's easy to build a streamlit app")
# st.exception(RuntimeError("RuntimeError exception"))
#
####
