import streamlit as st
import pandas as pd
import Module_spam_v2
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM
import matplotlib.pyplot as plt

####
# traitement des boutons et variable de session
####

# traitement du bouton de validité du fichier csv
def click_button():
    st.session_state.clicked = True

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

# traitement du bouton de validité du modèle
def set_state(i):
    st.session_state.stage = i

if 'stage' not in st.session_state:
    st.session_state.stage = 0

####
# slidbar :
# model à choisir parmi une liste des models 
# paramtètres à configurer selon le model choisi
# envoi des choix au programme python
# posible ajout d'un nouveau fichier .csv
####

# ouverture du fichier model_parametre.json
df_json = pd.read_json("C:/Users/naouf/Documents/Naoufel/projet/SpamClassifier/regrouppement/model_parametre.json", encoding = "utf-8", dtype=False)
print("df = ", df_json)



#affichage du titre
st.sidebar.title ("Configuration du programme :")


# affichage choix model
st.sidebar.header("Choix du modèle :")

# Création de la liste de model
liste_model = df_json['model'].values.tolist()
liste_model = list(set(liste_model))

# affichage les diférent model dans bar déroulent
choix_model = st.sidebar.selectbox('Choisi le modèle', liste_model)

# bouton pour selectioner model
st.sidebar.button('Validez le modèle', on_click=set_state, args=[1])

if st.session_state.stage >= 1:
    # action du bouton : quand clické : apparition des paramettes

    # affichage choix des paramettre
    st.sidebar.header("paramètre du modèle :")

    # Création de la liste des paramettre
    df_choix = df_json.loc[df_json['model']==choix_model]
    list_dict_parametre = df_choix.iloc[0]["parametre"]
    
    # affichage le nombre de features à prendre en compte
    k_input = st.sidebar.text_input(f"choisi le nombre de feature")
    try :
        if int(k_input) > 0 :
            k_valeur = int(k_input)
        else :
            print (f'erreur : tu dois choisir un nombre positif')
            st.sidebar.error(f'erreur : tu dois choisir un nombre positif')
    except :
        print (f'erreur : tu dois choisir un nombre entier')
        st.sidebar.error(f'erreur : tu dois choisir un nombre entier')
    
    dict_parametre = {}
    for dict in list_dict_parametre :
        # affichage les diférent paramettre avec input
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
        
        # affichage de la decription du parametre
        st.sidebar.caption(dict['describe'])

    # bouton pour selectioner les paramettre
    st.sidebar.button('Validez les paramettres', on_click=set_state, args=[2])


# affiche choix d'un nouveau fichier csv
st.sidebar.header("Nouveau data frame :")

# champs pour le mien du fichier csv
path_csv_new = st.sidebar.text_input(f"lien du nouveau fichier csv")
    
# bouton pour validé le fichier csv
st.sidebar.button('Validez le lien et le modèle', on_click=click_button)


####
# page principal :
# affiche les résultats de l'entrainement
# test avec le nouveau fichier csv
####

# affiche le titre de la fenetre principal
st.title ("Résultat du programme")

# si le model et les parametre ont bien été renplis :
if st.session_state.stage >= 2:
    st.header("Résultat du modèle choisi :")
    
    if choix_model == 'BernoulliNB' :
        model = BernoulliNB(**dict_parametre)
    if choix_model == 'CategoricalNB' :
        model = CategoricalNB(**dict_parametre)
    if choix_model == 'ComplementNB' :
        model = ComplementNB(**dict_parametre)
    if choix_model == 'GaussianNB' :
        model = GaussianNB(**dict_parametre)
    if choix_model == 'MultinomialNB' :
        model = MultinomialNB(**dict_parametre)
    if choix_model == 'SVC' :
        model = SVC(**dict_parametre)
    if choix_model == 'SVR' :
        model = SVR(**dict_parametre)
    if choix_model == 'LinearSVC' :
        model = LinearSVC(**dict_parametre)
    if choix_model == 'LinearSVR' :
        model = LinearSVR(**dict_parametre)
    if choix_model == 'NuSVC' :
        model = NuSVC(**dict_parametre)
    if choix_model == 'NuSVR' :
        model = NuSVR(**dict_parametre)
    if choix_model == 'OneClassSVM' :
        model = OneClassSVM(**dict_parametre)
    
    df_spam = pd.read_csv('https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection', 
                 sep='\t',on_bad_lines='skip', header=None)
    # appel de la classe model avec df_spam et model_commande
    model_spam = Module_spam_v2.model_IA(df_spam, model, k_valeur)
    score = model_spam.score
    confusion_matrix = model_spam.confusion_matrix
    classification_report = model_spam.classification_report 
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

    st.write('score = ', score)
    st.write('confusion_matrix = \n', confusion_matrix(y_test, y_pred))
    st.caption('classification_report = \n')
    code = classification_report(y_test, y_pred)
    st.code(code, language='python')     

    figure1 = plt.figure(figsize=(12,8))
    plt.plot(N_value, train_score.mean(axis=1), label='train score')
    plt.plot(N_value, val_score.mean(axis=1), label='validation score')
    plt.legend()
    st.pyplot(figure1)


# si le nouveau fichier est bien validé 
if st.session_state.clicked :
    st.header("Résutlat du nouveau data frame :")
    st.write(f'le chemin du fichier est : !{path_csv_new}!')
    try :
        df_new_fichier = pd.read_csv('path_csv_new', sep=',',on_bad_lines='skip', header=None)
        # appel de la classe model avec df_new_fichier et model_commande si existe, sinon => model_commande par défault
    except :
        st.error("Erreur : le fichier est introuvable ou imposible à lire")




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