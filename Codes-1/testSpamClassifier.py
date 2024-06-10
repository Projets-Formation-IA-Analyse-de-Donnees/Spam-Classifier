# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:39:14 2024

@author: naouf
"""

import streamlit as st
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection',names=("classification","sms"), 
                 sep='\t',on_bad_lines='skip', header=None)


df['longueur'] = df['sms'].apply(len)
df['nombredemots'] = df['sms'].str.split().apply(len)



radio=st.sidebar.radio('Choisissez votre catégorie:', df['classification'].unique())

radio2=df.loc[df.classification==radio]

nombredemessage=st.sidebar.slider('Choisissez le nombre de messages que vous souhaitez afficher:', min_value=0, max_value=len(radio2))

st.write(radio2.head(nombredemessage))

st.bar_chart(data=radio2.head(nombredemessage)['longueur'])


st.scatter_chart(data=radio2.head(nombredemessage), x='nombredemots', y='longueur')





