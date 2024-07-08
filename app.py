
import machine_learning
import streamlit as st
import data_management
import preprocessing

# Configuration de la page Streamlit
st.set_page_config(page_title="Projet machine learning", layout="wide")

# Titre de la barre latérale
st.sidebar.title("Sommaire")

# Liste des pages disponibles
pages = ["Chargement des données", "Gestion des données", "Machine learning"]

# Radio button pour la navigation entre les pages
page = st.sidebar.radio("Aller vers la page :", pages)

# Initialisation de la variable data
data = None

# Si la page sélectionnée est "Chargement des données"
if page == pages[0]:
    # Appel à la fonction preprocess pour charger et prétraiter les données
    data = data_management.preprocess()
    # Sauvegarde des données prétraitées dans la session
    st.session_state["result"] = data

# Si la page sélectionnée est "Gestion des données"
elif page == pages[1]:
    # Vérification si les données ont été chargées
    if "result" in st.session_state:
        # Appel à la fonction run du module preprocessing pour gérer les données
        data = preprocessing.run(st.session_state["result"])
        # Sauvegarde des données traitées dans la session
        st.session_state["result"] = data

# Si la page sélectionnée est "Machine learning"
elif page == pages[2]:
    # Vérification si les données ont été chargées et prétraitées
    if "result" in st.session_state:
        # Appel à la fonction run du module machine_learning pour exécuter les modèles de machine learning
        machine_learning.run(st.session_state["result"])