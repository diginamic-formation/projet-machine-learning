import machine_learning_classification
import machine_learning
import streamlit as st
import data_management
import preprocessing

st.set_page_config(page_title="Projet machine learning",
                   layout="wide")

st.sidebar.title("Sommaire")
pages = ["Chargement des données", "Gestion des données", "Machine learning - Reg", "Machine learning - Clas"]

page = st.sidebar.radio("Aller vers la page :", pages)

data = None
if page == pages[0]:
    data = data_management.preprocess()
    st.session_state["result"]= data
elif page == pages[1]:
    if "result" in st.session_state:
        preprocessing.run(st.session_state["result"])

elif page == pages[2]:
    if "result" in st.session_state:
        machine_learning.run(st.session_state["result"])

elif page == pages[3]:
    if "result" in st.session_state:
        machine_learning_classification.run(st.session_state["result"])
