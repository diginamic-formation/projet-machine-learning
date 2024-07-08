import matplotlib
import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

import plotly.express as px
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)


def load_data(file, separator, decimal='.'):
    """
    Charge un fichier CSV dans un DataFrame.

    Args:
        file (str): Le chemin vers le fichier CSV.
        separator (str): Le séparateur utilisé dans le fichier CSV (par exemple, ',' ou ';').
        decimal (str, optional): Le séparateur décimal. Par défaut, '.'.

    Returns:
        pd.DataFrame: Le DataFrame contenant les données du fichier CSV.
    """
    df = pd.read_csv(file, sep=separator, decimal=decimal)
    return df


def data_overview(df):
    #Affiche un aperçu du jeu de données dans un DataFrame.
    st.write("Aperçu du jeu de données :")
    st.dataframe(df)


def data_dimension(df):
    #Affiche la dimension du jeu de données (nombre de lignes et de colonnes).
    st.write("Dimension du jeu de données : ")
    st.write("Lignes = ", df.shape[0], "\nColonnes", df.shape[1])


def data_type(df):
    # Affiche les types de données
    st.write("Types de données :")
    st.write(df.dtypes)


def descriptive_statistics(df):
    # Affiche les statistiques descriptives
    st.write("Statistiques descriptives :")
    st.write(df.describe())


def missing_data(df):
    # Affiche la matrice des valeurs manquantes
    msno.matrix(df)
    st.pyplot()

def missing_data_stats(df):
    #Affiche le nombre de valeurs manquantes par variable dans un DataFrame.
    st.write("Nombre de valeurs manquantes par variable :")
    st.write(df.isnull().sum())

def displaying_outliers(df):
    #Affiche la distribution des caractéristiques sélectionnées dans un DataFrame.
    selected_columns = st.multiselect('Sélectionnez les caractéristiques à afficher', df.columns.tolist(), key='feature_selection') 
    if st.button("Afficher", key='display_outliers_button'):
        if len(selected_columns) > 0:
            with st.container():
                for column in selected_columns:
                    plt.figure(figsize=(8, 4))
                    sns.histplot(df[column], kde=True)
                    plt.title(f'Distribution de {column}')
                    plt.xlabel(column)
                    plt.ylabel('Fréquence')
                    st.pyplot(plt.gcf())
        else:
            st.warning("Veuillez sélectionner au moins une caractéristique.")  

def target_analyse():
    pass


def correlation_matrix(df: pd.DataFrame):
    correlation_matrix = df.corr()
    # Afficher la matrice de corrélation sous forme de heatmap avec Streamlit
    st.write("Matrice de corrélation :")
    st.write(correlation_matrix)
    # Afficher la matrice de corrélation sous forme de heatmap avec Seaborn et Matplotlib
    st.write("Heatmap de la matrice de corrélation :")
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()


def pair_plot_sns(df):
    # Affiche une paire de tracés pour les relations entre les colonnes
    sns.pairplot(df)
    st.pyplot()


def preprocess():
    """
    Prétraite un fichier CSV en chargeant les données, affichant un aperçu, les types de données,
    la dimension, les statistiques descriptives et la gestion des données manquantes.

    Returns:
        pd.DataFrame or None: Le DataFrame contenant les données ou None si aucun fichier n'est chargé.
    """
    file = st.file_uploader("Choisissez un fichier", type=["csv"])
    separator = st.selectbox("Séparateur", options=[",", ";"])
    decimal = st.selectbox("Séparateur pour les chiffres décimaux", options=[".", ","])

    if file is not None:
        df = load_data(file, separator, decimal)
        data_overview(df)
        data_type(df)
        data_dimension(df)
        descriptive_statistics(df)
        missing_data_stats(df)
        missing_data(df)
        st.subheader("Distribution des caractéristiques")
        displaying_outliers(df)
        #pair_plot_sns(df)

        return df
    return None