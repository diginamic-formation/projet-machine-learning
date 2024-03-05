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
    df = pd.read_csv(file, sep=separator, decimal=decimal)
    return df


def data_overview(df):
    st.write("Aperçu du jeu de données :")
    st.dataframe(df)


def data_dimension(df):
    st.write("Dimension du jeu de données : ")
    st.write("Lignes = ", df.shape[0], "\nColonnes", df.shape[1])


def data_type(df):
    st.write("Types de données :")
    st.write(df.dtypes)


def descriptive_statistics():
    pass


def missing_data(df):
    msno.matrix(df)
    st.pyplot()

def missing_data_stats(df):
    st.write("Nombre de valeurs manquantes par variable :")
    st.write(df.isnull().sum())

def displaying_outlierrs():
    pass


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
    sns.pairplot(df)
    st.pyplot()


def preprocess():
    file = st.file_uploader("Choisissez un fichier", type=["csv"])
    separator = st.selectbox("Séparateur", options=[",", ";"])
    decimal = st.selectbox("Séparateur pour les chiffres décimaux", options=[".", ","])

    if file is not None:
        df = load_data(file, separator, decimal)
        data_overview(df)
        data_type(df)
        data_dimension(df)
        missing_data_stats(df)
        missing_data(df)
        #pair_plot_sns(df)

        return df
    return None
