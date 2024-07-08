import streamlit as st
import pandas as pd
from data_management import data_type, missing_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)


def change_variable_types(df):
    """
    Convertit le type de données d'une variable dans un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
        pd.DataFrame: Le DataFrame mis à jour avec le nouveau type de données.
    """
    column_name = st.selectbox("Sélectionner votre variable : ", options=df.columns.to_list())
    new_type = st.selectbox("Sélectionner le nouveau type : ", ["float", "int", "object"])
    if st.button("Convertir"):
        df[column_name] = df[column_name].astype(new_type)
        data_type(df)
    return df


def delete_columns(df):
    """
    Supprime les colonnes sélectionnées d'un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
        pd.DataFrame: Le DataFrame mis à jour après la suppression des colonnes.
    """
    columns_to_delete = df.columns.to_list()
    columns_name = st.multiselect("Sélectionner la colonne à supprimer : ", columns_to_delete)
    if st.button("Supprimer"):
        for column_name in columns_name:
            df.drop(column_name, axis=1, inplace=True)
        st.success('Colonnes supprimées ')
        st.dataframe(df)
    return df


def columns_with_missing_data(df: pd.DataFrame):
    """
    Retourne les noms des colonnes contenant des valeurs manquantes dans un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
        pd.Index: L'index des colonnes avec des valeurs manquantes.
    """
    columns_na = df.columns[df.isnull().any()]
    return columns_na


def get_list_strategy(df, column_to_treat):
    """
    Retourne une liste de stratégies de traitement pour une colonne donnée dans un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        column_to_treat (str): Le nom de la colonne à traiter.

    Returns:
        list: Une liste de stratégies (par exemple, ["effacer", "moyenne", "médiane", "valeur personnalisée"]).
    """
    if df[column_to_treat].dtypes == 'float64' or df[column_to_treat].dtypes == 'int64':
        return ["effacer", "moyenne", "médiane", "valeur personnalisée"]
    else:
        return ["effacer", "valeur personnalisée"]


def missed_data_treatment(df):
    """
    Traite les valeurs manquantes dans un DataFrame.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.

    Returns:
        pandas.DataFrame: Le DataFrame avec les valeurs manquantes traitées.
    """
    columns_na = columns_with_missing_data(df)
    if columns_na is not None and len(columns_na) > 0:
        # Obtient les colonnes avec des valeurs manquantes
        column_to_treat = st.selectbox("Sélectionner la colonner à traiter", columns_na) 
        # Obtient les stratégies de traitement
        list_strategy = get_list_strategy(df, column_to_treat) 
        fill_strategy = st.selectbox("Sélectionner la stratégie de remplacement des valeurs manquantes",
                                     list_strategy)
        if fill_strategy == "valeur personnalisée":
            custom_value = st.text_input("Enter custom value")
            if st.button("Fill Missing Values with Custom Value"):
                df[column_to_treat] = df[column_to_treat].fillna(custom_value)
                st.success("Remplacement terminé")
        elif fill_strategy == "effacer":
            if st.button("Effacer"):
                df.dropna(subset=column_to_treat, inplace=True)
                st.success("Remplacement terminé")
        else:
            if st.button(f"Fill Missing Values with {fill_strategy}"):
                # Remplace les valeurs manquantes avec la moyenne ou la médiane
                df[column_to_treat] = df[column_to_treat].fillna(
                    df[column_to_treat].mean() if fill_strategy == "mean" else df[column_to_treat].median())
                st.success("Remplacement terminé")
    else:
        st.write("Votre jeux de données est complet")
    st.subheader("Visualisation du jeux de données nettoyé")
    st.dataframe(df)
    # Affiche les statistiques sur les valeurs manquantes
    missing_data(df) 
    return df

          
def matrice_correlation(numerical_df):
    correlation_matrix = numerical_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot()


def pair_plot(numerical_df):
    """
    Génère une paire de tracés (scatter plots) pour les variables numériques dans un DataFrame.
    """
    sns.pairplot(numerical_df)
    st.pyplot()


def plot_data(df):
    numerical_df = df.select_dtypes(include=['number'])
    columns_corr = st.multiselect("Selectionner les colonnes à visualiser", numerical_df.columns, key='corr_columns')
    if st.button("Afficher les corrélations", key='display_corr_button'):
        if len(columns_corr) > 0:
            numerical_df = numerical_df[columns_corr]
            with st.container():
                # Matrice de Correlation
                matrice_correlation(numerical_df)
                # Pair plot
                pair_plot(numerical_df)
        else:
            st.warning("Veuillez sélectionner au moins une colonne.")        

def run(df):
    """
    Exécute une série de traitements sur un DataFrame.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.

    Returns:
        pandas.DataFrame: Le DataFrame après les traitements.
    """
    if df is not None:
        st.subheader("Changement des types de variable")
        # Appelle la fonction pour changer les types de variable
        df = change_variable_types(df)
        st.subheader("Traitement des colonnes vides")
        # Appelle la fonction pour supprimer les colonnes vides
        df = delete_columns(df)
        st.subheader("Traitement des valeurs manquantes")
        # Appelle la fonction pour traiter les valeurs manquantes
        df = missed_data_treatment(df)
        st.subheader("Matrice de corrélation")
        # Affiche la matrice de corrélation
        plot_data(df.head())
        st.subheader("Taille du jeux de données")
         # Affiche la taille du DataFrame
        st.write(df.shape)
        return df