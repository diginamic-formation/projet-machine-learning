import streamlit as st
import pandas as pd
from data_management import data_type, missing_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)


def change_variable_types(df):
    column_name = st.selectbox("Selectionner votre variable : ", options=df.columns.to_list())
    new_type = st.selectbox("Selectionner le nouveau type : ", ["float", "int", "object"])
    if st.button("Convertir"):
        df[column_name] = df[column_name].astype(new_type)
        data_type(df)
    return df


def delete_columns(df):
    columns_to_delete = df.columns.to_list()
    columns_name = st.multiselect("Selectionner la colonne à supprimer : ", columns_to_delete)
    if st.button("Supprimer"):
        for column_name in columns_name:
            df.drop(column_name, axis=1, inplace=True)
        st.success('Colonnes supprimées ')
        st.dataframe(df)
    return df


def columns_with_missing_data(df: pd.DataFrame):
    columns_na = df.columns[df.isnull().any()]
    return columns_na


def get_list_strategy(df, column_to_treat):
    if df[column_to_treat].dtypes == 'float64' or df[column_to_treat].dtypes == 'int64':
        return ["effacer", "moyenne", "médiane", "valeur personnalisée"]
    else:
        return ["effacer", "valeur personnalisée"]


def missed_data_treatment(df):
    st.write("Traitement des valeurs manquantes")
    # Effacefr les lignes
    columns_na = columns_with_missing_data(df)
    if columns_na is not None and len(columns_na) > 0:
        column_to_treat = st.selectbox("Selectionner la colonner à traiter", columns_na)
        list_strategy = get_list_strategy(df, column_to_treat)
        fill_strategy = st.selectbox("Selectionner la stratégie de remplacement des valeurs manquantes",
                                     list_strategy)
        if fill_strategy == "valeur personnalisée":
            custom_value = st.text_input("Enter custom value")
            if st.button("Fill Missing Values with Custom Value"):
                df[column_to_treat] = df[column_to_treat].fillna(custom_value)
                st.success("Remplacement terminé")
        elif fill_strategy == "effacer":
            if st.button("Effacer"):
                df.dropna(subset=column_to_treat, inplace=True)
        else:
            if st.button(f"Fill Missing Values with {fill_strategy}"):
                df[column_to_treat] = df[column_to_treat].fillna(
                    df[column_to_treat].mean() if fill_strategy == "mean" else df[column_to_treat].median())
        # Remplacer par une valeur choisie par le client
        # Remplacer par une valeur calculée (moyenne, mediane, valeurs voisines)
        st.dataframe(df)
        missing_data(df)
    else:
        st.write("Votre jeux de données est complet")
    return df


def matrice_correlation(numerical_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = sns.diverging_palette(15, 160, n=11, s=100)
    mask = np.triu(numerical_df.corr())
    sns.heatmap(
        numerical_df.corr(),
        mask=mask,
        annot=True,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )
    st.pyplot()


def pair_plot(numerical_df):
    sns.pairplot(numerical_df)
    st.pyplot()


def plot_data(df):
    numerical_df = df.select_dtypes(include=['number'])
    columns_corr = st.multiselect("Selectinner les colonnes à visualiser", numerical_df.columns)
    if st.button("Afficher"):
        if len(columns_corr) > 0:
            numerical_df = numerical_df[columns_corr]
            # Matrice de Correlation
            matrice_correlation(numerical_df)
            # Pair plot
            pair_plot(numerical_df)

def run(df):
    if df is not None:
        # Changer les types de variables
        df = change_variable_types(df)
        # Supprimer les colonnes qu'il veut
        df = delete_columns(df)
        # Traitement des valeurs manquantes
        df = missed_data_treatment(df)
        # Affichage des graphiques
        plot_data(df)
        return df
