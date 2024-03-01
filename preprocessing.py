import streamlit as st
import pandas as pd
from data_management import data_type


def change_variable_types(df):
    column_name = st.selectbox("Selectionner votre variable : ", options=df.columns.to_list())
    new_type = st.selectbox("Selectionner le nouveau type : ", ["float", "int", "object"])
    if st.button("Convertir"):
        df[column_name] = df[column_name].astype(new_type)
        data_type(df)


def delete_columns(df):
    columns_to_delete = df.columns.to_list()
    columns_name = st.multiselect("Selectionner la colonne à supprimer : ", columns_to_delete)
    if st.button("Supprimer"):
        for column_name in columns_name:
            df.drop(column_name, axis=1, inplace=True)
            columns_to_delete.remove(column_name)
        st.success('Colonnes supprimées ')
        st.dataframe(df)
    return df


def missed_data_treatment(df):
    st.write("Traitement des valeurs manquantes")
    # Effacefr les lignes
    
    # Remplacer par une valeur choisie par le client
    # Remplacer par une valeur calculée (moyenne, mediane, valeurs voisines)
    pass


def run(df):
    if df is not None:
        # Changer les types de variables
        change_variable_types(df)
        # Supprimer les colonnes qu'il veut
        df = delete_columns(df)
        # Traitement des valeurs manquantes
        df = missed_data_treatment(df)
        return df