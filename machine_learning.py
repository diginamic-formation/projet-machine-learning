import streamlit as st
import  pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def make_algo_choice(df):
    st.selectbox("Selectionner la colonne cible")



    return df

# def regression_lineaire(df: pd.DataFrame):
#     FEATURES = [x for x in df.columns if x!='target']
#     TARGET = 'target'
#     TRAIN_SIZE = 0.3
#     RANDOM_STATE = 42
#     X = df[FEATURES]
#     y= df[TARGET]
#
#     model = LinearRegression()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
#     print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#     # Entraînement du modèle sur les données d'entraînement
#     model.fit(X_train, y_train)
#     # Prédiction sur l'ensemble de test
#     y_pred = model.predict(X_test)
#     # Affichage des résultats
#
#     # Affichage des résultats dans Streamlit
#     st.write("Actual prices vs Predicted prices:")
#     st.write("House Size - Actual Price - Predicted Price")
#     for i in range(len(X_test)):
#         st.write(f"{X_test[i][0]} - {y_test[i][0]} - {y_pred[i][0]}")



def run(df):
    # Encodage

    # Standardisation

    # Graphique d'aide à la décision (corrélation et nuages ...etc )

    # Choix des Feartures et Target

    # Choix de l'algo

    #make_algo_choice(df)
    regression_lineaire(df)
    # Choix des paramètres pour l'apprentissge (Test Size)

    # Entrainement du modèle

    # Prediction

    # Evaluation et validation

    return df
