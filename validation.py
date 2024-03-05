import pandas as pd
import streamlit as st
import numpy as np

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold, StratifiedKFold


def validation_k_fold(X, y, FEATURES, model):
    kf = KFold(n_splits=6, shuffle=True, random_state=2021)
    columns = ["R2", "MSE"]
    stats_df = pd.DataFrame(columns=columns)
    lignes = 2
    colonnes = 3
    fig, axes = plt.subplots(lignes, colonnes, figsize=(15, 10))
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        ligne = int(i / colonnes)
        colonne = i % colonnes
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train[FEATURES], y_train)
        y_pred = model.predict(X_test[FEATURES])
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred, squared=True)
        stats_df.loc[i] = [round(r2, 4), round(mse, 4)]

        # Créer le graphique
        axes[ligne,colonne].set_title(f'Itération {i+1} - r2: {round(r2, 4)} - mse: {round(mse,4)} ',loc='center')
        axes[ligne,colonne].scatter(y_test, y_pred)
        axes[ligne,colonne].plot(np.arange(y_test.min(), y_test.max()),
                np.arange(y_test.min(), y_test.max()),
                color='red')  # Ajoute une ligne de référence pour la comparaison

    st.header("Cross validation")
    st.subheader("Comportement du modèle")
    st.pyplot(fig)  # Afficher le graphique dans Streamlit
    st.subheader("Tableau des metrics de validation du modèle")
    stats_mean_df = stats_df.mean()
    stats_mean_df = stats_mean_df.to_frame().reset_index()

    # Renommer les colonnes
    stats_mean_df = stats_mean_df.rename(columns={'index': 'metric', 0: 'moyenne'})
    st.table(stats_mean_df)


