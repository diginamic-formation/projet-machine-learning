import streamlit as st
import pandas as pd
import numpy as np

# Algorithmes de regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


def regression_lineaire(df, X, y, train_size, FEATURES, TARGET):
    st.write("Démarrage de l'algorithme de regression")
    random_state = 42

    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    # Entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)
    results_df = pd.DataFrame()
    # Création du DataFrame pour les résultats
    for column in X_test.columns:
        results_df[column] = X_test[column]
    results_df['target reel'] = y_test
    results_df['target predit'] = y_pred

    # Affichage des résultats dans un tableau dans Streamlit
    st.write("Résultats de la régression linéaire :")
    st.table(results_df.head(10))
    # Visualisation des résultats avec un graphique
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Target réel")
    ax.set_ylabel("Target prédit")
    ax.set_title("Comparaison des Targets Réels et Prédits")
    # Tracer la droite de régression
    sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', ax=ax)
    st.pyplot(fig)

    return model


def regression_ridge(df: pd.DataFrame, X, y, train_size=0.2):

    df_resultat_ridge = []
    n_alphas = st.slider("Alpha",min_value=0.0, max_value=10.0, step=0.1)
    step = st.slider("Le pas de variation de alpha", min_value=0.01, max_value=0.2, step=0.01)
    alphas = np.arange(0, n_alphas, step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    for alpha in alphas:
        # A chaque itération, nous instancions notre modèle avec un nouvel alpha
        # Puis nous entraînons le modèle et l'évaluons
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = round(mean_squared_error(y_test.values, y_pred), 4)

        # Nous créons un DataFrame avec les variables, les coefficients, et ajoutons les évaluations (MSE) ainsi que l'alpha correspondant
        res = pd.DataFrame({"variable": X_test.columns, "coefficient": model.coef_})
        res['alpha'] = alpha
        res['mse'] = mse

        df_resultat_ridge.append(res)

    # À la fin, nous concaténons tous nos résultats
    df_resultat_ridge = pd.concat(df_resultat_ridge)
    st.write("Résultats de la régression Ridge :")
    st.write(df_resultat_ridge)
    # trie pour le meilleur scoreMSE afin de recuperer l'alpha correspondant
    st.write("Meilleur score MSE :")
    alphas_used = df_resultat_ridge.groupby("alpha")['mse'].mean()
    st.write(alphas_used)

    alphas_used = alphas_used.reset_index()

    st.subheader("Evolution de l'erreur quadratique par rapport au paramètre alpha")
    plt.plot(alphas_used['alpha'], alphas_used['mse'])
    plt.xlabel('Alpha')
    plt.ylabel('MSE')
    plt.title('MSE en fonction de alpha')
    st.pyplot()
    alphas_used = alphas_used.sort_values(by='mse')
    best_alpha = alphas_used['alpha'].iloc[0]
    st.success(f"Meilleur paramètre alpha pour le modèle :  {best_alpha}")
    # affichage des coefficient
    st.line_chart(df_resultat_ridge.set_index('alpha')[['coefficient']])

    model = Ridge(alpha=best_alpha)
    return model


def regressionlasso(df: pd.DataFrame, X, y,train_size=0.2):
    df_resultat_lasso = []
    n_alphas = st.slider("Alpha",min_value=0.0, max_value=10.0, step=0.1)
    step = st.slider("Le pas de variation de alpha", min_value=0.01, max_value=0.2, step=0.01)
    alphas = np.arange(0, n_alphas, step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    for alpha in alphas:
        model = Lasso(alpha=alpha).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = round(mean_squared_error(y_test.values, y_pred), 4)
        res = pd.DataFrame({"variable": X_test.columns, "coefficient": model.coef_})
        res['alpha'] = alpha
        res['mse'] = mse
        df_resultat_lasso.append(res)

    df_resultat_lasso = pd.concat(df_resultat_lasso)
    st.write("Résultats de la régression Lasso ")
    st.dataframe(df_resultat_lasso)
    alphas_used = df_resultat_lasso.groupby("alpha")['mse'].mean()
    alphas_used = alphas_used.reset_index()

    st.subheader("Evolution de l'erreur quadratique par rapport au paramètre alpha")
    plt.plot(alphas_used['alpha'], alphas_used['mse'])
    plt.xlabel('Alpha')
    plt.ylabel('MSE')
    plt.title('MSE en fonction de alpha')
    st.pyplot()
    alphas_used = alphas_used.sort_values(by='mse')
    best_alpha = alphas_used['alpha'].iloc[0]
    st.success(f"Meilleur paramètre alpha pour le modèle :  {best_alpha}")

    model = Lasso(alpha=best_alpha)
    return model
    # st.line_chart(df_resultat_lasso.set_index('alpha')['coefficient'])
