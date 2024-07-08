import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


def regression_lineaire(df, X, y, train_size, FEATURES, TARGET):
    """
    Exécute une régression linéaire sur les données fournies et affiche les résultats dans Streamlit.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données d'origine.
        X (pd.DataFrame): Le DataFrame contenant les features pour la régression.
        y (pd.Series): La série contenant la variable cible.
        train_size (float): La proportion des données à utiliser pour l'entraînement (entre 0 et 1).
        FEATURES (list): La liste des noms des colonnes utilisées comme features.
        TARGET (str): Le nom de la colonne cible.

    Returns:
        model (LinearRegression): Le modèle de régression linéaire entraîné.
    
    Processus :
        1. Divise les données en ensembles d'entraînement et de test en utilisant `train_test_split`.
        2. Entraîne un modèle de régression linéaire sur l'ensemble d'entraînement.
        3. Fait des prédictions sur l'ensemble de test.
        4. Crée un DataFrame pour les résultats, y compris les valeurs réelles et prédites.
        5. Affiche les résultats sous forme de tableau et de graphique dans Streamlit.

    Affichage dans Streamlit :
        - Affiche les premiers résultats de la régression linéaire sous forme de tableau.
        - Affiche un graphique de dispersion comparant les valeurs réelles et prédites, avec une ligne de régression.
    """
    st.write("Démarrage de l'algorithme de regression")
    random_state = 42

    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    # Entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_test = pd.DataFrame(X_test, columns=FEATURES)
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
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Target réel")
    ax.set_ylabel("Target prédit")
    ax.set_title("Comparaison des Targets Réels et Prédits")
    # Tracer la droite de régression
    sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', ax=ax)
    st.pyplot(fig)

    return model



def regression_ridge(df: pd.DataFrame, X, y, train_size=0.2):
    """
    Exécute une régression Ridge sur les données fournies, évalue le modèle avec différents paramètres alpha,
    et affiche les résultats dans Streamlit.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données d'origine.
        X (pd.DataFrame): Le DataFrame contenant les features pour la régression.
        y (pd.Series): La série contenant la variable cible.
        train_size (float): La proportion des données à utiliser pour l'entraînement (entre 0 et 1).

    Returns:
        model (Ridge): Le modèle de régression Ridge entraîné avec le meilleur paramètre alpha.

    Processus :
        1. Sélectionne les valeurs de alpha et le pas de variation de alpha via des sliders Streamlit.
        2. Divise les données en ensembles d'entraînement et de test en utilisant `train_test_split`.
        3. Pour chaque valeur de alpha, entraîne un modèle de régression Ridge et évalue l'erreur quadratique moyenne (MSE) sur l'ensemble de test.
        4. Crée un DataFrame contenant les variables, les coefficients, les évaluations (MSE) et le paramètre alpha correspondant.
        5. Affiche les résultats sous forme de tableau dans Streamlit.
        6. Affiche l'évolution de l'erreur quadratique moyenne (MSE) par rapport au paramètre alpha sous forme de graphique.
        7. Identifie le meilleur paramètre alpha en fonction du MSE moyen.
        8. Affiche les coefficients des modèles en fonction des différentes valeurs de alpha sous forme de graphique.
        9. Entraîne et retourne un modèle de régression Ridge avec le meilleur paramètre alpha.

    Affichage dans Streamlit :
        - Affiche les résultats de la régression Ridge sous forme de tableau.
        - Affiche l'évolution de l'erreur quadratique moyenne (MSE) par rapport au paramètre alpha sous forme de graphique.
        - Affiche le meilleur paramètre alpha avec un message de succès.
        - Affiche les coefficients des modèles en fonction des différentes valeurs de alpha sous forme de graphique.
    """
    df_resultat_ridge = []
    n_alphas = st.slider("Alpha", min_value=0.10, max_value=10.0, step=0.1)
    step = st.slider("Le pas de variation de alpha", min_value=0.01, max_value=0.2, step=0.01)
    alphas = np.arange(0, n_alphas, step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    for alpha in alphas:
        # A chaque itération, nous instancions notre modèle avec un nouvel alpha
        # Puis nous entraînons le modèle et l'évaluons
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        X_test = pd.DataFrame(X_test)
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
    # trie pour le meilleur score MSE afin de recuperer l'alpha correspondant
    st.write("Meilleur score MSE :")
    alphas_used = df_resultat_ridge.groupby("alpha")['mse'].mean()
    st.write(alphas_used)

    alphas_used = alphas_used.reset_index()

    st.subheader("Évolution de l'erreur quadratique par rapport au paramètre alpha")
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(alphas_used['alpha'], alphas_used['mse'])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("MSE")
    ax.set_title("MSE en fonction de alpha")
    st.pyplot(fig)
    alphas_used = alphas_used.sort_values(by='mse')
    best_alpha = alphas_used['alpha'].iloc[0]
    st.success(f"Meilleur paramètre alpha pour le modèle : {best_alpha}")
    # affichage des coefficient
    st.line_chart(df_resultat_ridge.set_index('alpha')[['coefficient']])

    model = Ridge(alpha=best_alpha)
    return model



def regressionlasso(df: pd.DataFrame, X, y, train_size=0.2):
    """
    Exécute une régression Lasso sur les données fournies, évalue le modèle avec différents paramètres alpha,
    et affiche les résultats dans Streamlit.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données d'origine.
        X (pd.DataFrame): Le DataFrame contenant les features pour la régression.
        y (pd.Series): La série contenant la variable cible.
        train_size (float): La proportion des données à utiliser pour l'entraînement (entre 0 et 1).

    Returns:
        model (Lasso): Le modèle de régression Lasso entraîné avec le meilleur paramètre alpha.

    Processus :
        1. Sélectionne les valeurs de alpha et le pas de variation de alpha via des sliders Streamlit.
        2. Divise les données en ensembles d'entraînement et de test en utilisant `train_test_split`.
        3. Pour chaque valeur de alpha, entraîne un modèle de régression Lasso et évalue l'erreur quadratique moyenne (MSE) sur l'ensemble de test.
        4. Crée un DataFrame contenant les variables, les coefficients, les évaluations (MSE) et le paramètre alpha correspondant.
        5. Affiche les résultats sous forme de tableau dans Streamlit.
        6. Affiche l'évolution de l'erreur quadratique moyenne (MSE) par rapport au paramètre alpha sous forme de graphique.
        7. Identifie le meilleur paramètre alpha en fonction du MSE moyen.
        8. Entraîne et retourne un modèle de régression Lasso avec le meilleur paramètre alpha.

    Affichage dans Streamlit :
        - Affiche les résultats de la régression Lasso sous forme de tableau.
        - Affiche l'évolution de l'erreur quadratique moyenne (MSE) par rapport au paramètre alpha sous forme de graphique.
        - Affiche le meilleur paramètre alpha avec un message de succès.
    """
    df_resultat_lasso = []
    n_alphas = st.slider("Alpha", min_value=0.10, max_value=10.0, step=0.1)
    step = st.slider("Le pas de variation de alpha", min_value=0.01, max_value=0.2, step=0.01)
    alphas = np.arange(0, n_alphas, step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    
    for alpha in alphas:
        model = Lasso(alpha=alpha).fit(X_train, y_train)
        X_test = pd.DataFrame(X_test)
        y_pred = model.predict(X_test)
        mse = round(mean_squared_error(y_test.values, y_pred), 4)
        res = pd.DataFrame({"variable": X_test.columns, "coefficient": model.coef_})
        res['alpha'] = alpha
        res['mse'] = mse
        df_resultat_lasso.append(res)

    df_resultat_lasso = pd.concat(df_resultat_lasso, ignore_index=True)
    st.write("Résultats de la régression Lasso ")
    st.dataframe(df_resultat_lasso)
    alphas_used = df_resultat_lasso.groupby("alpha")['mse'].mean()
    alphas_used = alphas_used.reset_index()

    st.subheader("Évolution de l'erreur quadratique par rapport au paramètre alpha")
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(alphas_used['alpha'], alphas_used['mse'])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("MSE")
    ax.set_title("MSE en fonction de alpha")
    st.pyplot(fig)
    alphas_used = alphas_used.sort_values(by='mse')
    best_alpha = alphas_used['alpha'].iloc[0]
    st.success(f"Meilleur paramètre alpha pour le modèle : {best_alpha}")

    model = Lasso(alpha=best_alpha)
    return model

    # st.line_chart(df_resultat_lasso.set_index('alpha')['coefficient'])

def regression_elasticnet(df: pd.DataFrame, X, y, train_size=0.2):
    """
    Effectue une régression ElasticNet sur un ensemble de données.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        X: Les variables indépendantes.
        y: La variable dépendante.
        train_size (float, optional): La proportion d'entraînement. Par défaut, 0.2.

    Returns:
        model: Le modèle ElasticNet entraîné avec le meilleur paramètre alpha.
    """
    df_resultat_elasticnet = []
    n_alphas = st.slider("Alpha", min_value=0.10, max_value=10.0, step=0.1)
    l1_ratio = st.slider("L1 Ratio", min_value=0.0, max_value=1.0, step=0.1)
    step = st.slider("Le pas de variation de alpha", min_value=0.01, max_value=0.2, step=0.01)
    alphas = np.arange(0, n_alphas, step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    
    for alpha in alphas:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X_train, y_train)
        X_test = pd.DataFrame(X_test)
        y_pred = model.predict(X_test)
        mse = round(mean_squared_error(y_test.values, y_pred), 4)
        res = pd.DataFrame({"variable": X_test.columns, "coefficient": model.coef_})
        res['alpha'] = alpha
        res['mse'] = mse
        df_resultat_elasticnet.append(res)

    df_resultat_elasticnet = pd.concat(df_resultat_elasticnet, ignore_index=True)
    st.write("Résultats de la régression ElasticNet")
    st.dataframe(df_resultat_elasticnet)
    alphas_used = df_resultat_elasticnet.groupby("alpha")['mse'].mean()
    alphas_used = alphas_used.reset_index()

    st.subheader("Évolution de l'erreur quadratique par rapport au paramètre alpha")
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(alphas_used['alpha'], alphas_used['mse'])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("MSE")
    ax.set_title("MSE en fonction de alpha")
    st.pyplot(fig)
    alphas_used = alphas_used.sort_values(by='mse')
    best_alpha = alphas_used['alpha'].iloc[0]
    st.success(f"Meilleur paramètre alpha pour le modèle :  {best_alpha}")

    model = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio)
    return model

def regression_random_forest(df: pd.DataFrame, X, y, train_size=0.2):
    """
    Exécute une régression par Forêt Aléatoire sur les données fournies,
    évalue le modèle et affiche les résultats dans Streamlit.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données d'origine.
        X (pd.DataFrame): Le DataFrame contenant les features pour la régression.
        y (pd.Series): La série contenant la variable cible.
        train_size (float): La proportion des données à utiliser pour l'entraînement (entre 0 et 1).

    Returns:
        model (RandomForestRegressor): Le modèle de régression par Forêt Aléatoire entraîné.

    Processus :
        1. Sélectionne les hyperparamètres du modèle via des sliders Streamlit :
           - Nombre d'arbres (`n_estimators`)
           - Profondeur maximale des arbres (`max_depth`)
        2. Divise les données en ensembles d'entraînement et de test en utilisant `train_test_split`.
        3. Entraîne le modèle de régression par Forêt Aléatoire avec les hyperparamètres sélectionnés.
        4. Prédiction sur l'ensemble de test.
        5. Évalue l'erreur quadratique moyenne (MSE) sur l'ensemble de test.
        6. Affiche l'erreur quadratique moyenne (MSE) dans Streamlit.
        7. Affiche un graphique de comparaison entre les valeurs réelles et prédites de la cible.
        
    Affichage dans Streamlit :
        - Affiche l'erreur quadratique moyenne (MSE).
        - Affiche un graphique comparant les valeurs réelles et prédites de la cible.
    """
    
    # Sélection des hyperparamètres du modèle via des sliders Streamlit
    n_estimators = st.slider("Nombre d'arbres", min_value=10, max_value=1000, step=10)
    max_depth = st.slider("Profondeur maximale", min_value=1, max_value=50, step=1)
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Entraînement du modèle de régression par Forêt Aléatoire
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Évaluation de l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Erreur quadratique moyenne pour la régression par Forêt Aléatoire: {mse}")
    
    # Visualisation des résultats avec un graphique
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Target réel")
    ax.set_ylabel("Target prédit")
    ax.set_title("Comparaison des Targets Réels et Prédits")
    # Tracer la droite de régression
    sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', ax=ax)
    st.pyplot(fig)

    return model


def regression_gradient_boosting(df: pd.DataFrame, X, y, train_size=0.2):
    """
    Exécute une régression par Gradient Boosting sur les données fournies, 
    évalue le modèle et affiche les résultats dans Streamlit.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données d'origine.
        X (pd.DataFrame): Le DataFrame contenant les features pour la régression.
        y (pd.Series): La série contenant la variable cible.
        train_size (float): La proportion des données à utiliser pour l'entraînement (entre 0 et 1).

    Returns:
        model (GradientBoostingRegressor): Le modèle de régression par Gradient Boosting entraîné.

    Processus :
        1. Sélectionne les hyperparamètres du modèle via des sliders Streamlit :
           - Nombre d'arbres (`n_estimators`)
           - Taux d'apprentissage (`learning_rate`)
           - Profondeur maximale des arbres (`max_depth`)
        2. Divise les données en ensembles d'entraînement et de test en utilisant `train_test_split`.
        3. Entraîne le modèle de régression par Gradient Boosting avec les hyperparamètres sélectionnés.
        4. Prédiction sur l'ensemble de test.
        5. Évalue l'erreur quadratique moyenne (MSE) sur l'ensemble de test.
        6. Affiche l'erreur quadratique moyenne (MSE) dans Streamlit.
        7. Affiche un graphique de comparaison entre les valeurs réelles et prédites de la cible.
        
    Affichage dans Streamlit :
        - Affiche l'erreur quadratique moyenne (MSE).
        - Affiche un graphique comparant les valeurs réelles et prédites de la cible.
    """
    
    # Sélection des hyperparamètres du modèle via des sliders Streamlit
    n_estimators = st.slider("Nombre d'arbres", min_value=10, max_value=1000, step=10)
    learning_rate = st.slider("Taux d'apprentissage", min_value=0.01, max_value=1.0, step=0.01)
    max_depth = st.slider("Profondeur maximale", min_value=1, max_value=50, step=1)
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Entraînement du modèle de régression par Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Évaluation de l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Erreur quadratique moyenne pour la régression par Gradient Boosting: {mse}")
    
    # Visualisation des résultats avec un graphique
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Target réel")
    ax.set_ylabel("Target prédit")
    ax.set_title("Comparaison des Targets Réels et Prédits")
    # Tracer la droite de régression
    sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', ax=ax)
    st.pyplot(fig)

    return model
