import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def make_algo_choice(df):
    st.selectbox("Selectionner la colonne cible")

    return df


def regression_lineaire(df: pd.DataFrame):
    FEATURES = [x for x in df.columns if x != 'target']
    TARGET = 'target'
    TRAIN_SIZE = 0.3
    RANDOM_STATE = 42

    # Séparation des features et de la target
    X = df[FEATURES]
    y = df[TARGET]

    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)

    # Entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Création du DataFrame pour les résultats
    results_df = pd.DataFrame({
        'Age': X_test['age'],
        'Sex': X_test['sex'],
        'BMI': X_test['bmi'],
        'BP': X_test['bp'],
        'S1': X_test['s1'],
        'S2': X_test['s2'],
        'S3': X_test['s3'],
        'S4': X_test['s4'],
        'S5': X_test['s5'],
        'S6': X_test['s6'],
        'Prix réel': y_test,
        'Prix prédit': y_pred
    })

    # Affichage des résultats dans un tableau dans Streamlit
    st.write("Résultats de la régression linéaire :")
    st.table(results_df.head(10))
    # Visualisation des résultats avec un graphique
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Prix réel")
    ax.set_ylabel("Prix prédit")
    ax.set_title("Comparaison des prix réels et prédits")
    # Tracer la droite de régression
    sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', ax=ax)

    st.pyplot(fig)


def regression_ridge(df: pd.DataFrame):
    FEATURES = [x for x in df.columns if x != 'target']
    TARGET = 'target'

    df_resultat_ridge = []
    n_alphas = 1000
    alphas = np.arange(0, n_alphas, 0.5)
    for alpha in alphas:
        # A chaque itération, nous instancions notre modèle avec un nouvel alpha
        # Puis nous entraînons le modèle et l'évaluons
        clf = Ridge(alpha=alpha).fit(df[FEATURES], df[TARGET])
        y_hat = clf.predict(df[FEATURES])
        mse = round(mean_squared_error(df[TARGET].values, y_hat), 4)

        # Nous créons un DataFrame avec les variables, les coefficients, et ajoutons les évaluations (MSE) ainsi que l'alpha correspondant
        res = pd.DataFrame({"variable": FEATURES, "coefficient": clf.coef_})
        res['alpha'] = alpha
        res['mse'] = mse

        df_resultat_ridge.append(res)

    # À la fin, nous concaténons tous nos résultats
    df_resultat_ridge = pd.concat(df_resultat_ridge)
    st.write("Résultats de la régression Ridge :")
    st.write(df_resultat_ridge)
    # trie pour le meilleur scoreMSE afin de recuperer l'alpha correspondant
    st.write("Meilleur score MSE :")
    st.write(df_resultat_ridge.groupby("alpha")['mse'].mean().sort_values())
    # affichage des coefficient
    st.line_chart(df_resultat_ridge.set_index('alpha')[['coefficient']])


def regression_lasso(df: pd.DataFrame):
    FEATURES = [x for x in df.columns if x != 'target']
    TARGET = 'target'
    df_resultat_lasso = []
    n_alphas = 5_00
    alphas = np.arange(0, n_alphas, 0.1)
    for alpha in alphas:
        clf = Lasso(alpha=alpha).fit(df[FEATURES], df[TARGET])
        y_hat = clf.predict(df[FEATURES])
        mse = round(mean_squared_error(df[TARGET].values, y_hat), 4)
        res = pd.DataFrame({"variable": FEATURES, "coefficient": clf.coef_})
        res['alpha'] = alpha
        res['mse'] = mse
    df_resultat_lasso.append(res)

    df_resultat_lasso = pd.concat(df_resultat_lasso)
    st.write("Résultats de la régression Lasso :")
    st.write(df_resultat_lasso)

    st.write(df_resultat_lasso.groupby("alpha")['mse'].mean().sort_values())

    VAR = "INDUS"
    plt.plot(df_resultat_lasso[df_resultat_lasso.variable == VAR].alpha,
             df_resultat_lasso[df_resultat_lasso.variable == VAR].coefficient)
    plt.title(f"Variable {VAR}")
    st.pyplot()

    # st.line_chart(df_resultat_lasso.set_index('alpha')['coefficient'])


def is_standardized(X):
    return X.apply(lambda x: x.mean()).abs().lt(0.01).all() and X.apply(lambda x: x.std()).subtract(1).abs().lt(
        0.01).all()


def data_standardisation(X):
    if X is not None and is_standardized(X):
        st.write("Votre jeu de données est standarisé")
    else:
        if st.checkbox("Standariser vos données"):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
    return X


def plot_new_data(df):
    pass


def target_encoding(y):
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        st.dataframe(y)
    return y

def feature_encoding(X):
    if X is not None and len(X) > 0 :
        for column in X.columns:

            if not pd.api.types.is_numeric_dtype(X[column]):
                label_encoder = LabelEncoder()
                X[column] = label_encoder.fit_transform(X[column])
        st.dataframe(X)


def select_features(df):
    features = st.multiselect("Choisir vos variables explicatives (FEATURES)", df.columns.to_list())
    return features


def select_target(df: pd.DataFrame):
    target = st.selectbox("Choisir votre variable à expliquer (TARGET)", df.columns.to_list())
    return target


def run(df):
    # choisir la target
    TARGET = select_target(df)
    y = df[TARGET]
    if y is not None :
        st.write("la colonne cible")
        st.dataframe(y)
    # Choisir les Features
    FEATURES = select_features(df)
    X = df[FEATURES]

    # Encodage (changer les valeurs catégorielle en indice numérique
    y = target_encoding(y)

    X = feature_encoding(X)

    # Standardisation
    X = data_standardisation(X)

    # Graphique d'aide à la décision (corrélation et nuages ...etc )
    plot_new_data(df)

    # Choix des Feartures et Target
    # model = make_algo_choice(df)

    # Choix de l'algo

    # make_algo_choice(df)
    # regression_lineaire(df)
    # regression_ridge(df)
    # regression_lasso(df)
    # Choix des paramètres pour l'apprentissge (Test Size)

    # Entrainement du modèle

    # Prediction

    # Evaluation et validation

    return df
