import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# Algorithmes de regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
# Algorithmes de classification
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ML_ALGO_TYPES = ["Regression", "Classification"]
REGRESSION_MODELS = ["Regression lineaire", "Lasso", "Ridge"]
CLASSIFICATION_MODELS = ["Logistic regression", "Tree decision", "KNeighbors Classifier"]


def select_algo_type():
    ml_algo_type = st.selectbox("Choisir le type de l'algo d'apprentissage", options=ML_ALGO_TYPES)
    return ml_algo_type


def select_algo_model(algo_type):
    models = []
    if algo_type == ML_ALGO_TYPES[0]:
        models = REGRESSION_MODELS
    elif algo_type == ML_ALGO_TYPES[1]:
        models = CLASSIFICATION_MODELS
    model = st.selectbox("Selectionner le modèle d'apprentissage", options=models)
    return model


def logistic_regression(X, y, valeur=0.2):
    clf = LogisticRegression()

    # acquisition des nom des colonnes
    # creation d'un dataframe et ajout de la colonne target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=valeur, random_state=42)

    clf_data = clf.fit(X_train, y_train)
    # predict en classification permet de prédire la classe de sortie

    y_pred = clf.predict(X_test)
    # predict_proba en classification permet de prédire les probabilités appartenant à chaque classe
    y_prob = clf.predict_proba(X_test)
    # Evaluation
    st.write('cm : matrice de confusion')
    cm_pred = confusion_matrix(y_test, y_pred)
    cm_pred
    st.write('cr : rapport de classification')
    cr_pred = classification_report(y_test, y_pred, output_dict=True)
    st.write("cr_pred")
    st.table(cr_pred)


def regression_lineaire(X, y, train_size):
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


def regressionlasso(df: pd.DataFrame):
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
    st.write("Résultats de la régression Lasso ")
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


def target_encoding(y):
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        st.dataframe(y)
    return y


def feature_encoding(X):
    if X is not None and len(X) > 0:
        for column in X.columns:

            if not pd.api.types.is_numeric_dtype(X[column]):
                label_encoder = LabelEncoder()
                X[column] = label_encoder.fit_transform(X[column])
        st.dataframe(X)
    return X


def select_features(df):
    features = st.multiselect("Choisir vos variables explicatives (FEATURES)", df.columns.to_list())
    return features


def select_target(df: pd.DataFrame):
    target = st.selectbox("Choisir votre variable à expliquer (TARGET)", df.columns.to_list())
    return target


def run_machine_learning(df, ml_algo_type, ml_algo_model, X, y):
    if ml_algo_type == ML_ALGO_TYPES[0]:
        if ml_algo_model == REGRESSION_MODELS[0]:
            train_size = st.slider("Taille du jeux de test", min_value=0.0, max_value=1.0, step=0.05,
                                   value=st.session_state.train_size_value)
            regression_lineaire(X, y, train_size)
        if ml_algo_model == REGRESSION_MODELS[1]:
            regressionlasso(df)
        if ml_algo_model == REGRESSION_MODELS[2]:
            regression_ridge(df)

    elif ml_algo_type == ML_ALGO_TYPES[1]:
        if ml_algo_model == CLASSIFICATION_MODELS[0]:
            valeur = st.number_input("Saisissez la valeur test_size :", step=0.01, value=0.02)
            valeur = 0.2
            logistic_regression(X, y, valeur)
        if ml_algo_model == CLASSIFICATION_MODELS[1]:
            pass
        if ml_algo_model == CLASSIFICATION_MODELS[2]:
            pass


def init_classification():
    X_validation = 5
    structure = {
        'Regression_Logistique': {
            'model': LogisticRegression(),
            'hyperparameters': {}

        },
        'Arbre_Decision': {
            'model': DecisionTreeClassifier(),
            'hyperparameters': {}
        },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'hyperparameters': {}
        },
        'KNeighbours_Classifier': {
            'model': KNeighborsClassifier(),
            'hyperparameters': {}
        },
        'SVC': {
            'model': SVC(),
            'hyperparameters': {}
        },
    }

    return structure, X_validation


def classification(X, y):
    structure, X_validation = init_classification()
    result = {}
    print('structure : ',structure)
    for iteration in structure:
        for tour in range(0, X_validation):
            info = str(iteration), f'Split numéro {tour + 1}'
            st.write(info)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42
            )
            model = structure[iteration]['model']
            parameters = structure[iteration]['hyperparameters']
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            result[info] = report_dict
            st.write(report_dict['macro avg']['f1-score'], )


def run(df):
    # choisir la target
    TARGET = select_target(df)
    y = df[TARGET]
    if y is not None:
        st.write("la colonne cible")
        st.dataframe(y)
    # Choisir les Features
    FEATURES = select_features(df)
    X = df[FEATURES]

    # Encodage (changer les valeurs catégorielle en indice numérique
    y = target_encoding(y)

    X = feature_encoding(X)

    # Standardisation
    # X = data_standardisation(X)

    # # Choix du type
    # ml_algo_type = select_algo_type()
    #
    # # Choix du modèle
    # ml_algo_model = select_algo_model(ml_algo_type)
    #
    # st.dataframe(X)
    # if st.button("Démarrer le process"):
    #     run_machine_learning(df, ml_algo_type, ml_algo_model, X, y)
    if st.button('start classification') :
        classification(X, y)

    # make_algo_choice(df)
    # regression_lineaire(df)
    # regression_ridge(df)
    # regression_lasso(df)
    # Choix des paramètres pour l'apprentissge (Test Size)

    # Entrainement du modèle

    # Prediction

    # Evaluation et validation

    return df
