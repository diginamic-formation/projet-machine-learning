import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from regression import regression_lineaire, regression_ridge, regressionlasso, regression_elasticnet, regression_random_forest, regression_gradient_boosting
from classification import tree_classifier, k_neighbors_classifier, logistic_regression, support_vector_classifier, naive_bayes_classifier, xgboost_classifier, sgd_classifier
from validation import validation_k_fold_classification,validation_k_fold_regression,compare_regression_models, compare_classification_models

ML_ALGO_TYPES = ["Regression", "Classification"]
REGRESSION_MODELS = ["Lineaire Regression", "Lasso", "Ridge", "ElasticNet", "Random Forest", "Gradient Boosting"]
CLASSIFICATION_MODELS = ["Logistic Regression", "Tree Classifier", "KNeighbors Classifier", "Support Vector Classifier", "Naive Bayes Classifier", "XGBoost Classifier", "Stochastic Gradient Descent Classifier"]

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
        label_mapping_df = pd.DataFrame(
            {'Category': label_encoder.classes_, 'Encoded': label_encoder.transform(label_encoder.classes_)})
        st.subheader("Encodage de la Target")
        st.write(pd.DataFrame(label_mapping_df))
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

def run_machine_learning(df, ml_algo_type, ml_algo_model, X, y, FEATURES, TARGET):
    """
    Exécute l'apprentissage automatique en fonction des paramètres fournis.

    Args:
        df (pandas.DataFrame): Le dataframe contenant les données.
        ml_algo_type (str): Le type d'algorithme (régression ou classification).
        ml_algo_model (str): Le modèle d'algorithme spécifique.
        X (pandas.DataFrame): Les caractéristiques d'entraînement.
        y (pandas.Series): La variable cible.
        FEATURES (list): Liste des noms de colonnes des caractéristiques.
        TARGET (str): Nom de la colonne de la variable cible.

    Returns:
        None
    """
    model = None
    if ml_algo_type == ML_ALGO_TYPES[0]:
        train_size = st.slider("Taille du jeux de test", min_value=0.0, max_value=1.0, step=0.05, value=0.2)
        if ml_algo_model == REGRESSION_MODELS[0]:
            model = regression_lineaire(df, X, y, train_size, FEATURES, TARGET)
        if ml_algo_model == REGRESSION_MODELS[1]:
            model = regressionlasso(df, X, y, train_size)
        if ml_algo_model == REGRESSION_MODELS[2]:
            model = regression_ridge(df, X, y, train_size)
        if ml_algo_model == REGRESSION_MODELS[3]:
            model = regression_elasticnet(df, X, y, train_size)
        if ml_algo_model == REGRESSION_MODELS[4]:
            model = regression_random_forest(df, X, y, train_size)
        if ml_algo_model == REGRESSION_MODELS[5]:
            model = regression_gradient_boosting(df, X, y, train_size)
    elif ml_algo_type == ML_ALGO_TYPES[1]:
        if ml_algo_model == CLASSIFICATION_MODELS[0]:
            valeur = st.number_input("Saisissez la valeur test_size :", step=0.01, value=0.02)
            valeur = 0.2
            model = logistic_regression(X, y, valeur)
        if ml_algo_model == CLASSIFICATION_MODELS[1]:
            model = tree_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[2]:
            model = k_neighbors_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[3]:
            model = support_vector_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[4]:
            model = naive_bayes_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[5]:
            model = xgboost_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[6]:
            model = sgd_classifier(X, y)
    if model is not None:
        st.subheader("Cross Validation")
        if ml_algo_type == ML_ALGO_TYPES[0]:
            validation_k_fold_regression(X, y, FEATURES, model)
        elif ml_algo_type == ML_ALGO_TYPES[1]:
            validation_k_fold_classification(X, y, FEATURES, model)

def run(df):
    st.subheader("Choix de la Target (variable expliquée)")
    # Choix de la target
    TARGET = select_target(df)
    y = df[TARGET]

    if y is not None:
        st.subheader("Colonne Taget")
        st.dataframe(y.head())

    # Encodage de la target
    y = target_encoding(y)

    st.subheader("Choix des Features (variables explicatives)")
    # Choisir les Features
    FEATURES = select_features(df)
    X = df[FEATURES]

    # Encodage des features
    X = feature_encoding(X)

    # Standardisation
    # X = data_standardisation(X)

    st.subheader("Sélection du type d'algorithme (Regression / Classification)")
    # Choix du type
    ml_algo_type = select_algo_type()

    st.subheader("Choix du modèle d'apprentissage")
    # Choix du modèle
    ml_algo_model = select_algo_model(ml_algo_type)

    st.dataframe(X)
    st.subheader("Démarrage de l'apprentissage")
    if st.checkbox("Démarrer le process"):
        run_machine_learning(df, ml_algo_type, ml_algo_model, X, y, FEATURES, TARGET)
    if st.checkbox("Démarrer la comparaison"):
        if ml_algo_type == ML_ALGO_TYPES[0]:
            compare_regression_models(X, y, FEATURES)
        elif ml_algo_type == ML_ALGO_TYPES[1]:    
            compare_classification_models(X, y, FEATURES)
    return df