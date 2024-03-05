import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from regression import regression_lineaire, regression_ridge, regressionlasso
from classification import tree_classifier, k_neighbors_classifier, logistic_regression
from validation import validation_k_fold

ML_ALGO_TYPES = ["Regression", "Classification"]
REGRESSION_MODELS = ["Regression lineaire", "Lasso", "Ridge"]
CLASSIFICATION_MODELS = ["Logistic regression", "Tree Classifier", "KNeighbors Classifier"]


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
        label_mapping_df = pd.DataFrame({'Category': label_encoder.classes_, 'Encoded': label_encoder.transform(label_encoder.classes_)})
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
    model = None
    if ml_algo_type == ML_ALGO_TYPES[0]:
        if ml_algo_model == REGRESSION_MODELS[0]:
            train_size = st.slider("Taille du jeux de test", min_value=0.0, max_value=1.0, step=0.05,value=0.2)
            model = regression_lineaire(df, X, y, train_size,FEATURES, TARGET )
        if ml_algo_model == REGRESSION_MODELS[1]:
            model = regressionlasso(df, X, y)
        if ml_algo_model == REGRESSION_MODELS[2]:
            model = regression_ridge(df, X, y)

    elif ml_algo_type == ML_ALGO_TYPES[1]:
        if ml_algo_model == CLASSIFICATION_MODELS[0]:
            valeur = st.number_input("Saisissez la valeur test_size :", step=0.01, value=0.02)
            valeur = 0.2
            model = logistic_regression(X, y, valeur)
        if ml_algo_model == CLASSIFICATION_MODELS[1]:
            model = tree_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[2]:
            model = k_neighbors_classifier(X, y)

    if model is not None:
        validation_k_fold(X,y, FEATURES, model)


def run(df):
    # choisir la target
    TARGET = select_target(df)
    y = df[TARGET]

    if y is not None:
        st.subheader("Colonne Taget")
        st.dataframe(y.head())

    # Encodage de la target
    y = target_encoding(y)

    # Choisir les Features
    FEATURES = select_features(df)
    X = df[FEATURES]

    # Encodage des features
    X = feature_encoding(X)

    # Standardisation
    # X = data_standardisation(X)

    # Choix du type
    ml_algo_type = select_algo_type()

    # Choix du modèle
    ml_algo_model = select_algo_model(ml_algo_type)

    st.dataframe(X)
    if st.checkbox("Démarrer le process"):
        run_machine_learning(df, ml_algo_type, ml_algo_model, X, y, FEATURES, TARGET)

    # Prediction

    # Evaluation et validation

    return df
