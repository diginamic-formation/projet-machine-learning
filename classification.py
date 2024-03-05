import streamlit as st
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree


def plot_matrice_cofusion(y_test, y_pred):
    st.subheader("Matrice de confusion")
    plt.figure(figsize=(6, 6))
    matrice_confusion = confusion_matrix(y_test, y_pred)
    sns.heatmap(matrice_confusion, annot=True, fmt=".0f", cmap="Blues", cbar=False)
    plt.xlabel("Prédictions")
    plt.ylabel("Réelles")
    st.pyplot()


def tree_classifier(X, y, valeur=0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=valeur, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    # Entraîner le classifieur sur les données d'entraînement
    model.fit(X_train, y_train)
    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    # Évaluer l'exactitude du classifieur
    exactitude = accuracy_score(y_test, y_pred)
    # Afficher l'exactitude
    st.success(f"Exactitude : {exactitude:.2f}")
    # Afficher le rapport de classification
    rapport_classification = classification_report(y_test, y_pred)
    st.subheader("Rapport de Classification :")
    st.write(rapport_classification)
    st.subheader("Représentation Graphique de l'Arbre de Décision :")
    plt.figure(figsize=(15, 10))
    plot_tree(model, filled=True, feature_names=X.columns,class_names=[str(i) for i in model.classes_])
    st.pyplot()

    # Afficher la matrice de confusion avec Seaborn
    plot_matrice_cofusion(y_test,y_pred)
    return model


def logistic_regression(X, y, valeur=0.2):
    model = LogisticRegression()

    # acquisition des nom des colonnes
    # creation d'un dataframe et ajout de la colonne target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=valeur, random_state=42)

    clf_data = model.fit(X_train, y_train)
    # predict en classification permet de prédire la classe de sortie

    y_pred = model.predict(X_test)
    # predict_proba en classification permet de prédire les probabilités appartenant à chaque classe
    y_prob = model.predict_proba(X_test)

    st.subheader('rapport de classification')
    cr_pred = classification_report(y_test, y_pred, output_dict=True)
    st.table(cr_pred)

    # Matrice de confusion
    plot_matrice_cofusion(y_test,y_pred)

    return model


def k_neighbors_classifier(X, y, test_size=0.2):
    st.write("KNeighbors Classifier")
    model = KNeighborsClassifier(n_neighbors=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # Entraîner le modèle
    model.fit(X_train, y_train)
    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)
    # Évaluer les performances du modèle
    accuracy = accuracy_score(y_test, y_pred)
    st.write(accuracy)
    report = classification_report(y_test, y_pred)
    reformatted_report = report.replace('\n\n', '\n')  # Supprimer les doubles sauts de ligne

    # Afficher le rapport de classification dans Streamlit
    st.text_area("Rapport de Classification", value=reformatted_report, height=300)
    # Matrice de confusion
    plot_matrice_cofusion(y_test, y_pred)
    return model
