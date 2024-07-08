import streamlit as st
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def plot_matrice_cofusion(y_test, y_pred, figsize=(6, 3)):
    """
    Affiche une matrice de confusion avec un graphique de heatmap.

    Paramètres :
    y_test : array-like
        Les vraies étiquettes de classe.
    y_pred : array-like
        Les étiquettes de classe prédites par le modèle.
    figsize : tuple, optionnel, par défaut (6, 3)
        La taille de la figure pour le graphique de la matrice de confusion.

    Cette fonction utilise Streamlit pour afficher le graphique dans une application web.
    """
    # Affichage d'un sous-titre pour la matrice de confusion
    st.subheader("Matrice de confusion")

    # Création d'une nouvelle figure avec la taille spécifiée
    plt.figure(figsize=figsize)

    # Calcul de la matrice de confusion
    matrice_confusion = confusion_matrix(y_test, y_pred)

    # Création d'un graphique heatmap pour la matrice de confusion
    sns.heatmap(matrice_confusion, annot=True, fmt=".0f", cmap="Reds", cbar=False)

    # Étiquetage des axes
    plt.xlabel("Prédictions")
    plt.ylabel("Réelles")

    # Affichage du graphique dans Streamlit
    st.pyplot()



def tree_classifier(X, y, valeur=0.2):
    """
    Entraîne et évalue un classifieur d'arbre de décision.

    Paramètres :
    X : DataFrame
        Les caractéristiques utilisées pour l'entraînement.
    y : Series ou array-like
        Les étiquettes de classe.
    valeur : float, optionnel, par défaut 0.2
        La proportion de l'ensemble de données à utiliser comme ensemble de test.

    Cette fonction utilise Streamlit pour afficher les résultats dans une application web.
    """
    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=valeur, random_state=42)

    # Création et entraînement du modèle d'arbre de décision
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calcul de l'exactitude du modèle
    exactitude = accuracy_score(y_test, y_pred)

    # Affichage de l'exactitude dans Streamlit
    st.success(f"Exactitude : {exactitude:.2f}")

    # Génération et affichage du rapport de classification dans Streamlit
    rapport_classification = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Rapport de Classification :")
    st.table(rapport_classification)

    # Affichage graphique de l'arbre de décision dans Streamlit
    st.subheader("Représentation Graphique de l'Arbre de Décision :")
    plt.figure(figsize=(15, 10))
    X = pd.DataFrame(X)
    plot_tree(model, filled=True, feature_names=X.columns, class_names=[str(i) for i in model.classes_])
    st.pyplot()


    # Afficher la matrice de confusion avec Seaborn
    plot_matrice_cofusion(y_test,y_pred)
    return model


def logistic_regression(X, y, valeur=0.2):
    """
    Entraîne et évalue un modèle de régression logistique.

    Paramètres :
    X : DataFrame
        Les caractéristiques utilisées pour l'entraînement.
    y : Series ou array-like
        Les étiquettes de classe.
    valeur : float, optionnel, par défaut 0.2
        La proportion de l'ensemble de données à utiliser comme ensemble de test.

    Cette fonction utilise Streamlit pour afficher les résultats dans une application web.
    """
    # Création du modèle de régression logistique
    model = LogisticRegression()

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=valeur, random_state=42)

    # Entraînement du modèle sur les données d'entraînement
    clf_data = model.fit(X_train, y_train)

    # Prédiction des étiquettes sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluation de l'exactitude du classifieur
    exactitude = accuracy_score(y_test, y_pred)

    # Affichage de l'exactitude dans Streamlit
    st.success(f"Exactitude : {exactitude:.2f}")

    # Prédiction des probabilités pour chaque classe sur l'ensemble de test
    y_prob = model.predict_proba(X_test)

    # Affichage du rapport de classification dans Streamlit
    st.subheader('rapport de classification')
    cr_pred = classification_report(y_test, y_pred, output_dict=True)
    st.table(cr_pred)

    # Affichage de la matrice de confusion dans Streamlit
    plot_matrice_cofusion(y_test, y_pred)

    return model



def k_neighbors_classifier(X, y, test_size=0.2):
    """
    Entraîne et évalue un modèle de K-Nearest Neighbors (KNN) pour la classification.

    Paramètres :
    X : DataFrame
        Les caractéristiques utilisées pour l'entraînement.
    y : Series ou array-like
        Les étiquettes de classe.
    test_size : float, optionnel, par défaut 0.2
        La proportion de l'ensemble de données à utiliser comme ensemble de test.

    Cette fonction utilise Streamlit pour afficher les résultats dans une application web.
    """
    # Afficher le titre du modèle dans Streamlit
    st.write("KNeighbors Classifier")

    # Création du modèle K-Nearest Neighbors avec 3 voisins
    model = KNeighborsClassifier(n_neighbors=3)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Entraînement du modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédiction des étiquettes sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluation de l'exactitude du classifieur
    exactitude = accuracy_score(y_test, y_pred)

    # Affichage de l'exactitude dans Streamlit
    st.success(f"Exactitude : {exactitude:.2f}")

    # Affichage du rapport de classification dans Streamlit
    st.subheader('rapport de classification')
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(report)

    # Affichage de la matrice de confusion dans Streamlit
    plot_matrice_cofusion(y_test, y_pred)

    return model


def support_vector_classifier(X, y, test_size=0.2):
    """
    Entraîne et évalue un modèle de Support Vector Classifier (SVC) pour la classification.

    Paramètres :
    X : DataFrame
        Les caractéristiques utilisées pour l'entraînement.
    y : Series ou array-like
        Les étiquettes de classe.
    test_size : float, optionnel, par défaut 0.2
        La proportion de l'ensemble de données à utiliser comme ensemble de test.

    Retourne :
    model : SVC
        Le modèle SVC entraîné.

    Cette fonction utilise Streamlit pour afficher les résultats dans une application web.
    """
    # Afficher le titre du modèle dans Streamlit
    st.write("Support Vector Classifier")

    # Création du modèle Support Vector Classifier avec probabilité
    model = SVC(probability=True)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Entraînement du modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédiction des étiquettes sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluation de l'exactitude du classifieur
    exactitude = accuracy_score(y_test, y_pred)

    # Affichage de l'exactitude dans Streamlit
    st.success(f"Exactitude : {exactitude:.2f}")

    # Affichage du rapport de classification dans Streamlit
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Rapport de Classification")
    st.table(report)

    # Affichage de la matrice de confusion dans Streamlit
    plot_matrice_cofusion(y_test, y_pred)

    return model



def naive_bayes_classifier(X, y, test_size=0.2):
    """
    Entraîne et évalue un modèle de Naive Bayes Classifier pour la classification.

    Paramètres :
    X : DataFrame
        Les caractéristiques utilisées pour l'entraînement.
    y : Series ou array-like
        Les étiquettes de classe.
    test_size : float, optionnel, par défaut 0.2
        La proportion de l'ensemble de données à utiliser comme ensemble de test.

    Retourne :
    model : GaussianNB
        Le modèle Naive Bayes entraîné.

    Cette fonction utilise Streamlit pour afficher les résultats dans une application web.
    """
    # Afficher le titre du modèle dans Streamlit
    st.write("Naive Bayes Classifier")

    # Création du modèle Naive Bayes
    model = GaussianNB()

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Entraînement du modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédiction des étiquettes sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluation de l'exactitude du classifieur
    exactitude = accuracy_score(y_test, y_pred)

    # Affichage de l'exactitude dans Streamlit
    st.success(f"Exactitude : {exactitude:.2f}")

    # Affichage du rapport de classification dans Streamlit
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Rapport de Classification")
    st.table(report)

    # Affichage de la matrice de confusion dans Streamlit
    plot_matrice_cofusion(y_test, y_pred)

    return model


def xgboost_classifier(X, y, test_size=0.2):
    """
    Entraîne et évalue un modèle de XGBoost Classifier pour la classification.

    Paramètres :
    X : DataFrame
        Les caractéristiques utilisées pour l'entraînement.
    y : Series ou array-like
        Les étiquettes de classe.
    test_size : float, optionnel, par défaut 0.2
        La proportion de l'ensemble de données à utiliser comme ensemble de test.

    Retourne :
    model : XGBClassifier
        Le modèle XGBoost entraîné.

    Cette fonction utilise Streamlit pour afficher les résultats dans une application web.
    """
    # Afficher le titre du modèle dans Streamlit
    st.write("XGBoost Classifier")

    # Création du modèle XGBoost avec une graine aléatoire fixe pour la reproductibilité
    model = XGBClassifier(random_state=42)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Entraînement du modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédiction des étiquettes sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluation de l'exactitude du classifieur
    exactitude = accuracy_score(y_test, y_pred)

    # Affichage de l'exactitude dans Streamlit
    st.success(f"Exactitude : {exactitude:.2f}")

    # Génération du rapport de classification
    report = classification_report(y_test, y_pred, output_dict=True)

    # Affichage du rapport de classification dans Streamlit
    st.subheader("Rapport de Classification")
    st.table(report)

    # Affichage de la matrice de confusion dans Streamlit
    plot_matrice_cofusion(y_test, y_pred)

    return model


def sgd_classifier(X, y, test_size=0.2):
    """
    Entraîne et évalue un modèle de Stochastic Gradient Descent (SGD) Classifier pour la classification.

    Paramètres :
    X : DataFrame
        Les caractéristiques utilisées pour l'entraînement.
    y : Series ou array-like
        Les étiquettes de classe.
    test_size : float, optionnel, par défaut 0.2
        La proportion de l'ensemble de données à utiliser comme ensemble de test.

    Retourne :
    model : SGDClassifier
        Le modèle SGD entraîné.

    Cette fonction utilise Streamlit pour afficher les résultats dans une application web.
    """
    # Afficher le titre du modèle dans Streamlit
    st.write("Stochastic Gradient Descent Classifier")

    # Création du modèle SGD avec une graine aléatoire fixe pour la reproductibilité
    model = SGDClassifier(random_state=42)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Entraînement du modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédiction des étiquettes sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluation de l'exactitude du classifieur
    exactitude = accuracy_score(y_test, y_pred)

    # Affichage de l'exactitude dans Streamlit
    st.success(f"Exactitude : {exactitude:.2f}")

    # Génération du rapport de classification
    report = classification_report(y_test, y_pred, output_dict=True)

    # Affichage du rapport de classification dans Streamlit
    st.subheader("Rapport de Classification")
    st.table(report)

    # Affichage de la matrice de confusion dans Streamlit
    plot_matrice_cofusion(y_test, y_pred)

    return model
