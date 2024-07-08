import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC


def validation_k_fold_classification(X, y, FEATURES, model):
    """
    Effectue une validation croisée K-Fold pour évaluer un modèle de classification.
    
    Args:
        X (pd.DataFrame): Les caractéristiques d'entrée.
        y (pd.Series): Les étiquettes de classe.
        FEATURES (list): La liste des caractéristiques à utiliser pour l'entraînement des modèles.
        model (object): Le modèle de classification à évaluer.

    Returns:
        dict: Les scores moyens de précision, rappel et F1-score.
    """
    kf = KFold(n_splits=6, shuffle=True, random_state=2021)
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        ligne = i // 3
        colonne = i % 3
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train[FEATURES], y_train)
        y_pred = model.predict(X_test[FEATURES])

        # Calculer les scores
        scores["accuracy"].append(accuracy_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred, average='weighted'))
        scores["recall"].append(recall_score(y_test, y_pred, average='weighted'))
        scores["f1"].append(f1_score(y_test, y_pred, average='weighted'))

        # Tracer la matrice de confusion
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Reds", ax=axes[ligne, colonne])
        axes[ligne, colonne].set_title(f'Fold {i+1}')

    st.subheader("Comportement du modèle")
    st.pyplot(fig)  # Afficher le graphique dans Streamlit

    # Calculer les scores moyens
    mean_scores = {metric: sum(values) / len(values) for metric, values in scores.items()}

    st.subheader("Moyenne des métriques de validation du modèle")
    mean_scores_df = pd.DataFrame.from_dict(mean_scores, orient='index', columns=['Mean Score'])
    st.table(mean_scores_df)

    return mean_scores


def validation_k_fold_regression(X, y, FEATURES, model):
    """
    Effectue une validation croisée k-fold sur un modèle de régression et affiche les résultats.

    Parameters:
    X (pd.DataFrame): Features dataset
    y (pd.Series): Target variable
    FEATURES (list): List of feature names
    model: Regression model instance
    
    Returns:
    dict: Mean scores for R2, MSE, and MAE across all folds
    """
    kf = KFold(n_splits=6, shuffle=True, random_state=2021)
    columns = ["R2", "MSE", "MAE"]
    stats_df = pd.DataFrame(columns=columns)
    lignes = 2
    colonnes = 3
    fig, axes = plt.subplots(lignes, colonnes, figsize=(15, 10))
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        ligne = i // colonnes
        colonne = i % colonnes
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train[FEATURES], y_train)
        y_pred = model.predict(X_test[FEATURES])
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        stats_df.loc[i] = [round(r2, 4), round(mse, 4), round(mae, 4)]

        # Créer le graphique de dispersion
        axes[ligne, colonne].scatter(y_test, y_pred)
        axes[ligne, colonne].plot(np.arange(y_test.min(), y_test.max()),
                                  np.arange(y_test.min(), y_test.max()),
                                  color='red', linestyle='--')  # Ligne de référence
        
        axes[ligne, colonne].set_title(f'R2: {round(r2, 4)} - MSE: {round(mse, 4)} - MAE: {round(mae, 4)}')
        axes[ligne, colonne].set_xlabel("Valeurs réelles")
        axes[ligne, colonne].set_ylabel("Valeurs prédites")

    mean_r2 = stats_df["R2"].mean()
    mean_mse = stats_df["MSE"].mean()
    mean_mae = stats_df["MAE"].mean()
    
    # Affichage des résultats dans Streamlit
    st.subheader("Comportement du modèle")
    st.pyplot(fig)
    
    st.subheader("Moyenne des métriques de validation du modèle")
    stats_mean_df = stats_df.mean().to_frame().reset_index()
    stats_mean_df = stats_mean_df.rename(columns={'index': 'Metric', 0: 'Mean'})
    st.table(stats_mean_df)

    return {"R2": mean_r2, "MSE": mean_mse, "MAE": mean_mae}


def compare_regression_models(X, y, FEATURES):
    """
    Compare plusieurs modèles de régression en utilisant la validation croisée k-fold et sélectionne le meilleur modèle.

    Parameters:
    X (pd.DataFrame): Features dataset
    y (pd.Series): Target variable
    FEATURES (list): List of feature names

    Returns:
    str: Name of the best model
    model: Best model instance
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42)
    }
    
    results = {}
    
    st.header("Comparaison des modèles de régression")
    
    for model_name, model in models.items():
        st.subheader(model_name)
        scores = validation_k_fold_regression(X, y, FEATURES, model)
        results[model_name] = scores
    
    # Sélection du meilleur modèle basé sur la moyenne des scores R2, MSE et MAE
    best_model_name = max(results, key=lambda x: results[x]["R2"])  # Par exemple, basé sur le R2
    st.subheader(f"Meilleur modèle: {best_model_name} avec un R2 moyen de {results[best_model_name]['R2']:.4f}")
    
    return best_model_name, models[best_model_name]


def compare_classification_models(X, y, FEATURES):
    """
    Compare plusieurs modèles de classification et sélectionne le meilleur modèle basé sur les scores moyens de précision, rappel et F1-score.
    
    Args:
        X (pd.DataFrame): Les caractéristiques d'entrée.
        y (pd.Series): Les étiquettes de classe.
        FEATURES (list): La liste des caractéristiques à utiliser pour l'entraînement des modèles.

    Returns:
        tuple: Le nom du meilleur modèle et l'instance du meilleur modèle.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Tree Classifier": DecisionTreeClassifier(random_state=42),
        "KNeighbors Classifier": KNeighborsClassifier(n_neighbors=3),
        "Support Vector Classifier": SVC(probability=True),
        "Naive Bayes Classifier": GaussianNB(),
        "XGBoost Classifier": XGBClassifier(random_state=42),
        "Stochastic Gradient Descent Classifier": SGDClassifier(random_state=42)
    }
    
    results = {}
    
    st.header("Comparaison des modèles de classification")
    
    for model_name, model in models.items():
        st.subheader(model_name)
        mean_scores = validation_k_fold_classification(X, y, FEATURES, model)
        results[model_name] = mean_scores
    
    # Sélectionner le modèle avec le meilleur compromis entre précision, rappel et F1-score
    best_model_name = max(results, key=lambda model: (results[model]['accuracy'] + results[model]['precision'] + results[model]['recall'] + results[model]['f1']) / 4)
    best_model_score = results[best_model_name]
    
    st.subheader(f"Meilleur modèle: {best_model_name} avec des scores moyens de:")
    st.write(f"Accuracy: {best_model_score['accuracy']:.4f}, Precision: {best_model_score['precision']:.4f}, Recall: {best_model_score['recall']:.4f}, F1-Score: {best_model_score['f1']:.4f}")
    
    return best_model_name, models[best_model_name]