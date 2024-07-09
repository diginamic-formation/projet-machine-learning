# Projet Machine Learning

Groupe 1 :
- Garmi GROBOST

- Rijandrisolo

- Jaouad SALAHY

- Fares MENTSEUR

Objectif du projet : 
Conception d'une application streamlit pour présenter un pipeline complet de Machine Learning . Cette application interactive nous permettra d'explorer, d'analyser et de prédire à partir de jeux de données sur le vin et le diabète.

### Configuration de l'application 
L'application est conçue avec trois pages principales : Chargement des données, Gestion des données, et Machine Learning.
Le code commence par configurer la page Streamlit avec un titre et une mise en page étendue. Une barre latérale est utilisée pour la navigation entre les pages, implémentée à l'aide d'un bouton radio.
### Aperçu du jeu de données diabète :
![image](https://github.com/diginamic-formation/projet-machine-learning/assets/75080561/fe9bf587-2e2e-45b5-9c06-98f4f35b8dac)

### Aperçu du jeu de données vin :
![image](https://github.com/diginamic-formation/projet-machine-learning/assets/75080561/ccf14973-c13e-4694-a26c-a2fda2fdce19)

### Traitement et analyse des données 
On a procéder d'abord à une analyse sur les données afin de détecter les valeurs manquantes, les colonnes vides ainsi permettre à l'utilisateur de les corriger : 

 -  sélectionner et supprimer les colonnes vides
 - effacer, remplacer une valeur manquante par la moyenne, la médiane ou par une valeur personnalisée. 
 - changer de type de variable.
 #####  Analyse descriptive : 
Obtenir un aperçu complet des caractéristiques de nos données en utilisant des statistiques telles que la moyenne, l’écart-type, la médiane, les quartiles, etc.
Exemple du jeu de données Vin:
![image](https://github.com/diginamic-formation/projet-machine-learning/assets/75080561/dd4b2060-6cbb-4504-bc0e-b5d485654462)

##### Visualisations :
Les histogrammes montrent la distribution des données pour chaque caractéristique, aidant à identifier les biais et les distributions non normales.
![image](https://github.com/diginamic-formation/projet-machine-learning/assets/75080561/b8e73afa-993c-4c88-a4ea-66286edad574)

##### Corrélations :
Identifier les variables les plus corrélées à notre variable cible grâce à l'analyse des coefficients de corrélation.
##### Standardisation :
Standardiser les données pour faciliter l'entraînement des modèles de machine learning en réduisant les biais dus aux échelles différentes des caractéristiques.
### Pipeline de Machine Learning
##### Sélection de l'Algorithme
Offrir à l'utilisateur le choix entre plusieurs algorithmes adaptés au type de problème (classification ou régression) détecté.
##### Split des Données
Diviser automatiquement les données en ensembles d'entraînement et de test, avec une option pour ajuster les proportions.
##### Entraînement du Modèle
Lancer l'entraînement du modèle sélectionné sur les données d'entraînement, en affichant la progression.
##### Prédictions
Effectuer des prédictions sur l'ensemble de test et offrir la possibilité de prédire sur de nouvelles données fournies par l'utilisateur.
### Évaluation du modèle de Régression
##### Métriques de performance :
Afficher le R2, le MSE, le MAE
##### Visualisations :
Afficher un graphique comparant les valeurs réelles et prédites de la cible.
Afficher l'évolution de l'erreur quadratique moyenne (MSE) par rapport au paramètre alpha sous forme de graphique.
##### Comparaison des modèles :
1.Utiliser la validation croisée K-Fold pour évaluer chaque modèle.

2.Calculer le R2, MSE, MAE pour chaque modèle.

3.Sélectionner le meilleur modèle basé sur le R2 moyen
### Évaluation du modèle de Classification
##### Métriques de performance :
Afficher les métriques telles que l'accuracy, la précision, le recall
##### Visualisations :
Générer la matrice de confusion
##### Comparaison des modèles :
1.Utiliser la validation croisée K-Fold pour évaluer chaque modèle
.
2.Calculer la précision, le rappel et le F1-score pour chaque modèle.

3.Comparer les scores moyens de chaque modèle.

4.Sélectionner le modèle avec le meilleur compromis basé sur ces métriques.
