# import streamlit as st
# import pandas as pd 
# import numpy as np 
# import seaborn as sns 
# import matplotlib.pyplot as plt
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge
# import statsmodels.formula.api as smf
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC

# df_diabete = pd.read_csv("diabete.csv")
# df_vin = pd.read_csv("vin.csv")

# @st.cache
# def load_data(file):
#     data = pd.read_csv(file)
#     return data

# st.sidebar.title("Sommaire")

# pages = ["Traitement des données", "Visualisation", "Modélisation", "Evaluation"]

# page = st.sidebar.radio("Aller vers la page :", pages)

# if page == pages[0] : 
#     st.write("### Traitement des données")
#     options = ['diabete', 'vin']
#     choice = st.selectbox('Choisir le jeu de donné', options)
#     if choice == 'diabete':
#         st.dataframe(df_diabete.head())
#         st.write("Dimensions du dataframe :")
#         st.write(df_diabete.shape)
#         if st.checkbox("Afficher les valeurs manquantes") : 
#             st.dataframe(df_diabete.isna().sum())
        
#         if st.checkbox("Afficher les doublons") : 
#             st.write(df_diabete.duplicated().sum())
#         # # Définir une fonction pour détecter les valeurs aberrantes en utilisant le score z
#         # def detect_outliers_zscore(df_diabete, threshold=3):
#         #     z_scores = np.abs((df_diabete - df_diabete.mean()) / df_diabete.std())
#         #     return z_scores > threshold

#         # # Sélectionner uniquement les colonnes numériques pour la détection des valeurs aberrantes
#         # numeric_columns = df_diabete.select_dtypes(include=np.number)

#         # # Appliquer la fonction de détection des valeurs aberrantes à chaque colonne numérique
#         # outliers = numeric_columns.apply(detect_outliers_zscore)

#         # # Afficher les valeurs aberrantes
#         # if st.checkbox("Afficher les valeurs aberrantes"):
#         #     st.write(outliers)
#     else :
#         st.dataframe(df_vin.head())
#         st.write("Dimensions du dataframe :")
#         st.write(df_vin.shape)
#         if st.checkbox("Afficher les valeurs manquantes") : 
#             st.dataframe(df_vin.isna().sum())
        
#         if st.checkbox("Afficher les doublons") : 
#             st.write(df_vin.duplicated().sum())

# elif page == pages[1]:
#     st.write("### Visualisation de données")
#     options = ['diabete', 'vin']
#     choice = st.selectbox('Choisir le jeu de donné', options)
#     if choice == 'diabete':
#         # Créer un histogramme pour visualiser la répartition des âges
#         fig, ax = plt.subplots()
#         sns.histplot(df_diabete['age'], kde=True, ax=ax)
#         ax.set_title("Répartition des âges des patients diabétiques")
#         ax.set_xlabel("Âge")
#         ax.set_ylabel("Nombre de patients")
#         # Enregistrer la figure
#         st.pyplot(fig)

#         # Compter le nombre de patients par sexe
#         counts_by_sex = df_diabete['sex'].value_counts()

#         # # Créer un diagramme à barres pour visualiser la répartition des patients par sexe
#         # fig, ax = plt.subplots()
#         # counts_by_sex.plot(kind='bar', ax=ax)
#         # ax.set_title("Répartition des patients par sexe")
#         # ax.set_xlabel("Sexe")
#         # ax.set_ylabel("Nombre de patients")

#         # # Afficher les étiquettes des barres
#         # for i, count in enumerate(counts_by_sex):
#         #     ax.text(i, count, str(count), ha='center', va='bottom')
#         # # Afficher le diagramme à barres dans Streamlit
#         # st.pyplot(fig)

#          # Créer un nuage de points pour visualiser la relation entre l'IMC et la pression artérielle
#         fig, ax = plt.subplots()
#         sns.scatterplot(x='bmi', y='bp', data=df_diabete, ax=ax)
#         ax.set_title("Relation entre l'IMC et la pression artérielle")
#         ax.set_xlabel("IMC")
#         ax.set_ylabel("Pression artérielle")
#         # Afficher le nuage de points dans Streamlit
#         st.plotly_chart(fig)

#         # Calculer la matrice de corrélation
#         corr_matrix = df_diabete.corr()
#         # Créer une heatmap pour afficher les corrélations
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
#         plt.title("Matrice de corrélation entre les caractéristiques")
#         plt.xticks(rotation=45)
#         plt.yticks(rotation=45)
#         # Afficher la heatmap dans Streamlit
#         st.pyplot(plt)

#         # # Compter les occurrences de chaque classe dans la colonne cible
#         # class_counts = df_diabete['target'].value_counts()
#         # # Créer un diagramme à secteurs
#         # plt.figure(figsize=(6, 6))
#         # plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
#         # plt.title("Distribution des classes cibles")
#         # plt.axis('equal')  # Assure que le diagramme à secteurs est un cercle
#         # # Afficher le diagramme dans Streamlit
#         # st.pyplot(plt)

#         # Créer un diagramme en boîte pour la distribution de la glycémie à jeun
#         fig, ax = plt.subplots()
#         sns.boxplot(x='s1', data=df_diabete, ax=ax)
#         ax.set_title("Distribution de la glycémie à jeun")
#         ax.set_xlabel("Glycémie à jeun (S1)")
#         # Afficher le diagramme dans Streamlit
#         st.pyplot(fig)

#         # Tracer une courbe de régression linéaire entre l'âge et le niveau de glucose dans le sang
#         fig = sns.lmplot(x='age', y='s1', data=df_diabete, height=6)
#         fig.set_axis_labels("Âge", "Niveau de glucose dans le sang")
#         # Afficher le graphique dans Streamlit
#         st.pyplot(fig)

# elif page == pages[2]:
#     @st.cache_resource()
#     def load_data(file):
#         data = pd.read_csv(file)
#         return data
    
#     st.write("### Modélisation")
#     file = st.file_uploader("Upload your dataset in CSV format", type=["csv"])
    
#     if file is not None:
#         data = load_data(file)
#         st.dataframe(data.head())
#         target = st.selectbox("Select the target variable", data.columns)
#         data = data.dropna(subset=[target])  # au cas où il y aura des valeurs manquantes dans la variable cible
        
#         first_val = data[target][0]
        
#         if isinstance(first_val, (int, float, np.float32)):
#             target_type = 'numerique'
#         else:
#             target_type = 'categorie'

#         if target_type == 'numerique':
#             models = ['Régression Linéaire', 'Régression Lasso', 'Régression Ridge']
#             choice_model = st.selectbox('Choisir le modèle', models)

#             if choice_model == 'Régression Linéaire':
#                 model = LinearRegression()
#                 X = data.drop(columns=[target])
#                 Y = data[target]
                
#             elif choice_model == 'Régression Lasso':
#                 model = Lasso()
#             elif choice_model == 'Régression Ridge':
#                 model = Ridge()

#             # # Séparation des données en features et target
#             # X = data.drop(columns=[target])
#             # y = data[target]

#             # # Division des données en ensemble d'entraînement et de test
#             # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             # # Entraînement du modèle
#             # model.fit(X_train, y_train)

#             # # Prédiction sur l'ensemble de test
#             # y_pred = model.predict(X_test)

    
import machine_learning
import streamlit as st
import data_management
import preprocessing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Projet machine learning",
                   layout="wide")

st.sidebar.title("Sommaire")
pages = ["Traitement des données", "Visualisation", "Modélisation", "Evaluation"]

page = st.sidebar.radio("Aller vers la page :", pages)

data = None
if page == pages[0]:
    data = data_management.preprocess()
    st.session_state["result"]= data
elif page == pages[1]:
    if "result" in st.session_state:
        preprocessing.run(st.session_state["result"])

elif page == pages[2]:
    if "result" in st.session_state:
        machine_learning.run(st.session_state["result"])