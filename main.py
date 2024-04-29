import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write('''
# UCAO Application pour la prévision des fleurs d'Iris
Cette application prédit la catégorie des fleurs d'Iris
''')

st.sidebar.header("Les paramètres d'entrée")

def user_input():
    sepal_length=st.sidebar.slider('La longueur du Sepal',4.3,7.9,5.3)
    sepal_width=st.sidebar.slider('La largeur du Sepal',2.0,4.4,3.3)
    petal_length=st.sidebar.slider('La longueur du Petal',1.0,6.9,2.3)
    petal_width=st.sidebar.slider('La largeur du Petal',0.1,2.5,1.3)
    data={'sepal_length':sepal_length,
    'sepal_width':sepal_width,
    'petal_length':petal_length,
    'petal_width':petal_width
    }
    fleur_parametres=pd.DataFrame(data,index=[0])
    return fleur_parametres

df=user_input()

st.subheader('On veut trouver la catégorie de cette fleur')
st.write(df)

# Chargement des données Iris
iris=datasets.load_iris()
X = iris.data
y = iris.target

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèles
models = {
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Logistic Regression': LogisticRegression()
}

# Sidebar pour la sélection du modèle
model_name = st.sidebar.selectbox("Sélectionnez le modèle", list(models.keys()))

# Entraînement du modèle sélectionné
model = models[model_name]
model.fit(X_train, y_train)

# Prédiction
prediction = model.predict(df)

st.subheader("La catégorie de la fleur d'Iris est:")
st.write(iris.target_names[prediction])

# Evaluation des modèles
st.subheader('Evaluation des modèles')

# Calcul des scores
accuracy = accuracy_score(y_test, model.predict(X_test))
precision = precision_score(y_test, model.predict(X_test), average='weighted')
recall = recall_score(y_test, model.predict(X_test), average='weighted')
f1 = f1_score(y_test, model.predict(X_test), average='weighted')

# Affichage des scores
st.write(f"Accuracy: {accuracy}")
st.write(f"Precision: {precision}")
st.write(f"Recall: {recall}")
st.write(f"F1-score: {f1}")

# Matrice de confusion
confusion_matrix = pd.crosstab(y_test, model.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
st.subheader("Matrice de confusion")
st.write(confusion_matrix)

# Affichage des scores dans un graphique
scores_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Score': [accuracy, precision, recall, f1]
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Score', data=scores_df)
plt.title('Scores des modèles')
plt.ylim(0, 1)
plt.xticks(rotation=45)
st.pyplot()
