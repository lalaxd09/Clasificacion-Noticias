"""""#ROSALES ONOFRE TANIA
#6BV1
#Inteligencia Artificial
#17/24/2023

#Código para la creación de los modelos de comparación bayes ingenuo,knn,regresión logistica con k-folder-validation
#En este caso los datos requeridos son un csv con contenido de noticias y etiquitado ya balanceada dos observaciones de las categorias
#Funciones
#1.Normalizar->Ayuda al preprocesamiento del texto que se encuentra en preprocesamiento.py"""


from preprocesamiento import normalizar
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("noticias_clases.csv")
#Aplicamos funcion normalizar en preprocesamiento.py
df['Documento_Normalizado'] = df['Contenido'].apply(normalizar)
 #Vectoriza los documentos
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
tfidf_matrix = vectorizer.fit_transform(df['Documento_Normalizado'])

# Divide los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['Categoria'], test_size=0.2, random_state=42)

# Aplicar oversampling al conjunto de entrenamiento
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Naive Bayes con el conjunto de entrenamiento balanceado

clf_nb = MultinomialNB()
clf_nb.fit(X_train_res, y_train_res)
k = 5 # número de pliegues en k-fold cross-validation
kf = KFold(n_splits=k)
scores = cross_val_score(clf_nb, X_train_res, y_train_res, cv=kf)
print("Precisión Naive Bayes:", np.mean(scores))

# Evaluar el modelo en el conjunto de prueba
y_pred_nb = clf_nb.predict(X_test)
precision_nb = accuracy_score(y_test, y_pred_nb)
print("Precisión Naive Bayes en el conjunto de prueba:", precision_nb)
print(classification_report(y_test, y_pred_nb))

# Regresión logística con el conjunto de entrenamiento balanceado
#Modelo
clf_lr = LogisticRegression(random_state=42)
clf_lr.fit(X_train_res, y_train_res)
k = 5 # número de pliegues en k-fold cross-validation
kf = KFold(n_splits=k)
scores = cross_val_score(clf_lr, X_train_res, y_train_res, cv=kf)
print("Precisión Regresión Logística:", np.mean(scores))

# Evaluar el modelo en el conjunto de prueba
y_pred_lr = clf_lr.predict(X_test)
precision_lr = accuracy_score(y_test, y_pred_lr)
print("Precisión Regresión Logística en el conjunto de prueba:", precision_lr)
print(classification_report(y_test, y_pred_lr))

# KNN con el conjunto de entrenamiento balanceado
clf_knn = KNeighborsClassifier(n_neighbors=5)
#Ajuste de emtrenamiento
clf_knn.fit(X_train_res, y_train_res)
k = 5 # número de pliegues en k-fold cross-validation
kf = KFold(n_splits=k)
scores = cross_val_score(clf_knn, X_train_res, y_train_res, cv=kf)
print("Precisión KNN:", np.mean(scores))

# Evaluar el modelo en el conjunto de prueba
y_pred_knn = clf_knn.predict(X_test)
precision_knn = accuracy_score(y_test, y_pred_knn)
print("Precisión KNN en el conjunto de prueba:", precision_knn)
print(classification_report(y_test, y_pred_knn))