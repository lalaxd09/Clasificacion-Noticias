"""ROSALES ONOFRE TANIA
#6BV1
#Inteligencia Artificial
#Última actualizacó:17/24/2023
#Programa utilizando capa de embeding pre-entrenada
#El programa recibe un archivo csv con 3 columnas que la principal debe tener una descripción de alguna noticia y la categoría que serán
las clases en este caso 4 clases(deportes,estados,mundo y politica).
La recuperación de las observaciones fue de 2000 datos con clases no balanceadas para ello se hace un balanceo con oversampling
#En este caso los datos requeridos son csv que tenga 4 clases,con observaciones balanceadas
#Funciones requeridas
#Normalizar->Ayuda al preprocesamiento del texto que se encuentra en preprocesamiento.py"""






import pandas as pd
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
from preprocesamiento import  normalizar,frequency_words
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
#Leer CSV
df = pd.read_csv("noticias_clases.csv")

#Aplicamos función que se encuentra en el arachivo preprocesamiento
df['Documento_Normalizado'] = df['Contenido'].apply(normalizar)

#Verificar palabras con mayor frecuencia
#frequency_words(df['Contenido'])

#Lista de palabras del documento ya normalizado
sentences = [text.split() for text in df['Documento_Normalizado'].tolist()]

#obtenemos el numero de clases
num_classes = df['Categoria'].nunique()

# Entrenamos el modelo Word2Vec
model = Word2Vec(sentences, vector_size=400, window=5, min_count=1, workers=4)

# Crear la matriz de características de entrada que va a tener 4000
X = np.zeros((len(sentences), 400))
for i, sentence in enumerate(sentences):
    for word in sentence:
        X[i] += model.wv[word]
    X[i] /= len(sentence)

#Convertimos etiquetas en formato numero
mlb = MultiLabelBinarizer()
#Transformas cada eitqueta de categoria en una lista de una sola etiqueta
y = mlb.fit_transform(df['Categoria'].apply(lambda x: [x]))
#Se obtiene la etiqueta de categoria numeria correspondiente a cada uno de los ejemplos de entrenamiento
y = np.argmax(y, axis=1)


# Definir el número de folds
k = 5

# Crear el objeto StratifiedKFold
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Crear listas para almacenar los resultados
loss_scores = []
accuracy_scores = []

# Iterar sobre los k-folds
for train_index, test_index in skf.split(X, y):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Aplicar oversampling al conjunto de entrenamiento
    sm = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    #Crear  numero entero en un vector binario de tamaño 4
    y_train_res = to_categorical(y_train_res, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)

    # Crear el modelo de clasificación
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_shape=(400,)))
    model.add(Dense(units=4, activation='softmax'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X_train_res, y_train_res, epochs=14, batch_size=32, validation_data=(X_test, y_test))

    # Evaluar el rendimiento del modelo
    loss, accuracy = model.evaluate(X_test, y_test)

    # Almacenar los resultados en las listas
    loss_scores.append(loss)
    accuracy_scores.append(accuracy)

# Calcular los resultados promedio
avg_loss = np.mean(loss_scores)
avg_accuracy = np.mean(accuracy_scores)

# Imprimir los resultados promedio
print('Loss promedio:', avg_loss)
print('Accuracy promedio:', avg_accuracy)
