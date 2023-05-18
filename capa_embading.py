"""#ROSALES ONOFRE TANIA
#6BV1
#Inteligencia Artificial
#17/24/2023

#Programa para la creación de la capa de embading por medio de tensor flow

#En este caso los datos requeridos son un csv con contenido de noticias y etiquitado ya balanceada dos observaciones de las categorias
#Funciones
#1.Normalizar->Ayuda al preprocesamiento del texto que se encuentra en preprocesamiento.py"""

import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocesamiento import normalizar
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar datos
df = pd.read_csv('noticias_clases.csv')

# Normalizar texto
df['Documento_Normalizado'] = df['Contenido'].apply(normalizar)

# Obtener número de clases
num_classes = df['Categoria'].nunique()
print('Número de clases:', num_classes)

# Codificar etiquetas
labels, levels = pd.factorize(df['Categoria'])
y = to_categorical(labels, num_classes=num_classes)

# Crear diccionario de palabras
word_dict = {}
for sentence in df['Documento_Normalizado']:
    for word in sentence:
        if word not in word_dict:
            word_dict[word] = len(word_dict) + 1

# Longitud máxima de secuencia
max_len = max([len(x) for x in df['Documento_Normalizado']])

# Crear matriz de características
X = []
for sentence in df['Documento_Normalizado']:
    seq = []
    for word in sentence:
        seq.append(word_dict[word])
    X.append(seq)
X = pad_sequences(X, maxlen=max_len, padding='post')



# Definir el modelo
model = Sequential()
#Primera capa convertir palabras de cada documento
model.add(Embedding(len(word_dict) + 1, 50, input_length=max_len))
#segunda capa para aplanar los los vectores de salida de la capa anterior
model.add(Flatten())
#Produce salida del tamaño de las clsaes
model.add(Dense(num_classes, activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# dividir datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el número de folds y crear la instancia de KFold
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Realizar la validación cruzada
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
