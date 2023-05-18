#ROSALES ONOFRE TANIA
#6BV1
#Inteligencia Artificial
#17/24/2023


"""""
#Datos requeridos dos noticieros con columnas que contegan contenido,titulo,categoria de la noticias en formato csv,
como son utilizado dos tipos de noticieron se quitaron categorias que no congeneaban con el otro noticiero y se hace
selección de las clases con mayor observaciones y devuelve un data frame que sera ocupando en clases.py
concanefiles->Ayuda a concatenar dos archivos csv y eliminación de etiquetas se extrae del archivo concatane_files"""

import pandas as pd

def concatanefiles(noticia1,noticia2):
    jornada = pd.read_csv(noticia1)
    #Elimina columnas que no son necesarias para el modelado
    jornada.drop(['Noticiero','link', 'Fecha','Descripción'],axis='columns', inplace=True)

    #Elimanos clases que no contienen información en contenido
    jornada = jornada[~jornada['Categoria'].isin(['galeria_imagenes', 'cartones','autos','reportaje','opinion','economia','chomsky','sociedad','ciencia-y-tecnologia','cultura','espectaculos'])]


    razon=pd.read_csv(noticia2)
    razon= razon[~razon['Categoria'].isin(['espectaculos'])]
    # Concatenar los dos DataFrames en uno solo
    df = pd.concat([jornada, razon], ignore_index=True)

    # Guardar el DataFrame resultante en un archivo CSV


    return df