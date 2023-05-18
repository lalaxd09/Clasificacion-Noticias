#ROSALES ONOFRE TANIA
#6BV1
#Inteligencia Artificial
#17/24/2023


""" Programa para concatenar los dos archivos y seleccionar solo 500 de dos categorias que tenia más observaciones que las demás,
asi igual ayudar a crear el archivo para realizar los moleados y comparación de cada uno
#En este caso los datos requeridos son dos csv con distintas noticieros sin importancia de la cantidad de datos que se tenga cada uno
pero si debe tener 3 columnas que tenga "contenido","Titulo", y "Categoria"
#Funciones
concanefiles->Ayuda a concatenar dos archivos csv y eliminación de etiquetas se extrae del archivo concatane_files.py"""
import pandas as pd
import numpy as np
from concatane_files import concatanefiles


def clases(noticia1,noticia2):
    df=concatanefiles(noticia1,noticia2)
    df['Contenido'] = df['Contenido'].replace('[\[\]]', '', regex=True).replace('', np.nan)
    df = df.dropna(subset=['Contenido'], how='any')


    # Seleccionar las primeras 500 filas para las categorías 'estados' y 'politica'
    df_estados = df[df['Categoria'] == 'estados'].head(500)
    df_politica = df[df['Categoria'] == 'politica'].head(500)

    # Unir los dos dataframes en uno solo
    df_estados = pd.concat([df_estados, df_politica])

    # Obtener todos los datos para las categorías sobrantes

    df_sobrantes = df[~df['Categoria'].isin(['estados', 'politica','capital'])]



    df_agrupar = pd.concat([df_estados,df_sobrantes])
    # Agrupar por categoría
    agrupado = df_agrupar.groupby('Categoria')

    # Calcular el conteo de los valores duplicados y guardar el resultado en una nueva columna
    df_agrupar['conteo'] = agrupado['Categoria'].transform('count')

    # Identificar los datos duplicados
    duplicados = df_agrupar.duplicated(subset='Categoria', keep=False)

    # Crear una lista con los valores duplicados y su suma correspondiente
    lista_duplicados = []
    for index, fila in df_agrupar[duplicados].iterrows():
        lista_duplicados.append((fila['Categoria'], fila['conteo']))

    # Obtener los valores duplicados únicos
    lista_duplicados = list(set(lista_duplicados))

    # Convertir la lista en una tabla
    tabla_duplicados = pd.DataFrame(lista_duplicados, columns=['Categoria', 'conteo'])

    # Imprimir la tabla de duplicados
    print(tabla_duplicados)



    df_agrupar.to_csv('noticias_clases.csv',index=False)


clases('noticias_jornada.csv','newsrazon.csv')