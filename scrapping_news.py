#ROSALES ONOFRE TANIA
#6BV1
#Inteligencia Artificial
#17/24/2023


""" Programa para  realizar web scrapping he ir actualizando más observaciones en el conjunto de datos"""




import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://www.jornada.com.mx/feeds/rss'

resp = requests.get(url)
soup =BeautifulSoup(resp.content ,features ='xml')
print(soup.prettify())

itemss = soup.findAll('item')
len(itemss)





news_items=[]
for item in itemss:
  news_item={}
  news_item['Noticiero']='La Jornada'
  news_item['Titulo'] = item.title.text
  news_item['Categoria'] = item.category.text
  news_item['Descripción'] = item.description.text
  news_item['link']=item.link.text
  #news_item['Autor'] = item.creator.text
  news_item['Fecha'] = item.pubDate.text

  news_items.append(news_item)

print(news_items)



df = pd.DataFrame(news_items)
print(df)




Titulos_noticias = []
Contenido = []

for noticia_link in news_items:
    url_news= noticia_link.get('link')
    #url_news = 'https://www.jornada.com.mx/notas/2023/02/22/politica/llama-amlo-a-resolver-diferencias-en-nicaragua-mediante-dialogo/'
    page_noticia = requests.get(url_news, headers={'User-Agent': 'Mozilla/5.0'})
    soup_noticia = BeautifulSoup(page_noticia.content, "html.parser")
    for i in soup_noticia.find('h2', attrs={'id': 'article-title-tts'}):
        try:
            titulo = i
        except:
            titulo = 'none'
        Titulos_noticias.append(str(titulo))

    for i in soup_noticia.findAll("div", class_="article-content ljn-nota-contenido"):
        try:
            content = i.find_all('p')
        except:
            content = 'none'
        Contenido.append(str(content))

#print(Titulos_noticias)
#print(Contenido)
#df.insert(2, "Titles", Titulos_noticias, allow_duplicates=False)
df.insert(3,'Contenido',Contenido, allow_duplicates=False)


#df.to_csv('noticias_jornada.csv', index=False)


df.to_csv('C:/Users/Tania/Desktop/Septimo Semestre/Aplicaciones de lenguaje natural/Practica1/noticias_jornada.csv', mode='a', header=False, index=False , encoding="utf-8")

print('Se agregaron los nuevos datos al archivo CSV')




