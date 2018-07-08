# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 01:09:05 2017

@author: Sergio
"""

import time

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,AffinityPropagation,SpectralClustering,Birch,MeanShift
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from math import floor
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

#Accede a la base de datos de accidentes
accidentes = pd.read_csv('accidentes_2013.csv')
######################################################################
#El primer caso tendrá en cuenta la iluminación,el total de muertos y heridos en AUTOVIAS.
variablesCasoPrimero = ['TOT_VICTIMAS','TOT_MUERTOS','TOT_HERIDOS_GRAVES','TOT_HERIDOS_LEVES','LUMINOSIDAD','TOT_VEHICULOS_IMPLICADOS']
#Seleccion de los elementos del caso de estudio concreto
accidentesSeleccionados = accidentes[accidentes['ZONA'].str.contains("CARRETERA")]
#accidentesSeleccionados = accidentesSeleccionados[accidentesSeleccionados['ZONA'].str.contains("CARRETERA")]
accidentesSeleccionados = accidentesSeleccionados[accidentesSeleccionados['TIPO_ACCIDENTE'].str.contains("Atropello a animales sueltos")]
#Conjunto de datos para el estudio
data = accidentesSeleccionados[variablesCasoPrimero]
#Se pasa la variable de luminosidad a numérica ordenada, menos iluminación, menos valor
labels = {"LUMINOSIDAD":{"NOCHE: SIN ILUMINACIÓN":0,"NOCHE: ILUMINACIÓN INSUFICIENTE":1,"NOCHE: ILUMINACIÓN SUFICIENTE":2,"CREPÚSCULO":3,"PLENO DÍA":4}}
data.replace(labels,inplace=True)
#Extración de un subconjunto de los datos para los algoritmos
#data = data.sample(3500,random_state=123456)
#Normalización de las variables entre 0 y 1
data_normaliced =  preprocessing.normalize(data, norm='l2')
#######################################################################################
#Una vez que se tienen los datos se crean lo objetos correspondientes a los 5 algoritmos

#K-means
k_means = KMeans(init='k-means++', n_clusters=4, n_init=5)
#Jerárquico
jerarquico = AgglomerativeClustering(n_clusters=11)
#DBSCAN
#↕dbscan = DBSCAN(eps = 0.2 ,min_samples=20) 
meanshift = MeanShift()
#Birch
birch = Birch(n_clusters=11,threshold=0.02)#Bajar el threshold si da algún error
#Espectral
spectral = SpectralClustering(n_clusters=11,random_state=123456)
#Lista de algoritmos
algoritmos = (
		('KMeans',k_means),
		('AgglomerativeClustering',jerarquico),
		#('DBSCAN',dbscan),
		('MeanShift',meanshift),
		('Birch',birch),
		('SpectralClustering',spectral)
		)
cluster_predicts = {} #Aquí se almacenan los resultados
k = {} #Número de clusters de cada algoritmo
silhouette ={}#silhouette score para cada algoritmo
calinski_harabaz={}#calinski-harabaz score para cada algoritmo
times = {}#Tiempos de ejecución
#Ejecuta todos los algoritmos de la lista
for nombre, algoritmo in algoritmos:
	print("Ejecutando "+nombre+"...")
	t = time.time()#inicia cronómetro
	cluster_predicts[nombre] = algoritmo.fit_predict(data_normaliced)#Ejecuta el algoritmo
	t = time.time()-t#Calcula el tiempo de ejecución
	times[nombre] = t#Llena el valor time para el algoritmo
	k[nombre] = len(set(cluster_predicts[nombre]))#Consigue el número de clusters del algoritmo
	#Métrica de silhouette del algoritmo
	silhouette[nombre] = metrics.silhouette_score(data_normaliced,
		    cluster_predicts[nombre], 
			 metric='euclidean', 
			 sample_size=floor(0.1*len(data)), 
			 random_state=123456)
	#Métrica de calinski para el algoritmo
	calinski_harabaz[nombre] = metrics.calinski_harabaz_score(data_normaliced,cluster_predicts[nombre])
#Endfor
"""
#Crea la tabla con todos los datos en un archivo .csv con codificación utf-8 para los caracteres especiales
head = ("Algoritmo","tiempo de ejecución","Número de clusters","Silhouette","Calinski-Harabasz")
listatAlgoritmos = ("KMeans","AgglomerativeClustering", "SpectralClustering", "Birch","MeanShift")
resultados = pd.DataFrame(columns=head)
for alg in listatAlgoritmos:
	resultados = resultados.append({'Algoritmo':alg,
								 "tiempo de ejecución":times[alg],
								 "Número de clusters":k[alg],
								 "Silhouette":silhouette[alg],
								 "Calinski-Harabasz":calinski_harabaz[alg]},ignore_index=True)
resultados.to_csv("caso2_exp2.csv",encoding='utf-8')

#Dibuja el scater matrix de k_means
print("Preparando scater matrix...")
clusters = pd.DataFrame(cluster_predicts['KMeans'],index=data.index,columns=['cluster'])
X_kmeans = pd.concat([data, clusters], axis=1)

import seaborn as sns
sns.set()
variables = list(X_kmeans)
variables.remove('cluster')
sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
sns_plot.savefig("scater_matrix_conjunto_2_Kmeans.png")
print('done')


#Prepara la tabla con la media de cada valor
#Para cada cluster del algoritmo
tabla = pd.DataFrame(columns = variablesCasoPrimero)
for cluster in range(k["KMeans"]):
	clusterEntry = X_kmeans.loc[X_kmeans['cluster'] == cluster] #Tabla filtrada por cluster
	#Para cada variable se crea su media geometrica
	medias={}
	for variable in variablesCasoPrimero:
		medias[variable] = (clusterEntry[variable].mean())
	#EndFor
	#Crea la tabla con la media de cada variable
	tabla = tabla.append({'TOT_VICTIMAS':medias['TOT_VICTIMAS'],
					   'TOT_MUERTOS':medias['TOT_MUERTOS'],
					   'TOT_HERIDOS_GRAVES':medias['TOT_HERIDOS_GRAVES'],
					   'TOT_HERIDOS_LEVES':medias['TOT_HERIDOS_LEVES'],
					   'LUMINOSIDAD':medias['LUMINOSIDAD'],
					   'TOT_VEHICULOS_IMPLICADOS':medias['TOT_VEHICULOS_IMPLICADOS']
					   },ignore_index=True)
#Endfor
#Saca la tabla al csv
tabla.to_csv("KMeans_caso2E1.csv",encoding='utf-8') 

#Visualización del agglomerative
from scipy.cluster import hierarchy
linkage_array = hierarchy.ward(data_normaliced)
plt.figure(1)
plt.clf()
hierarchy.dendrogram(linkage_array,orientation='left') #lo pongo en horizontal para compararlo con el generado por seaborn

#Ahora lo saco usando seaborn (que a su vez usa scipy) para incluir un heatmap
import seaborn as sns
X_filtrado_normal_DF = pd.DataFrame(data_normaliced,index=data.index,columns=variablesCasoPrimero)
sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)
"""