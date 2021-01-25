# TP2: Enunciado

## Introducción
Luego de la presentación del informe y el baseline Fiumark quiere profundizar su campaña de marketing. Gracias al éxito 
logrado en la primera campaña la empresa tiene más confianza en ustedes y sus “algoritmos” y está ansiosa por probar 
las avanzadas técnicas de inteligencia artificial de las que todo el mundo habla.  


## Tarea
La directora de marketing de Fiumark está muy interesada en utilizar algoritmos de machine learning desde que escuchó 
que su más grande competidor Fiubaplex lo utiliza para dirigir sus campañas de marketing, por esto es que hizo un curso 
para entender cómo funciona. Con su conocimiento al respecto exige que probemos varios modelos reportando cual de todos 
fue el mejor (según la métrica AUC-ROC), también pretende que utilicemos técnicas para buscar la mejor configuración de 
hiperparámetros, que intentemos hacer al menos un ensamble, que utilicemos cross-validation para comparar los modelos y 
que presentemos varias métricas del modelo final:  
- AUC-ROC
- Matriz de confusión
- Accuracy
- Precisión
- Recall

La directora también sabe los dilemas que resultan de llevar un prototipo a producción, por lo que nos pidió 
encarecidamente que dejemos muy explícitos los pasos de pre-procesamiento/feature engineering que usamos en cada 
modelo, y que dejemos toda la lógica del preprocesado en un archivo python llamado preprocesing.py en donde van a 
estar todas las funciones utilizadas para preprocesamiento, claro está, ella espera que si dos modelos tienen el 
mismo preprocesado entonces usen la misma función en preprocessing.py.  

## Entrega
El formato de entrega va a ser un breve informe en PDF con:

**(TABLA 1)** Tabla que liste todos los pre-procesamientos utilizados, con el estilo:  
\<nombre preprocesamiento\> \< explicación simple\> \< nombre de la función de python \>  
En donde:  
- **nombre preprocesamiento:** es un nombre que ustedes elijan para representar lo que hace el preprocesado.
- **explicación simple:** es una descripción en no más de 2 líneas de la lógica de preprocesado.
- **nombre de la función de python:** nombre de la función de python que va a estar localizada en preprocessing.py


**(TABLA 2)** Tabla que liste:  
\<Nombre Modelo\> \<nombre preprocesamiento\>  \<AUC-ROC\> \<Accuracy\> \<Precision\> \<Recall\> \<F1 score\>  
En donde :  
- **Nombre Modelo:** es el mejor modelo de los de su mismo tipo, enumerados en orden secuencial que se fueron realizando (\<número\> - \<nombre\>).
- **nombre preprocesamiento:** es el nombre del preprocesamiento (tiene que estar presente en la tabla anterior)
- (el resto de las columnas son las métricas de ese Nombre Modelo)

En concordancia con lo anterior, se espera que cada Nombre Modelo este en un notebook separado con el nombre
 \<Nombre Modelo\>.ipynb y que dentro del mismo esté de forma clara la llamada al preprocesado, su entrenamiento, 
 la evaluación del mismo y finalmente una predicción en formato csv de un archivo nuevo localizado
  en: https://drive.google.com/file/d/1I980-_K9iOucJO26SG5_M8RELOQ5VB6A/view?usp=sharing La directora nos pide que 
  por cada modelo listado en la tabla, hagamos las predicciones de este archivo y en la entrega junto con los notebook 
  también entreguemos todas las predicciones. El nombre del archivo con las predicciones tiene que 
  ser \<Nombre Modelo\>.csv, ella tiene pensado chequear las métricas de estas predicciones minutos antes de 
  pagarnos ( esperemos que no nos cague  👀 ).

**(CONCLUSIÓN)**
Finalmente luego de poner las tablas TABLA 1 y TABLA 2, nos piden que lleguemos a una conclusión sobre qué modelo 
recomendamos y por qué y que lo comparemos con respecto al baseline que anteriormente implementamos.


## Notas Técnicas:
- El formato esperado para las predicciones realizadas en cada .csv es igual al del archivo que ya utilizan en el 
entrenamiento : https://drive.google.com/file/d/1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0/view?usp=sharing en donde por cada 
línea del archivo se tiene:  
\<id usuario\> \<volvería\>  
- Todos los notebooks deben poder ser ejecutados de principio a fin por su corrector produciendo los mismos resultados
- Todas las dependencias de librerías deben estar en un requirements.txt
- La entrega se tiene que realizar en un .zip al mail orga.datos.fiuba@gmail.com

## Fecha de entrega:
- Entrega: 16 de Febrero
- Defensa oral del tp: 23 de Febrero  

Nota: entre la entrega y la defensa oral se pueden llegar a pedir correcciones para antes de la defensa oral de 
considerarse necesarias.
