import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

def prepararSetDeValidacion(usuario_volveria_df: pd.DataFrame):
    # Trabajo con el target
    usuario_volveria_df['volveria'] = usuario_volveria_df['volveria'].astype(np.int8)
    usuario_volveria_df.drop(columns='id_usuario', inplace=True)
    return usuario_volveria_df

def prepararSetDeDatos(info_fiumark_df: pd.DataFrame):
    """Realiza la limpieza y preparacion investigada durante el TP1"""

    # Feature Engineering
    info_fiumark_df['edad'] = info_fiumark_df['edad'].apply(np.floor)

    sufijos_y_nombres = info_fiumark_df.nombre.str.split(pat=' ', n=1, expand=True)
    sufijos_y_nombres.columns = ['sufijo', 'nombre']
    info_fiumark_df.drop('nombre', axis=1, inplace=True)
    info_fiumark_df = pd.concat([sufijos_y_nombres, info_fiumark_df], axis='columns')

    # Trabajo con valores nulos
    sede_mas_frecuente = info_fiumark_df['nombre_sede'].value_counts().index[0]
    info_fiumark_df['nombre_sede'].fillna(sede_mas_frecuente, inplace=True)

    info_fiumark_df['fila'].fillna("No responde", inplace=True)

    mediana_edad_senioras = info_fiumark_df[info_fiumark_df['sufijo'] == 'Señora'].edad.dropna().median()
    mediana_edad_senioritas = info_fiumark_df[info_fiumark_df['sufijo'] == 'Señorita'].edad.dropna().median()

    info_fiumark_df['autocompletamos_edad'] = False

    info_fiumark_df.loc[
        (info_fiumark_df['sufijo'] == 'Señora') & (info_fiumark_df['edad'].isnull()), 'autocompletamos_edad'] = True
    info_fiumark_df.loc[
        (info_fiumark_df['sufijo'] == 'Señorita') & (info_fiumark_df['edad'].isnull()), 'autocompletamos_edad'] = True
    info_fiumark_df.loc[info_fiumark_df['sufijo'] == 'Señorita', 'edad'] = info_fiumark_df.loc[
        info_fiumark_df['sufijo'] == 'Señorita', 'edad'].fillna(mediana_edad_senioritas)
    info_fiumark_df.loc[info_fiumark_df['sufijo'] == 'Señora', 'edad'] = info_fiumark_df.loc[
        info_fiumark_df['sufijo'] == 'Señora', 'edad'].fillna(mediana_edad_senioras)

    mediana_edad_senior = info_fiumark_df[info_fiumark_df['sufijo'] == 'Señor'].edad.dropna().median()

    info_fiumark_df.loc[
        (info_fiumark_df['sufijo'] == 'Señor') & (info_fiumark_df['edad'].isnull()), 'autocompletamos_edad'] = True
    info_fiumark_df.loc[info_fiumark_df['sufijo'] == 'Señor', 'edad'] = info_fiumark_df.loc[
        info_fiumark_df['sufijo'] == 'Señor', 'edad'].fillna(mediana_edad_senior)

    # Conversion de valores para ahorrar memoria
    info_fiumark_df["tipo_de_sala"] = info_fiumark_df["tipo_de_sala"].astype("category")
    info_fiumark_df["genero"] = info_fiumark_df["genero"].astype("category")
    info_fiumark_df["nombre_sede"] = info_fiumark_df["nombre_sede"].astype("category")
    info_fiumark_df["fila"] = info_fiumark_df["fila"].astype("category")
    info_fiumark_df["sufijo"] = info_fiumark_df["sufijo"].astype("category")

    info_fiumark_df.drop(columns=['id_usuario'], inplace=True)

    return info_fiumark_df


################ AUXILIARES ################

def codificacionOrdinal(datos_a_codificar):
    encoder = OrdinalEncoder() # TODO USAR FIT TRANSFORM
    return encoder.fit_transform(datos_a_codificar)

def codificacionOneHot(datos_a_codificar):
    encoder = OneHotEncoder(drop='first', sparse=False)
    return encoder.fit_transform(datos_a_codificar)

def normalizar(datos_juntos):
    return (datos_juntos - datos_juntos.mean()) / datos_juntos.std()

def conversionAVariablesNumericasNormalizadas(fiumark_procesado_df):
    df_procesado = fiumark_procesado_df.drop(columns=['nombre'])
    datos_a_codificar = df_procesado[['sufijo', 'tipo_de_sala', 'genero', 'autocompletamos_edad', 'fila', 'nombre_sede']]
    datos_codificados = codificacionOneHot(datos_a_codificar)
    datos_numericos = df_procesado[['edad', 'amigos', 'parientes', 'precio_ticket']]
    datos_juntos = np.hstack((np.array(datos_numericos), datos_codificados))
    return normalizar(datos_juntos)

def conversionAVariablesNumericas(fiumark_procesado_df):
    df_procesado = fiumark_procesado_df.drop(columns=['nombre'])
    datos_a_codificar = df_procesado[['sufijo', 'tipo_de_sala', 'genero', 'autocompletamos_edad', 'fila', 'nombre_sede']]
    datos_codificados = codificacionOneHot(datos_a_codificar)
    datos_numericos = df_procesado[['edad', 'amigos', 'parientes', 'precio_ticket']]
    datos_juntos = np.hstack((np.array(datos_numericos), datos_codificados))
    return datos_juntos

################ PREPROCESSING ################

def categoricalNBPreprocessing(fiumark_procesado_df: pd.DataFrame):
    """Preparara y dejara listo para usar un dataframe en el modelo de NB categorico.
    Necesita que venga ya preprocesado anteriormente por la funcion del TP1"""

    df_procesado = fiumark_procesado_df.drop(columns=['nombre', 'edad', 'precio_ticket', 'id_ticket', 'autocompletamos_edad'])
    # Sacamos estas columnas, ya que no tiene sentido calcular la probabilidad de cada una de ellas.
    # Por ejemplo, cual es la probabilidad de que te llames de tal forma.

    return codificacionOrdinal(df_procesado)

def multinomialNBPreprocessing(fiumark_procesado_df: pd.DataFrame):
    """Preparara y dejara listo para usar un dataframe en el modelo de NB multinomial.
    Necesita que venga ya preprocesado anteriormente por la funcion del TP1"""
    df_procesado = fiumark_procesado_df.drop(columns=['nombre', 'edad', 'id_ticket', 'autocompletamos_edad'])

    return codificacionOrdinal(df_procesado)

def gaussianNBPreprocessing(fiumark_procesado_df: pd.DataFrame):
    """Preparara y dejara listo para usar un dataframe en el modelo de NB gaussiano.
    Necesita que venga ya preprocesado anteriormente por la funcion del TP1"""
    df_procesado = fiumark_procesado_df[['edad','precio_ticket','autocompletamos_edad']]

    return np.array(df_procesado)

def arbolDeDecisionPreprocessing(fiumark_procesado_df: pd.DataFrame):
    """Preparara y dejara listo para usar un dataframe en el modelo de arbol de decision.
        Necesita que venga ya preprocesado anteriormente por la funcion del TP1"""
    df_procesado = fiumark_procesado_df.drop(columns=['id_ticket'])

    return conversionAVariablesNumericas(df_procesado)

def knnPreprocessing(fiumark_procesado_df: pd.DataFrame):
    """Preparara y dejara listo para usar un dataframe en el modelo de KNN.
    Necesita que venga ya preprocesado anteriormente por la funcion del TP1"""
    return conversionAVariablesNumericasNormalizadas(fiumark_procesado_df)

def svmPreprocessing(fiumark_procesado_df: pd.DataFrame):
    """Preparara y dejara listo para usar un dataframe en el modelo de SVM.
    Necesita que venga ya preprocesado anteriormente por la funcion del TP1"""
    return conversionAVariablesNumericasNormalizadas(fiumark_procesado_df)


def redesNeuronalesPreprocessing(fiumark_procesado_df: pd.DataFrame):
    """Preparara y dejara listo para usar un dataframe en los modelos de redes neuronales.
    Necesita que venga ya preprocesado anteriormente por la funcion del TP1"""
    return conversionAVariablesNumericasNormalizadas(fiumark_procesado_df)

def rlPreprocessing(fiumark_procesado_df: pd.DataFrame):
    """Preparara y dejara listo para usar un dataframe en el modelo de Regresion Logistica.
    Necesita que venga ya preprocesado anteriormente por la funcion del TP1"""
    return conversionAVariablesNumericasNormalizadas(fiumark_procesado_df)

def rfPreprocessing(fiumark_procesado_df: pd.DataFrame):
    """Preparara y dejara listo para usar un dataframe en el modelo de Random Forest.
    Necesita que venga ya preprocesado anteriormente por la funcion del TP1"""
    return conversionAVariablesNumericasNormalizadas(fiumark_procesado_df)
