import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

################ CONFIGURACIONES INICIALES ################

def prepararSetDeValidacion(usuario_volveria_df: pd.DataFrame):
    """ Trabajo con el target, se lo deja listo para ser usado en los modelos."""

    usuario_volveria_df['volveria'] = usuario_volveria_df['volveria'].astype(np.int8)
    usuario_volveria_df.drop(columns='id_usuario', inplace=True)
    return np.array(usuario_volveria_df).ravel()

def prepararSetDeDatos(info_fiumark_df: pd.DataFrame):
    """ Realiza la limpieza y preparacion investigada durante el TP1 """

    info_fiumark_df = prepararDatosExistentes(info_fiumark_df)
    completarValoresNulos(info_fiumark_df)
    conversionDeTipos(info_fiumark_df)
    info_fiumark_df.drop(columns=['id_usuario', 'nombre', 'id_ticket'], inplace=True)

    return info_fiumark_df


def conversionDeTipos(info_fiumark_df):
    """ Convierte columnas a un tipo de dato mas acorde."""
    info_fiumark_df["tipo_de_sala"] = info_fiumark_df["tipo_de_sala"].astype("category")
    info_fiumark_df["genero"] = info_fiumark_df["genero"].astype("category")
    info_fiumark_df["nombre_sede"] = info_fiumark_df["nombre_sede"].astype("category")
    info_fiumark_df["fila"] = info_fiumark_df["fila"].astype("category")
    info_fiumark_df["sufijo"] = info_fiumark_df["sufijo"].astype("category")


def completarValoresNulos(info_fiumark_df):
    completarSedeDelCine(info_fiumark_df)
    info_fiumark_df['fila'].fillna("No responde", inplace=True)
    completarEdades(info_fiumark_df)


def completarEdades(info_fiumark_df):
    info_fiumark_df['autocompletamos_edad'] = False

    completarMujeres(info_fiumark_df)
    completarHombres(info_fiumark_df)


def completarHombres(info_fiumark_df):
    medianaEdadSenior = info_fiumark_df[info_fiumark_df['sufijo'] == 'Señor'].edad.dropna().median()
    esSenior = info_fiumark_df['sufijo'] == 'Señor'
    sinEdad = info_fiumark_df['edad'].isnull()
    info_fiumark_df.loc[esSenior & sinEdad, 'autocompletamos_edad'] = True
    info_fiumark_df.loc[esSenior, 'edad'] = info_fiumark_df.loc[esSenior, 'edad'].fillna(medianaEdadSenior)


def completarMujeres(info_fiumark_df):
    medianaEdadSenioras = info_fiumark_df[info_fiumark_df['sufijo'] == 'Señora'].edad.dropna().median()
    medianaEdadSenioritas = info_fiumark_df[info_fiumark_df['sufijo'] == 'Señorita'].edad.dropna().median()
    esSeniora = info_fiumark_df['sufijo'] == 'Señora'
    esSeniorita = info_fiumark_df['sufijo'] == 'Señorita'
    sinEdad = info_fiumark_df['edad'].isnull()

    info_fiumark_df.loc[esSeniora & sinEdad, 'autocompletamos_edad'] = True
    info_fiumark_df.loc[esSeniorita & sinEdad, 'autocompletamos_edad'] = True

    info_fiumark_df.loc[esSeniorita, 'edad'] = info_fiumark_df.loc[esSeniorita, 'edad'].fillna(medianaEdadSenioritas)
    info_fiumark_df.loc[esSeniora, 'edad'] = info_fiumark_df.loc[esSeniora, 'edad'].fillna(medianaEdadSenioras)


def completarSedeDelCine(info_fiumark_df):
    sede_mas_frecuente = info_fiumark_df['nombre_sede'].value_counts().index[0]
    info_fiumark_df['nombre_sede'].fillna(sede_mas_frecuente, inplace=True)


def prepararDatosExistentes(info_fiumark_df):
    info_fiumark_df['edad'] = info_fiumark_df['edad'].apply(np.floor)

    sufijos_y_nombres = info_fiumark_df.nombre.str.split(pat=' ', n=1, expand=True)
    sufijos_y_nombres.columns = ['sufijo', 'nombre']
    info_fiumark_df.drop('nombre', axis=1, inplace=True)

    info_fiumark_df = pd.concat([sufijos_y_nombres, info_fiumark_df], axis='columns')
    return info_fiumark_df


def prepararSetDeHoldout(holdout_df: pd.DataFrame):
    """ Prepara el set de holdout remplazando el valor que no se encuentra en el entrenamiento (atras) por un 'No responde'.
        Esto se debe a que no se tiene informacion de este caso."""

    holdout_df = prepararSetDeDatos(holdout_df)
    holdout_df["fila"].replace(to_replace="atras", value="No responde", inplace=True)
    return holdout_df

################ AUXILIARES ################

def codificacionOrdinal(datos_a_codificar):
    encoder = OrdinalEncoder()
    return encoder.fit_transform(datos_a_codificar)

def codificacionOneHot(datos_a_codificar):
    encoder = OneHotEncoder(drop='first', sparse=False)
    datos_codificados = encoder.fit_transform(datos_a_codificar)
    nombres_de_los_features = encoder.get_feature_names(datos_a_codificar.columns)
    return datos_codificados, nombres_de_los_features

def normalizar(datos_juntos):
    return (datos_juntos - datos_juntos.mean()) / datos_juntos.std()

def estratificar_edades(edad):
    if(edad <= 10):
        return 'ninio'
    elif(edad <=25):
        return 'joven'
    elif(edad <= 50):
        return 'adulto'
    return 'mayor'

def estratificar_precios(total_pagado):
    if total_pagado <= 1:
        return 'Pago poco'
    elif total_pagado <= 6:
        return 'Pago normal'
    elif total_pagado <= 12:
        return 'Pago mucho'
    return 'Pago demasiado'


def categorizar_invitados(cantidad_invitados):
    if cantidad_invitados == 0:
        return 'Fue solo'
    return 'Fue en grupo'

################ PREPROCESSING ################

def expansionDelDataset(info_fiumark_df: pd.DataFrame):
    """ Creara varias nuevas features a partir del dataframe original. Este dataframe debe de venir preproceado previamente por
        prepararSetDeDatos. """

    nombres, fiumark_numerico = conversionAVariablesNumericas(info_fiumark_df)
    info_fiumark_df['2_clusters'] = KMeans(n_clusters=2, random_state=0).fit_predict(fiumark_numerico)
    info_fiumark_df['4_clusters'] = KMeans(n_clusters=4, random_state=0).fit_predict(fiumark_numerico)
    info_fiumark_df['10_clusters'] = KMeans(n_clusters=10, random_state=0).fit_predict(fiumark_numerico)

    info_fiumark_df['cantidad_total_invitados'] = info_fiumark_df['parientes'] + info_fiumark_df['amigos']

    info_fiumark_df['total_pagado'] = (info_fiumark_df['cantidad_total_invitados'] + 1 ) * info_fiumark_df['precio_ticket']

    info_fiumark_df['pago_categorizado'] = info_fiumark_df['total_pagado'].apply(estratificar_precios)

    info_fiumark_df['edades_estratificadas'] = info_fiumark_df['edad'].apply(estratificar_edades)

    info_fiumark_df['categoria_invitados'] = info_fiumark_df['cantidad_total_invitados'].apply(categorizar_invitados)

    return info_fiumark_df



def conversionAVariablesNormalizadas(fiumark_procesado_df,columnas_codificables_extra = [],columnas_numericas_extra = []):
    """ Necesita que el dataset fiumark ya venga preprocesado por la funcion prepararSetDeDatos (aplica las transformaciones del TP1)
        Convertira todas las variables a valores numericos para que puedan ser usados en los modelos. Una vez hecho eso, los normalizara.
        Devuelve solamente los datos. Si se quieren los nombres  de los features se deben de llamar separadas las funciones."""

    nombres_de_los_features, datos_juntos = conversionAVariablesNumericas(fiumark_procesado_df,columnas_codificables_extra,columnas_numericas_extra)
    return normalizar(datos_juntos)

def conversionAVariablesNumericas(fiumark_procesado_df,columnas_codificables_extra = [],columnas_numericas_extra = []):
    """ Necesita que el dataset fiumark ya venga preprocesado por la funcion prepararSetDeDatos (aplica las transformaciones del TP1)
        Si se cumple la precondicion, se encargara de dejar de forma numerica todas las features.
        Los valores categoricos seran convertidos mediante OneHotEncoding.
        Al momento de devolverlos, lo hace junto a los nombres de los features, por si se quiere recuperar luego el DataFrame."""

    columnas_a_codificar = ['sufijo', 'tipo_de_sala', 'genero', 'autocompletamos_edad', 'fila', 'nombre_sede'] + columnas_codificables_extra
    datos_a_codificar = fiumark_procesado_df[columnas_a_codificar]
    datos_codificados, nombres_de_los_features_codificados = codificacionOneHot(datos_a_codificar)

    columnas_numericas = ['edad', 'amigos', 'parientes', 'precio_ticket'] + columnas_numericas_extra
    datos_numericos = fiumark_procesado_df[columnas_numericas]
    datos_juntos = np.hstack((np.array(datos_numericos), datos_codificados))
    nombres_de_los_features = columnas_numericas + nombres_de_los_features_codificados.tolist()
    return nombres_de_los_features, datos_juntos

def conversionAVariablesCodificadas(fiumark_procesado_df: pd.DataFrame, columnasConLasQueNosQuedamos = [] ):
    """ Necesita que el dataset fiumark ya venga preprocesado por la funcion prepararSetDeDatos (aplica las transformaciones del TP1).
        Si se cumple, la funcion se encargara de dejar solamente las variables indicadas. Estas se convierten mediante una codificacion ordinal."""

    df_procesado = fiumark_procesado_df[columnasConLasQueNosQuedamos]

    return codificacionOrdinal(df_procesado)

def conversionAVariablesContinuas(fiumark_procesado_df: pd.DataFrame):
    """ Necesita que el dataset fiumark ya venga preprocesado por la funcion prepararSetDeDatos (aplica las transformaciones del TP1).
        Si se cumple, la funcion se encargara de dejar solamente las variables continuas. Estas seran devueltas en un array."""

    df_procesado = fiumark_procesado_df[['edad','precio_ticket','autocompletamos_edad']]

    return np.array(df_procesado)