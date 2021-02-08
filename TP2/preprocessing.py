import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

################ CONFIGURACIONES INICIALES ################

def prepararSetDeValidacion(usuario_volveria_df: pd.DataFrame):
    """ Trabajo con el target, se lo deja listo para ser usado en los modelos."""

    usuario_volveria_df['volveria'] = usuario_volveria_df['volveria'].astype(np.int8)
    usuario_volveria_df.drop(columns='id_usuario', inplace=True)
    return np.array(usuario_volveria_df).ravel()

def prepararSetDeDatos(info_fiumark_df: pd.DataFrame):
    """ Realiza la limpieza y preparacion investigada durante el TP1 """

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

    info_fiumark_df.drop(columns=['id_usuario', 'nombre', 'id_ticket'], inplace=True)

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

################ PREPROCESSING ################

def conversionAVariablesNumericasNormalizadas(fiumark_procesado_df):
    """ Necesita que el dataset fiumark ya venga preprocesado por la funcion prepararSetDeDatos (aplica las transformaciones del TP1)
        Convertira todas las variables a valores numericos para que puedan ser usados en los modelos. Una vez hecho eso, los normalizara."""

    nombres_de_los_features, datos_juntos = conversionAVariablesNumericas(fiumark_procesado_df)
    return normalizar(datos_juntos)

def conversionAVariablesNumericas(fiumark_procesado_df):
    """ Necesita que el dataset fiumark ya venga preprocesado por la funcion prepararSetDeDatos (aplica las transformaciones del TP1)
        Si se cumple la precondicion, se encargara de dejar de forma numerica todas las features.
        Los valores categoricos seran convertidos mediante OneHotEncoding.
        Al momento de devolverlos, lo hace junto a los nombres de los features, por si se quiere recuperar luego el DataFrame."""

    datos_a_codificar = fiumark_procesado_df[['sufijo', 'tipo_de_sala', 'genero', 'autocompletamos_edad', 'fila', 'nombre_sede']]
    datos_codificados, nombres_de_los_features_codificados = codificacionOneHot(datos_a_codificar)
    datos_numericos = fiumark_procesado_df[['edad', 'amigos', 'parientes', 'precio_ticket']]
    datos_juntos = np.hstack((np.array(datos_numericos), datos_codificados))
    nombres_de_los_features = ['edad', 'amigos', 'parientes', 'precio_ticket'] + nombres_de_los_features_codificados.tolist()
    return nombres_de_los_features, datos_juntos

def conversionAVariablesCategoricasCodificadas(fiumark_procesado_df: pd.DataFrame):
    """ Necesita que el dataset fiumark ya venga preprocesado por la funcion prepararSetDeDatos (aplica las transformaciones del TP1).
        Si se cumple, la funcion se encargara de dejar solamente las variables categoricas. Estas se convierten mediante una codificacion ordinal."""

    df_procesado = fiumark_procesado_df.drop(columns=['edad', 'precio_ticket', 'autocompletamos_edad'])
    # Sacamos estas columnas, ya que no tiene sentido calcular la probabilidad de cada una de ellas.
    # Por ejemplo, cual es la probabilidad de que tu edad sea X.

    return codificacionOrdinal(df_procesado)

def conversionAVariablesDiscretasCodificadas(fiumark_procesado_df: pd.DataFrame):
    """ Necesita que el dataset fiumark ya venga preprocesado por la funcion prepararSetDeDatos (aplica las transformaciones del TP1).
        Si se cumple, la funcion se encargara de dejar solamente las variables discretas. Estas se convierten mediante una codificacion ordinal."""

    df_procesado = fiumark_procesado_df.drop(columns=['edad', 'autocompletamos_edad'])

    return codificacionOrdinal(df_procesado)

def conversionAVariablesContinuas(fiumark_procesado_df: pd.DataFrame):
    """ Necesita que el dataset fiumark ya venga preprocesado por la funcion prepararSetDeDatos (aplica las transformaciones del TP1).
        Si se cumple, la funcion se encargara de dejar solamente las variables continuas. Estas seran devueltas en un array."""

    df_procesado = fiumark_procesado_df[['edad','precio_ticket','autocompletamos_edad']]

    return np.array(df_procesado)