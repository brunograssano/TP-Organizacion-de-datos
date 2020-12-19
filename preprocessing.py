import pandas as pd
import numpy as np


def prepararSetDeDatos(info_fiumark_df: pd.DataFrame, usuario_volveria_df: pd.DataFrame):
    """Realiza la limpieza y preparacion investigada durante el TP1"""
    # Trabajo con el target
    usuario_volveria_df['volveria'] = usuario_volveria_df['volveria'].astype(np.int8)
    usuario_volveria_df.drop(columns='id_usuario', inplace=True)

    # Feature Engineering
    info_fiumark_df['edad'] = info_fiumark_df['edad'].apply(np.floor)

    sufijos_y_nombres = info_fiumark_df.nombre.str.split(pat=' ', n=1, expand=True)
    sufijos_y_nombres.columns = ['sufijo', 'nombre']
    info_fiumark_df.drop('nombre', axis=1, inplace=True)
    info_fiumark_df = pd.concat([sufijos_y_nombres, info_fiumark_df], axis='columns')

    # Trabajo con valores nulos
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

    return info_fiumark_df, usuario_volveria_df
