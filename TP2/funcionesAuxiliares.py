import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

def obtenerDatasetDeUrl(url):
    direccion = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    return pd.read_csv(direccion)

def obtenerDatasets():
    y = obtenerDatasetDeUrl('https://drive.google.com/file/d/1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0/view?usp=sharing')
    X = obtenerDatasetDeUrl('https://drive.google.com/file/d/1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq/view?usp=sharing')
    return X, y

def obtenerHoldout():
    return obtenerDatasetDeUrl('https://drive.google.com/file/d/1I980-_K9iOucJO26SG5_M8RELOQ5VB6A/view?usp=sharing')

def mostrarROCCurve(modelo,nombreModelo,X_test, X_train, y_test, y_train):
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, modelo.predict_proba(X_test)[:, 1])
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, modelo.predict_proba(X_train)[:, 1])

    zero_test = np.argmin(np.abs(thresholds_test))
    zero_train = np.argmin(np.abs(thresholds_train))

    plt.plot(fpr_test, tpr_test, label="ROC Curve "+nombreModelo+" Test")
    plt.plot(fpr_train, tpr_train, label="ROC Curve  " + nombreModelo + " Train")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr_test[zero_test], tpr_test[zero_test], 'o', markersize=10, label="threshold zero test",
             fillstyle="none", c="k", mew=2)
    plt.plot(fpr_train[zero_train], tpr_train[zero_train], 'x', markersize=10, label="threshold zero train",
             fillstyle="none", c="k", mew=2)
    plt.legend(loc=4)
    plt.show()


def mostrarMatrizDeConfusion(y_pred,y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fig, ax = plt.subplots(dpi=100)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, vmin=0, yticklabels=["No volveria", "Volveria"],
                xticklabels=["No Volveria", "Volveria"], ax=ax)
    ax.set_title("Matriz de confusion")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

def mostrarAUCScore(modelo,nombreModelo,X_test,y_test):
    auc_score = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
    print("AUC para "+nombreModelo+": {:.3f}".format(auc_score))

def escribirPrediccionesAArchivo(predicciones : np.array,nombreModelo,ids_usuarios):
    archivo = open("PrediccionesHoldout/"+nombreModelo+".csv", "w")
    archivo.write("id_usuario,volveria\n")
    i = 0
    for prediccion in predicciones:
        archivo.write(str(ids_usuarios[i])+ "," + str(prediccion) + "\n")
        i = i + 1
    archivo.close()