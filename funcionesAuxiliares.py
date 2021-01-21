import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

def mostrarROCCurve(modelo,nombreModelo,X_test,y_test):
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, modelo.predict_proba(X_test)[:, 1])

    zero_test = np.argmin(np.abs(thresholds_test))

    plt.plot(fpr_test, tpr_test, label="ROC Curve "+nombreModelo+" Test")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr_test[zero_test], tpr_test[zero_test], 'o', markersize=10, label="threshold zero test",
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

def escribirPrediccionesAArchivo(predicciones : np.array,nombreModelo):
    archivo = open(nombreModelo+".csv", "w")
    for prediccion in predicciones:
        archivo.write(str(prediccion) + "\n")
    archivo.close()