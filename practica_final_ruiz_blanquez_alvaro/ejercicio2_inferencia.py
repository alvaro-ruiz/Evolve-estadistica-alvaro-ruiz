"""
Ejercicio 2: Inferencia con Scikit-Learn
==============================================
Dataset: House Prices – Advanced Regression Techniques (Kaggle)
Target:  SalePrice (variable numérica continua)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def tratamiento_datos(df):

    # Eliminar columnas con más del 80% de valores nulos
    nulls = df.isnull().mean() * 100
    cols_a_eliminar = nulls[nulls > 80].index
    df = df.drop(columns=cols_a_eliminar)

    # Rellenamos los valores nulos que queden, los numericos con la media y los categoricos con la moda
    numericas = df.select_dtypes(include=[np.number]).columns
    categoricas = df.select_dtypes(include=['object', 'string']).columns
    df[numericas] = df[numericas].fillna(df[numericas].mean())
    df[categoricas] = df[categoricas].fillna(df[categoricas].mode().iloc[0])

    # Codificacion de variables categoricas
    df = pd.get_dummies(df, columns=categoricas, drop_first=True)
    return df

# =============================================================================
# MODELO A - REGRESIÓN LINEAL
# =============================================================================

def modelo_regresion_lineal(X_train, X_test, y_train, y_test):
    
    # 1. Instanciamos y Entrenamos el modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 2. Realizamos predicciones
    y_pred = modelo.predict(X_test)

    # 3. Evaluamos el modelo
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Guardamos las metricas en un archivo
    with open("output/ej2_metricas_regresion.txt", "w") as f:
        f.write("Metricas Regresion Lineal:\n")
        f.write(f"MAE:  {mae:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R2:   {r2:.4f}\n")

    # 4. Grafico Top 10 Coeficientes
    coeficientes = pd.Series(modelo.coef_, index=X.columns)

    # Ordenamos los coeficientes por valor absoluto
    top_10_coef = coeficientes.abs().sort_values(ascending=False).head(10)
    top_10_coef = coeficientes[top_10_coef.index]
    plt.figure(figsize=(10, 6))
    top_10_coef.sort_values().plot(kind='barh', color='skyblue', edgecolor='black')
    plt.title("Top 10 Coeficientes más importantes (Regresión Lineal)")
    plt.xlabel("Peso del coeficiente")
    plt.tight_layout()
    plt.savefig("output/ej2_coeficientes.png")
    plt.close()

    # 5. Grafico de Residuos
    residuos = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicciones")
    plt.ylabel("Residuos")
    plt.title("Residuos vs Predicciones")
    plt.tight_layout()
    plt.savefig("output/ej2_residuos.png")
    plt.close()

# =============================================================================
# MODELO B - REGRESIÓN LOGÍSTICA
# =============================================================================
def  modelo_regresion_logistica(X_train, X_test, y_train, y_test):
    # 1. Instanciamos y Entrenamos el modelo con los parámetros obligatorios
    modelo_log = LogisticRegression(solver='lbfgs', max_iter=1000)
    modelo_log.fit(X_train, y_train)

    # 2. Realizamos predicciones
    y_pred = modelo_log.predict(X_test)

    # 3. Evaluamos el modelo
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Guardamos las metricas en un archivo
    with open("output/ej2_metricas_logistica.txt", "w") as f:
        f.write("Metricas Regresion Logistica:\n")
        f.write(f"Accuracy:  {acc:.2f}\n")
        f.write(f"Precision: {prec:.2f}\n")
        f.write(f"Recall:    {rec:.2f}\n")
        f.write(f"F1:        {f1:.2f}\n")
    
    # 4. Generamos y guardamos la matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=modelo_log.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo_log.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig("output/ej2_matriz_confusion.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df = pd.read_csv("data/house_price.csv")
    df = tratamiento_datos(df)

    # Separar la variable objetivo
    y = df['SalePrice']
    X = df.drop(columns=['SalePrice'])

    # Separar en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelo_regresion_lineal(X_train, X_test, y_train, y_test)
    
    # Dividimos los precios originales en 4 categorías: bajo, medio-bajo, medio-alto, alto
    y_categorica = pd.qcut(y, q=4, labels=['bajo', 'medio-bajo', 'medio-alto', 'alto'])
    
    # Volvemos a dividir los datos pero esta vez usando nuestra nueva 'y_categorica'
    # Reciclamos las X originales para no arrastrar problemas
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_categorica, test_size=0.2, random_state=42)

     # Aplicamos un nuevo escalador exclusivamente para estos datos
    scaler_log = StandardScaler()
    X_train_log = scaler_log.fit_transform(X_train_log)
    X_test_log = scaler_log.transform(X_test_log)
    
    # Llamamos a nuestra nueva función
    modelo_regresion_logistica(X_train_log, X_test_log, y_train_log, y_test_log)