# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
Añade aqui tu descripción y analisis:

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset seleccionado proviene de Kaggle, es uno de los dataset que porpusite en el enunciado de la practica, conocido como "House Prices - Advanced Regression Techniques".
La variable objetivo es "SalePrice", ya que tiene setido por ser una variable numerica continua que representa el precio de las viviendas.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Las variables gráficamente presentan desviaciones estándar hacia la derecha. En particular, SalePrice tiene un sesgo positivo extenso (skewness ~1.88) dibujando una cola larga.
> Sí, he encontrado outliers. Dada la fuerte asimetría del dataset, he decidido usar el método analítico del Rango Intercuartílico (IQR).
> Se detectaron numerosos outliers (ej. en LotArea, BsmtFinSF1, y el target SalePrice). He volcado sus conteos al archivo `ej1_outliers.txt`, pero he decidido conscientemente NO borrarlos. En un contexto de negocio inmobiliario, estos no son producto de "desvíos erróneos de sensor", sino mansiones o fincas de alto valor. Quitarlas del registro le restaría a mi modelo futuro toda capacidad matemática de tasar características en el sector residencial de lujo.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Analizando nuestro heatmap y ordenándolo por dependencia a SalePrice, el Ranking Top-3 en base a su similitud de Pearson es:
> 1. **OverallQual** (Calidad general de materiales): r ≈ 0.79
> 2. **GrLivArea** (Espacio habitable en pies cuadrados): r ≈ 0.71
> 3. **GarageCars** (Capacidad de número de coches del garaje): r ≈ 0.64

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Sí, existen un volumen considerable de `NaNs`. Tras escrutarlos mediante la consola (`df.isnull().mean() * 100`), hemos dictado una política para tratar la falta de completitud:
> Hemos eliminado en su totalidad 4 columnas del marco relacional porque tenían un déficit de más del 80%, haciéndolas irrecuperables por imputación: `PoolQC` (~100% nulos), `MiscFeature` (~96%), `Alley` (~93%) y `Fence` (~80%).

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
### Preprocesamiento
Para preparar los datos de cara a los modelos predictivos de Scikit-Learn, efectuamos las siguientes fases:
1. **Limpieza**: Se eliminaron las 4 columnas irrecuperables (más del 80% de nulos). Los valores nulos restantes se imputaron empleando la media para columnas numéricas y la moda para columnas categóricas, asegurando que no existiese pérdida adicional de filas.
2. **Codificación**: Aplicamos `pd.get_dummies(..., drop_first=True)` para transformar las variables categóricas en binarias.
3. **División (Train/Test)**: Los datos fueron divididos para su aprendizaje en un 80% entrenamiento y un 20% testeo, aislando nuestro set de pruebas y fijando su aleatoreidad matemática (`random_state=42`).
4. **Escalado Numérico**: Se usó `StandardScaler` puramente para estandarizar el terreno en magnitudes absolutas (media=0, std=1), forzando a que las columnas objetivamente estadísticamente y no por sus valores unitarios de superficie. 

### Comparativa: Lineal vs Logística
Mientras que la regresión lineal sirve para calcular un precio concreto y continuo de una vivienda, el modelo logístico nos permite clasificar si una casa pertenece o no a un nivel de lujo, con una precisión aproximada.
Ambos modelos llegan a una conclusión parecida. Para que una vivienda se considere de mayor nivel, normalmente necesita materiales de alta calidad y más metros construidos. Estas variables son las que más influyen, como se ve en los coeficientes más altos del gráfico.
La regresión lineal funciona especialmente bien para viviendas de gama media, donde los precios siguen patrones más estables. Sin embargo, cuando aparecen los outliers, el modelo puede verse más afectado.
En cambio, la regresión logística es más estable frente a esos casos extremos, porque no intenta predecir un precio exacto, sino que clasifica las viviendas por rangos o niveles.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> Según se extrae de nuestras mediciones sobre el conjunto final subyacente, los registros de rendimiento son:
- MAE (Error Absoluto Medio): ~20.435$.
- RMSE (Raíz del Error Cuadrático Medio): ~52.308$.
- R²: ~0.6433.
> 
> El modelo funciona bien, pero tiene problemas con los outliers, con las casas baratas y de precio medio funcionan correctamente, pero se pierde por completo con las mansiones. Como casi no tiene ejemplos de casas de lujo para aprender, el error se dispara en cuanto los precios suben a las nubes.


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
Añade aqui tu descripción y analisis:

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> _Escribe aquí tu respuesta_

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       |                |
| β₁        | 2.0       |                |
| β₂        | -1.0      |                |
| β₃        | 0.5       |                |

> _Escribe aquí tu respuesta_

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> _Escribe aquí tu respuesta_

---

## Ejercicio 4 — Series Temporales
---
Añade aqui tu descripción y analisis:

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> _Escribe aquí tu respuesta_

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> _Escribe aquí tu respuesta_

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> _Escribe aquí tu respuesta_

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> _Escribe aquí tu respuesta_

---

*Fin del documento de respuestas*
