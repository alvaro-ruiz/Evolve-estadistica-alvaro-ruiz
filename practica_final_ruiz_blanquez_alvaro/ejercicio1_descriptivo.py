"""
Ejercicio 1: Análisis Estadístico Descriptivo
==============================================
Dataset: House Prices – Advanced Regression Techniques (Kaggle)
Target:  SalePrice (variable numérica continua)
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
from scipy.stats import gaussian_kde
import os

# ── Crear carpeta output ──────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

# ── Paleta de colores ─────────────────────────────────────────────────────────
ACCENT  = "#4cc9f0"
ACCENT2 = '#ff7c8c'
ACCENT3 = '#7cffb8'
ACCENT4 = '#ffc77c'
PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT4, '#c07cff', '#7cf0ff']

plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4a',
    'axes.labelcolor':  '#c8cdd8',
    'xtick.color':      '#8a8f9e',
    'ytick.color':      '#8a8f9e',
    'text.color':       '#c8cdd8',
    'grid.color':       '#2a2d3a',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})


# =============================================================================
# A) RESUMEN ESTRUCTURAL
# =============================================================================

def resumen_estructural(df):
    """
    Imprime el resumen estructural del DataFrame y guarda el CSV de nulos.

    Parámetros:
        df (pd.DataFrame): Dataset de entrada.

    Retorna:
        pd.DataFrame: DataFrame sin columnas con >80% de valores nulos.
    """
    print("=== A) Resumen Estructural ===")
    print(f"Filas:    {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Memoria:  {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nTipos de dato:")
    print(df.dtypes.to_string())

    # Nulos antes de limpiar
    nulls = df.isnull().mean() * 100
    print("\nValores nulos antes de eliminar columnas:")
    print(nulls.sort_values(ascending=False).to_string())

    # Eliminar columnas con >80% nulos (PoolQC, MiscFeature, Alley, Fence)
    cols_a_eliminar = nulls[nulls > 80].index
    df = df.drop(columns=cols_a_eliminar)
    print(f"\nColumnas eliminadas (>80% nulos): {list(cols_a_eliminar)}")

    nulls_clean = df.isnull().mean() * 100
    print("\nValores nulos tras limpieza:")
    print(nulls_clean[nulls_clean > 0].sort_values(ascending=False).to_string())

    return df


# =============================================================================
# B) ESTADÍSTICOS DESCRIPTIVOS
# =============================================================================

def estadisticos_descriptivos(df, target):
    """
    Calcula y guarda los estadísticos descriptivos de las variables numéricas.

    Parámetros:
        df     (pd.DataFrame): Dataset limpio.
        target (str):          Nombre de la variable objetivo.

    Retorna:
        list: Lista de nombres de columnas numéricas.
    """
    print("\n=== B) Estadísticos Descriptivos ===")

    numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    # Tabla pandas describe
    descriptivo = df[numericas].describe()
    print(descriptivo)

    # Tabla extendida
    estadisticos = pd.DataFrame({
        "media":      df[numericas].mean(),
        "mediana":    df[numericas].median(),
        "moda":       df[numericas].mode().iloc[0],
        "desviacion": df[numericas].std(),
        "varianza":   df[numericas].var(),
        "min":        df[numericas].min(),
        "max":        df[numericas].max(),
        "q1":         df[numericas].quantile(0.25),
        "q3":         df[numericas].quantile(0.75),
    })
    estadisticos.to_csv("output/ej1_descriptivo.csv")
    print(estadisticos)

    return numericas


# =============================================================================
# C) DISTRIBUCIONES — Histogramas con KDE
# =============================================================================

def generar_histogramas(df, numericas):
    """
    Genera y guarda histogramas con KDE de todas las variables numéricas.

    Parámetros:
        df        (pd.DataFrame): Dataset limpio.
        numericas (list):         Lista de columnas numéricas.

    Retorna:
        None
    """
    print("\nGenerando histogramas...")
    ncols  = 5
    nrows  = int(np.ceil(len(numericas) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 3.4))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle("Distribuciones de Variables Numéricas (Histograma + KDE)",
                 fontsize=16, color='white', fontweight='bold', y=1.01)
    axes_flat = axes.flatten()

    for i, col in enumerate(numericas):
        ax   = axes_flat[i]
        data = df[col].dropna()

        # Histograma normalizado (density=True para superponer KDE)
        ax.hist(data, bins=35, color=ACCENT, edgecolor='#0f1117',
                alpha=0.65, linewidth=0.4, density=True)

        # Líneas de Media y Mediana
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color=ACCENT3, linestyle='dashed', linewidth=1)
        ax.axvline(median_val, color=ACCENT2, linestyle='dotted', linewidth=1)

        # KDE superpuesta
        try:
            kde = gaussian_kde(data)
            xs  = np.linspace(data.min(), data.max(), 300)
            ax.plot(xs, kde(xs), color=ACCENT4, linewidth=1.6)
        except Exception:
            pass

        ax.set_title(col, fontsize=9, color='white', pad=4)
        ax.grid(axis='y', alpha=0.3)
        s = data.skew()
        ax.text(0.97, 0.95, f'skew={s:.2f}', transform=ax.transAxes,
                fontsize=7, color=ACCENT4, ha='right', va='top')

    for j in range(len(numericas), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.savefig("output/ej1_histogramas.png", dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ej1_histogramas.png OK")


# =============================================================================
# C) DISTRIBUCIONES — Boxplots por categórica
# =============================================================================

def generar_boxplots(df, categoricas, target):
    """
    Genera boxplots de la variable objetivo segmentados por cada variable categórica.

    Parámetros:
        df          (pd.DataFrame): Dataset limpio.
        categoricas (list):         Variables categóricas con ≤15 categorías.
        target      (str):          Nombre de la variable objetivo.

    Retorna:
        None
    """
    print("Generando boxplots...")
    cats_plot = [c for c in categoricas if df[c].nunique() <= 15]
    ncols_b   = 3
    nrows_b   = int(np.ceil(len(cats_plot) / ncols_b))

    fig, axes = plt.subplots(nrows_b, ncols_b, figsize=(22, nrows_b * 4))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle(f"Boxplots de {target} por Variable Categórica",
                 fontsize=16, color='white', fontweight='bold', y=1.01)
    axes_flat = axes.flatten()

    for i, cat in enumerate(cats_plot):
        ax    = axes_flat[i]
        order = df.groupby(cat)[target].median().sort_values().index.tolist()
        data_by_cat = [df[df[cat] == g][target].dropna().values for g in order]

        bp = ax.boxplot(data_by_cat, patch_artist=True,
                        medianprops=dict(color=ACCENT4, linewidth=2),
                        whiskerprops=dict(color='#8a8f9e'),
                        capprops=dict(color='#8a8f9e'),
                        flierprops=dict(marker='o', color=ACCENT2,
                                        markersize=2, alpha=0.5))
        extended_palette = PALETTE * 5
        for patch, color in zip(bp['boxes'], extended_palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        labels_with_n = [f"{g}\n(N={len(d)})" for g, d in zip(order, data_by_cat)]
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(labels_with_n, rotation=35, ha='right', fontsize=7.5)
        ax.set_title(cat, fontsize=10, color='white', pad=4)
        ax.set_ylabel(target, fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))

    for j in range(len(cats_plot), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=1.0)
    plt.savefig("output/ej1_boxplots.png", dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ej1_boxplots.png OK")
    return cats_plot


# =============================================================================
# C) OUTLIERS — Método IQR + exportar ej1_outliers.txt
# =============================================================================

def detectar_outliers(df, numericas):
    """
    Detecta outliers mediante el método IQR y guarda el informe en ej1_outliers.txt.

    Justificación del método: IQR es más robusto que Z-score ante distribuciones
    asimétricas.

    Parámetros:
        df        (pd.DataFrame): Dataset limpio.
        numericas (list):         Lista de columnas numéricas.

    Retorna:
        None
    """
    lineas = [
        "=== Detección de Outliers — Método IQR ===",
        "",
        "Justificación: Se usa el método IQR porque SalePrice presenta una asimetría positiva (~1.88)",
        " y IQR es resistente a dicha asimetría.",
        "",
        f"{'Variable':<22} {'Outliers':>8} {'% sobre total':>14} {'Límite inf':>12} {'Límite sup':>12}",
        "-" * 72,
    ]

    for col in numericas:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        mask  = (df[col] < lo) | (df[col] > hi)
        n_out = mask.sum()

        if n_out > 0:
            pct   = n_out / len(df) * 100
            linea = f"{col:<22} {n_out:>8d} {pct:>13.1f}% {lo:>12.1f} {hi:>12.1f}"
            lineas.append(linea)

    with open("output/ej1_outliers.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))


# =============================================================================
# D) VARIABLES CATEGÓRICAS
# =============================================================================

def analisis_categoricas(df, cats_plot):
    """
    Genera gráficos de frecuencia absoluta y relativa de las variables categóricas
    y analiza el desbalance.

    Parámetros:
        df        (pd.DataFrame): Dataset limpio.
        cats_plot (list):         Variables categóricas con ≤15 categorías.

    Retorna:
        None
    """
    print("\nGenerando gráficos categóricos...")
    ncols_c = 3
    nrows_c = int(np.ceil(len(cats_plot) / ncols_c))

    fig, axes = plt.subplots(nrows_c, ncols_c, figsize=(22, nrows_c * 4))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle("Frecuencia Absoluta y Relativa de Variables Categóricas",
                 fontsize=16, color='white', fontweight='bold', y=1.01)
    axes_flat = axes.flatten()

    print("\n=== D) Frecuencias Categóricas ===")
    for i, cat in enumerate(cats_plot):
        ax     = axes_flat[i]
        vc     = df[cat].value_counts()
        rel    = vc / len(df) * 100
        n_cats = len(vc)
        colors = (PALETTE * 5)[:n_cats]

        bars = ax.barh(range(n_cats), vc.values, color=colors,
                       alpha=0.85, edgecolor='#0f1117', linewidth=0.5)
        ax.invert_yaxis()
        ax.set_yticks(range(n_cats))
        ax.set_yticklabels(vc.index, fontsize=7.5)
        ax.set_title(cat, fontsize=10, color='white', pad=4)
        ax.set_xlabel('Frecuencia absoluta', fontsize=8)
        ax.grid(axis='x', alpha=0.3)

        dom = rel.max()
        if dom > 80:
            ax.text(0.95, 0.05, f'⚠ Desbalance: {dom:.0f}%',
                    transform=ax.transAxes, fontsize=8, color=ACCENT2,
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#2a1a1a',
                              ec=ACCENT2, alpha=0.7))

        for bar, pct in zip(bars, rel.values):
            if pct > 3:
                ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                        f'{pct:.1f}%', ha='left', va='center',
                        fontsize=6.5, color='#c8cdd8')

        print(f"\n{cat}:")
        for cat_val, cnt, pct in zip(vc.index, vc.values, rel.values):
            print(f"  {str(cat_val):<20}  abs={cnt:5d}  rel={pct:.1f}%")

    for j in range(len(cats_plot), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=1.0)
    plt.savefig("output/ej1_categoricas.png", dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ej1_categoricas.png OK")


# =============================================================================
# E) CORRELACIONES
# =============================================================================

def analisis_correlaciones(df, numericas, target):
    """
    Genera el heatmap de correlaciones de Pearson, identifica las 3 variables
    más correlacionadas con el target y detecta multicolinealidad.

    Parámetros:
        df        (pd.DataFrame): Dataset limpio.
        numericas (list):         Lista de columnas numéricas.
        target    (str):          Nombre de la variable objetivo.

    Retorna:
        None
    """
    # Ordenar variables por correlación absoluta con el target
    correlaciones_target = df[numericas].corr()[target].abs().sort_values(ascending=False)
    numericas_ordenadas = correlaciones_target.index.tolist()
    corr = df[numericas_ordenadas].corr()

    # Top 3 con target
    top3 = corr[target].drop(target).abs().sort_values(ascending=False).head(3)

    # Multicolinealidad
    found = False
    for i in range(len(numericas_ordenadas)):
        for j in range(i + 1, len(numericas_ordenadas)):
            r = corr.iloc[i, j]
            if abs(r) > 0.9:
                found = True

    # Heatmap
    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    cmap = sns.diverging_palette(240, 10, s=85, l=35, as_cmap=True)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f',
                annot_kws={'size': 6.5, 'color': 'black'},
                linewidths=0.3, linecolor='#0f1117',
                cbar_kws={'shrink': 0.7, 'aspect': 30}, ax=ax)

    ax.set_title("Matriz de Correlaciones de Pearson — Variables Numéricas",
                 fontsize=15, color='white', fontweight='bold', pad=15)
    ax.tick_params(colors='#8a8f9e', labelsize=7.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#8a8f9e', labelsize=8)
    cbar.ax.set_ylabel('Correlación de Pearson', color='#8a8f9e', fontsize=8)

    plt.tight_layout()
    plt.savefig("output/ej1_heatmap_correlacion.png", dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ej1_heatmap_correlacion.png OK")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Cargar dataset
    df = pd.read_csv('data/house_price.csv')
    target = 'SalePrice'

    # A) Resumen estructural
    df = resumen_estructural(df)

    # B) Estadísticos descriptivos
    numericas = estadisticos_descriptivos(df, target)

    # C) Histogramas con KDE
    generar_histogramas(df, numericas)

    # C) Boxplots por categórica
    categoricas = df.select_dtypes(include=['object']).columns.tolist()
    cats_plot   = generar_boxplots(df, categoricas, target)

    # C) Outliers → ej1_outliers.txt
    detectar_outliers(df, numericas)

    # D) Variables categóricas
    analisis_categoricas(df, cats_plot)

    # E) Correlaciones
    analisis_correlaciones(df, numericas, target)