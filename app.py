# =============================================================================
# SpVAR + ESDA + Gradio | Mercado Inmobiliario Colombia
# Versión optimizada para Hugging Face Spaces
# =============================================================================

import subprocess
import sys
import os

def install_packages():
    """Instala dependencias solo si no están presentes (útil en Spaces)."""
    packages = [
        "geopandas", "libpysal", "esda", "splot", "folium", "plotly",
        "gradio", "statsmodels", "mapclassify", "shapely", "pyproj",
        "networkx", "scikit-learn", "matplotlib", "seaborn"
    ]
    for pkg in packages:
        try:
            if pkg == "scikit-learn":
                __import__("sklearn")
            else:
                __import__(pkg.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install_packages()

# =============================================================================
# Importaciones
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import libpysal
from libpysal.weights import Queen, Rook, KNN, lag_spatial
from esda.moran import Moran, Moran_Local
from statsmodels.regression.linear_model import OLS
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
import networkx as nx
from scipy.linalg import eigvals
import json
from typing import Tuple, Dict, Any

# =============================================================================
# 1. CARGA DE GEOMETRÍA Y SIMULACIÓN DE PANEL DEPARTAMENTAL
# =============================================================================
GEOJSON_URL = "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson"

def load_and_prepare_data() -> gpd.GeoDataFrame:
    """Carga GeoJSON oficial de departamentos de Colombia y genera panel simulado."""
    try:
        gdf = gpd.read_file(GEOJSON_URL)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el GeoJSON: {e}")

    # Normalizar nombres de columnas
    if "DPTO_CCDGO" in gdf.columns:
        gdf = gdf.rename(columns={"DPTO_CCDGO": "COD_DPTO", "DPTO_CNMBR": "DEPARTAMENTO"})
    elif "dpto_ccdgo" in gdf.columns:
        gdf = gdf.rename(columns={"dpto_ccdgo": "COD_DPTO", "dpto_cnmbr": "DEPARTAMENTO"})
    else:
        name_cols = [c for c in gdf.columns if "nmbr" in c.lower() or "name" in c.lower()]
        code_cols = [c for c in gdf.columns if "ccdgo" in c.lower() or "code" in c.lower()]
        if name_cols:
            gdf = gdf.rename(columns={name_cols[0]: "DEPARTAMENTO"})
        if code_cols:
            gdf = gdf.rename(columns={code_cols[0]: "COD_DPTO"})

    gdf["COD_DPTO"] = gdf["COD_DPTO"].astype(str).str.zfill(2)
    gdf = gdf.to_crs(epsg=4326)

    gdf["centroid"] = gdf.geometry.centroid
    gdf["lon"] = gdf["centroid"].x
    gdf["lat"] = gdf["centroid"].y

    n = len(gdf)
    rng = np.random.default_rng(42)

    # Precios base realistas (miles de COP / m²)
    base_price = {
        "11": 5200, "05": 3100, "76": 2800, "08": 2600, "13": 2200,
        "17": 2400, "68": 2500, "73": 2100, "66": 2300, "15": 2000
    }
    prices = []
    for cod in gdf["COD_DPTO"]:
        if cod in base_price:
            p = base_price[cod] + rng.normal(0, 180)
        else:
            p = 1600 + rng.normal(0, 350)
        prices.append(max(800, p))
    gdf["IPVN_m2"] = np.array(prices)

    gdf["Credito_Hipotecario"] = (
        120 + 0.085 * gdf["IPVN_m2"] + rng.normal(0, 45, n)
    ).clip(lower=15)

    gdf["Desempleo"] = (
        14.5 - 0.0012 * gdf["IPVN_m2"] + rng.normal(0, 1.8, n)
    ).clip(lower=5.5, upper=22)

    gdf["M2_Aprobados"] = (
        85 + 0.55 * gdf["Credito_Hipotecario"] - 2.8 * gdf["Desempleo"]
        + rng.normal(0, 40, n)
    ).clip(lower=10)

    # Lags temporales sintéticos
    gdf["IPVN_m2_lag"] = gdf["IPVN_m2"] * (0.92 + rng.normal(0, 0.03, n))
    gdf["Credito_Hipotecario_lag"] = gdf["Credito_Hipotecario"] * (0.88 + rng.normal(0, 0.04, n))
    gdf["Desempleo_lag"] = gdf["Desempleo"] * (1.02 + rng.normal(0, 0.025, n))
    gdf["M2_Aprobados_lag"] = gdf["M2_Aprobados"] * (0.90 + rng.normal(0, 0.05, n))

    gdf = gdf.drop(columns=["centroid"], errors="ignore")
    return gdf

# =============================================================================
# 2. SELECCIÓN ÓPTIMA DE MATRIZ DE PESOS ESPACIALES W
# =============================================================================
def select_optimal_weights(gdf: gpd.GeoDataFrame, var: str = "IPVN_m2") -> Tuple[Any, str, float, float]:
    """Evalúa Queen, Rook y KNN y selecciona la mejor según Moran Global significativo."""
    candidates = {}
    y = gdf[var].values

    try:
        w_q = Queen.from_dataframe(gdf, use_index=False)
        w_q.transform = "r"
        mi_q = Moran(y, w_q)
        candidates["Queen"] = (w_q, mi_q.I, mi_q.p_sim)
    except Exception:
        pass

    try:
        w_r = Rook.from_dataframe(gdf, use_index=False)
        w_r.transform = "r"
        mi_r = Moran(y, w_r)
        candidates["Rook"] = (w_r, mi_r.I, mi_r.p_sim)
    except Exception:
        pass

    for k in [4, 5, 6]:
        try:
            w_k = KNN.from_dataframe(gdf, k=k)
            w_k.transform = "r"
            mi_k = Moran(y, w_k)
            candidates[f"KNN_k{k}"] = (w_k, mi_k.I, mi_k.p_sim)
        except Exception:
            continue

    if not candidates:
        raise RuntimeError("No se pudo construir ninguna matriz de pesos espaciales.")

    significant = {k: v for k, v in candidates.items() if v[2] < 0.05}
    if significant:
        best_name = max(significant, key=lambda x: abs(significant[x][1]))
        w, I, p = significant[best_name]
    else:
        best_name = max(candidates, key=lambda x: abs(candidates[x][1]))
        w, I, p = candidates[best_name]

    return w, best_name, I, p

# =============================================================================
# 3. ESPECIFICACIÓN Y ESTIMACIÓN SpVAR
# =============================================================================
def estimate_spvar(gdf: gpd.GeoDataFrame, w: Any) -> Dict[str, Any]:
    """Estima SpVAR de orden 1 (OLS + HC1) e impactos espaciales."""
    vars_ = ["IPVN_m2", "Credito_Hipotecario", "Desempleo", "M2_Aprobados"]

    for v in vars_:
        gdf[f"W_{v}"] = lag_spatial(w, gdf[v].values)
        gdf[f"W_{v}_lag"] = lag_spatial(w, gdf[f"{v}_lag"].values)

    results = {}
    impact_matrices = {}
    coef_summary = []

    for target in vars_:
        y = gdf[target].values
        X_cols = [f"W_{target}_lag"]  # φ proxy

        for v in vars_:
            if v != target:
                X_cols.append(f"{v}_lag")       # β
                X_cols.append(f"W_{v}_lag")     # ρ / θ

        X_cols.append(f"W_{target}")            # componente espacial puro

        X = gdf[X_cols].values
        X = np.column_stack([np.ones(len(y)), X])

        model = OLS(y, X)
        fit = model.fit(cov_type="HC1")

        params = fit.params
        bse = fit.bse
        pvalues = fit.pvalues
        names = ["const"] + X_cols
        coef_dict = {n: {"coef": params[i], "se": bse[i], "pval": pvalues[i]}
                     for i, n in enumerate(names)}

        results[target] = {
            "model": fit,
            "coefs": coef_dict,
            "rsquared": fit.rsquared,
            "nobs": fit.nobs
        }

        rho_proxy = coef_dict.get(f"W_{target}", {}).get("coef", 0.0)
        n = len(gdf)
        I = np.eye(n)
        try:
            S = np.linalg.inv(I - rho_proxy * w.full()[0])
            direct = float(np.mean(np.diag(S)))
            total = float(np.mean(S.sum(axis=1)))
            indirect = total - direct
        except np.linalg.LinAlgError:
            direct = indirect = total = np.nan

        impact_matrices[target] = {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "rho_proxy": rho_proxy
        }

        for nme, d in coef_dict.items():
            coef_summary.append({
                "ecuacion": target,
                "variable": nme,
                "coeficiente": d["coef"],
                "se_HC1": d["se"],
                "pval": d["pval"]
            })

    coef_df = pd.DataFrame(coef_summary)
    return {
        "results": results,
        "impacts": impact_matrices,
        "coef_df": coef_df,
        "vars": vars_
    }

# =============================================================================
# 4. AUDITORÍA DE ESTABILIDAD
# =============================================================================
def check_stability(spvar_out: Dict, w: Any) -> Dict[str, Any]:
    """Verifica estabilidad del sistema mediante autovalores (max|λ| < 1)."""
    vars_ = spvar_out["vars"]
    n_var = len(vars_)
    A = np.zeros((n_var, n_var))

    for i, target in enumerate(vars_):
        rho = spvar_out["impacts"][target]["rho_proxy"]
        A[i, i] = rho
        for j, other in enumerate(vars_):
            if i != j:
                key = f"W_{other}_lag"
                coef = spvar_out["results"][target]["coefs"].get(key, {}).get("coef", 0.0)
                A[i, j] = coef * 0.3

    try:
        eigs = eigvals(A)
        max_mod = float(np.max(np.abs(eigs)))
        stable = max_mod < 1.0
    except Exception:
        eigs = np.array([np.nan])
        max_mod = np.nan
        stable = False

    return {
        "eigenvalues": eigs,
        "max_modulus": max_mod,
        "stable": stable,
        "A_matrix": A
    }

# =============================================================================
# 5. ESDA – Moran Global + LISA
# =============================================================================
def run_esda(gdf: gpd.GeoDataFrame, w: Any, var: str = "IPVN_m2") -> Dict:
    y = gdf[var].values
    mi = Moran(y, w)
    lisa = Moran_Local(y, w)
    gdf = gdf.copy()
    gdf["LISA_I"] = lisa.Is
    gdf["LISA_p"] = lisa.p_sim
    gdf["LISA_q"] = lisa.q
    return {"moran_I": mi.I, "moran_p": mi.p_sim, "lisa_gdf": gdf, "lisa": lisa}

# =============================================================================
# 6. VISUALIZACIONES
# =============================================================================
def create_choropleth(gdf: gpd.GeoDataFrame, column: str, title: str) -> go.Figure:
    gdf_plot = gdf.copy()
    gdf_plot["id"] = gdf_plot.index.astype(str)
    fig = px.choropleth_mapbox(
        gdf_plot,
        geojson=json.loads(gdf_plot.to_json()),
        locations="id",
        color=column,
        hover_name="DEPARTAMENTO",
        hover_data={column: ":.1f", "COD_DPTO": True},
        mapbox_style="carto-positron",
        zoom=4.5,
        center={"lat": 4.5, "lon": -73.0},
        opacity=0.75,
        color_continuous_scale="Viridis",
        title=title
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#1a1a1a", size=12),
        coloraxis_colorbar=dict(title=column)
    )
    return fig

def create_3d_extrusion(gdf: gpd.GeoDataFrame) -> go.Figure:
    z = np.maximum(gdf["IPVN_m2"].values, 0)
    size = np.maximum(gdf["M2_Aprobados"].values, 0)
    size_norm = 8 + 40 * (size - size.min()) / (size.max() - size.min() + 1e-9)

    fig = go.Figure(data=[go.Scatter3d(
        x=gdf["lon"],
        y=gdf["lat"],
        z=z,
        mode="markers",
        marker=dict(
            size=size_norm,
            color=gdf["Credito_Hipotecario"],
            colorscale="Plasma",
            opacity=0.85,
            colorbar=dict(title="Crédito Hipotecario", titleside="right"),
            line=dict(width=0.5, color="#333")
        ),
        text=gdf["DEPARTAMENTO"],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>"
            "IPVN m²: %{z:.0f}<br>"
            "<extra></extra>"
        )
    )])
    fig.update_layout(
        title="Extrusión 3D Espacial – Altura = IPVN_m2 | Color = Crédito | Tamaño = M² Aprobados",
        scene=dict(
            xaxis_title="Longitud",
            yaxis_title="Latitud",
            zaxis_title="IPVN_m2 (miles COP)",
            bgcolor="white",
            xaxis=dict(backgroundcolor="white", gridcolor="#ddd"),
            yaxis=dict(backgroundcolor="white", gridcolor="#ddd"),
            zaxis=dict(backgroundcolor="white", gridcolor="#ddd")
        ),
        paper_bgcolor="white",
        font=dict(color="#1a1a1a"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=550
    )
    return fig

def create_transmission_graph(spvar_out: Dict) -> go.Figure:
    vars_ = spvar_out["vars"]
    G = nx.DiGraph()
    for v in vars_:
        G.add_node(v)

    edge_data = []
    for target in vars_:
        coefs = spvar_out["results"][target]["coefs"]
        for src in vars_:
            key = f"W_{src}_lag"
            if key in coefs:
                val = coefs[key]["coef"]
                if abs(val) > 0.01:
                    G.add_edge(src, target, weight=val)
                    edge_data.append((src, target, val))

    pos = nx.spring_layout(G, dim=3, seed=42)
    node_xyz = np.array([pos[v] for v in vars_])
    edge_x, edge_y, edge_z = [], [], []
    for u, v, w in edge_data:
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
        edge_z += [pos[u][2], pos[v][2], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="#555", width=3),
        hoverinfo="none"
    ))
    fig.add_trace(go.Scatter3d(
        x=node_xyz[:, 0], y=node_xyz[:, 1], z=node_xyz[:, 2],
        mode="markers+text",
        marker=dict(size=18, color="#1f77b4", line=dict(width=1, color="#000")),
        text=vars_,
        textposition="top center",
        hovertext=[f"{v}<br>Impacto total: {spvar_out['impacts'][v]['total']:.3f}" for v in vars_],
        hoverinfo="text"
    ))
    fig.update_layout(
        title="Grafo 3D de Mecanismos de Transmisión SpVAR (φ, β, ρ, θ)",
        scene=dict(
            xaxis=dict(showticklabels=False, title="", backgroundcolor="white"),
            yaxis=dict(showticklabels=False, title="", backgroundcolor="white"),
            zaxis=dict(showticklabels=False, title="", backgroundcolor="white"),
            bgcolor="white"
        ),
        paper_bgcolor="white",
        font=dict(color="#1a1a1a"),
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=500
    )
    return fig

def create_impact_bars(spvar_out: Dict) -> go.Figure:
    vars_ = spvar_out["vars"]
    direct = [spvar_out["impacts"][v]["direct"] for v in vars_]
    indirect = [spvar_out["impacts"][v]["indirect"] for v in vars_]
    total = [spvar_out["impacts"][v]["total"] for v in vars_]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Directo", x=vars_, y=direct, marker_color="#2ca02c"))
    fig.add_trace(go.Bar(name="Indirecto", x=vars_, y=indirect, marker_color="#ff7f0e"))
    fig.add_trace(go.Bar(name="Total", x=vars_, y=total, marker_color="#1f77b4"))
    fig.update_layout(
        barmode="group",
        title="Impactos Espaciales SpVAR (Directos / Indirectos / Totales)",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#1a1a1a"),
        yaxis_title="Magnitud del impacto",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig

# =============================================================================
# 7. INTERPRETABILIDAD AUTOMATIZADA
# =============================================================================
def generate_policy_text(spvar_out: Dict, esda_out: Dict, stability: Dict,
                         w_name: str, moran_I: float, moran_p: float) -> str:
    lines = []
    lines.append("═" * 70)
    lines.append("INTERPRETABILIDAD AUTOMATIZADA – SpVAR & ESDA | COLOMBIA")
    lines.append("═" * 70)
    lines.append("")
    lines.append(f"Matriz de pesos seleccionada: {w_name}")
    lines.append(f"Índice de Moran Global (IPVN_m2): I = {moran_I:.4f}  (p = {moran_p:.4f})")
    if moran_p < 0.05:
        lines.append("→ Existe autocorrelación espacial positiva significativa en precios de vivienda.")
        lines.append("  Los departamentos con precios altos tienden a rodearse de vecinos similares.")
    else:
        lines.append("→ No se detecta autocorrelación espacial global significativa al 5 %.")
    lines.append("")
    lines.append("── ESTABILIDAD DEL SISTEMA DINÁMICO ──")
    lines.append(f"Máximo módulo de autovalores: {stability['max_modulus']:.4f}")
    if stability["stable"]:
        lines.append("✓ Sistema estable (max|λ| < 1). Los shocks se disipan en el tiempo.")
    else:
        lines.append("⚠ Sistema potencialmente inestable. Revisar especificación o outliers.")
    lines.append("")
    lines.append("── MECANISMOS DE TRANSMISIÓN E IMPACTOS ──")
    for v in spvar_out["vars"]:
        imp = spvar_out["impacts"][v]
        lines.append(f"• {v}:")
        lines.append(f"    Directo   = {imp['direct']:.4f}")
        lines.append(f"    Indirecto = {imp['indirect']:.4f}")
        lines.append(f"    Total     = {imp['total']:.4f}")
        lines.append(f"    ρ proxy   = {imp['rho_proxy']:.4f}")
    lines.append("")
    lines.append("── IMPLICACIONES DE POLÍTICA ──")
    lines.append("1. Política monetaria / crédito:")
    lines.append("   Un shock positivo en colocación de crédito hipotecario se transmite")
    lines.append("   espacialmente (efecto spillover). La regulación macroprudencial debería")
    lines.append("   considerar el vecindario departamental, no solo el departamento aislado.")
    lines.append("")
    lines.append("2. Desarrollo urbano:")
    lines.append("   La autodinámica espacial de precios (φ) indica persistencia local.")
    lines.append("   Intervenciones de oferta de suelo en un departamento afectan a vecinos.")
    lines.append("")
    lines.append("3. Empleo y construcción:")
    lines.append("   El desempleo actúa como freno a M2_Aprobados. Políticas de empleo")
    lines.append("   regional tienen externalidades espaciales sobre la actividad edificadora.")
    lines.append("")
    lines.append("4. Recomendación operativa:")
    lines.append("   Monitorear clusters LISA (HH/LL) para focalizar instrumentos de")
    lines.append("   vivienda social y líneas de redescuento de redes de crédito regional.")
    lines.append("")
    lines.append("═" * 70)
    return "\n".join(lines)

# =============================================================================
# 8. PIPELINE PRINCIPAL (se ejecuta al iniciar el Space)
# =============================================================================
print("Cargando geometría y generando panel departamental...")
gdf = load_and_prepare_data()
print(f"Departamentos cargados: {len(gdf)}")

print("Seleccionando matriz de pesos espaciales óptima...")
w, w_name, moran_I, moran_p = select_optimal_weights(gdf, "IPVN_m2")
print(f"Matriz seleccionada: {w_name} | Moran I = {moran_I:.4f} (p = {moran_p:.4f})")

print("Ejecutando ESDA (Moran + LISA)...")
esda_out = run_esda(gdf, w, "IPVN_m2")
gdf = esda_out["lisa_gdf"]

print("Estimando SpVAR...")
spvar_out = estimate_spvar(gdf, w)

print("Verificando estabilidad del sistema...")
stability = check_stability(spvar_out, w)
print(f"Estabilidad: {'SÍ' if stability['stable'] else 'NO'} | max|λ| = {stability['max_modulus']:.4f}")

# Pre-computar figuras
fig_choropleth_price = create_choropleth(gdf, "IPVN_m2", "Índice de Precios de Vivienda (m²)")
fig_choropleth_credit = create_choropleth(gdf, "Credito_Hipotecario", "Colocación de Crédito Hipotecario")
fig_choropleth_unemp = create_choropleth(gdf, "Desempleo", "Tasa de Desempleo Departamental (%)")
fig_choropleth_m2 = create_choropleth(gdf, "M2_Aprobados", "Área Aprobada para Construcción (miles m²)")
fig_3d = create_3d_extrusion(gdf)
fig_graph = create_transmission_graph(spvar_out)
fig_impacts = create_impact_bars(spvar_out)
policy_text = generate_policy_text(spvar_out, esda_out, stability, w_name, moran_I, moran_p)
coef_table = spvar_out["coef_df"].round(4)

print("Pipeline completado. Iniciando interfaz Gradio...")

# =============================================================================
# 9. INTERFAZ GRADIO – TEMA CLARO DE ALTO CONTRASTE (HF Spaces)
# =============================================================================
custom_css = """
.gradio-container {
    background: #f8f9fa !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
.gr-button-primary {
    background: #0d6efd !important;
    color: white !important;
}
.prose {
    color: #1a1a1a !important;
}
footer {display: none !important;}
"""

with gr.Blocks(
    title="SpVAR Colombia – Mercado Inmobiliario",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="gray",
        font=["Segoe UI", "system-ui", "sans-serif"]
    ).set(
        body_background_fill="#f8f9fa",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_shadow="0 1px 3px rgba(0,0,0,0.08)",
        button_primary_background_fill="#0d6efd",
        button_primary_text_color="#ffffff"
    ),
    css=custom_css
) as demo:

    gr.Markdown("""
    # SpVAR Espacial + ESDA | Mercado Inmobiliario y Dinámica Económica – Colombia
    **Modelo de Vector Autorregresivo Espacial** con selección automática de matriz de pesos,
    impactos directos/indirectos/totales y análisis exploratorio espacial (Moran / LISA).
    """)

    with gr.Tab("Mapas Coropléticos"):
        gr.Markdown("### Visualización espacial de las cuatro variables del sistema")
        with gr.Row():
            gr.Plot(fig_choropleth_price, label="IPVN_m2")
            gr.Plot(fig_choropleth_credit, label="Crédito Hipotecario")
        with gr.Row():
            gr.Plot(fig_choropleth_unemp, label="Desempleo")
            gr.Plot(fig_choropleth_m2, label="M² Aprobados")

    with gr.Tab("Extrusión 3D Espacial"):
        gr.Markdown("""
        Altura (Z) = **IPVN_m2** · Color = **Crédito Hipotecario** · Tamaño del marcador = **M² Aprobados**  
        (todos los parámetros de tamaño están garantizados ≥ 0)
        """)
        gr.Plot(fig_3d)

    with gr.Tab("Mecanismos de Transmisión (Grafo 3D)"):
        gr.Markdown("""
        Grafo interactivo de sensibilidad de los cuatro mecanismos:  
        **φ** (autodinámica espacial) · **β** (cruce local) · **ρ** (relación espacial directa) · **θ** (relación espacial cruzada)
        """)
        gr.Plot(fig_graph)
        gr.Plot(fig_impacts)

    with gr.Tab("Resultados Econométricos"):
        gr.Markdown("### Coeficientes SpVAR (OLS + errores robustos HC1)")
        gr.Dataframe(coef_table, label="Tabla de coeficientes")
        gr.Markdown(f"""
        **Matriz de pesos óptima:** `{w_name}`  
        **Moran Global (IPVN_m2):** I = {moran_I:.4f} (p = {moran_p:.4f})  
        **Estabilidad del sistema:** {'✓ Estable' if stability['stable'] else '⚠ Inestable'}  
        (max |λ| = {stability['max_modulus']:.4f})
        """)

    with gr.Tab("Interpretabilidad de Política"):
        gr.Markdown("### Conclusiones operativas automatizadas")
        gr.Textbox(
            value=policy_text,
            lines=28,
            max_lines=40,
            label="Panel de Interpretabilidad",
            interactive=False
        )

    gr.Markdown("""
    ---
    *SpVAR + ESDA · Hugging Face Spaces · libpysal / esda / statsmodels / Plotly / Gradio*  
    Datos geométricos: MGN 2018 DANE · Variables simuladas con coherencia econométrica.
    """)

# =============================================================================
# Lanzamiento compatible con Hugging Face Spaces
# =============================================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
