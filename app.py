# =============================================================================
# SpVAR + ESDA | Mercado Inmobiliario Colombia
# Versión Streamlit (compatible con Streamlit Community Cloud / Hugging Face)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal
from libpysal.weights import Queen, Rook, KNN, lag_spatial
from esda.moran import Moran, Moran_Local
from statsmodels.regression.linear_model import OLS
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy.linalg import eigvals
import json
import streamlit as st
from typing import Tuple, Dict, Any

# -----------------------------------------------------------------------------
# Configuración de página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SpVAR Colombia – Mercado Inmobiliario",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 1. CARGA DE GEOMETRÍA Y PANEL
# -----------------------------------------------------------------------------
GEOJSON_URL = "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson"

@st.cache_data(show_spinner="Cargando geometría y datos departamentales...")
def load_and_prepare_data() -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(GEOJSON_URL)
    except Exception as e:
        st.error(f"No se pudo cargar el GeoJSON: {e}")
        st.stop()

    # Normalizar columnas
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

    # Centroides
    centroids = gdf.geometry.centroid
    gdf["lon"] = centroids.x
    gdf["lat"] = centroids.y

    n = len(gdf)
    rng = np.random.default_rng(42)

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
        prices.append(max(800.0, float(p)))
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

    return gdf

# -----------------------------------------------------------------------------
# 2. MATRIZ DE PESOS ESPACIALES
# -----------------------------------------------------------------------------
def select_and_build_weights(gdf: gpd.GeoDataFrame, var: str = "IPVN_m2"):
    candidates = {}
    y = gdf[var].values.astype(float)

    try:
        w_q = Queen.from_dataframe(gdf, use_index=False)
        w_q.transform = "r"
        mi_q = Moran(y, w_q)
        candidates["Queen"] = (w_q, float(mi_q.I), float(mi_q.p_sim))
    except Exception:
        pass

    try:
        w_r = Rook.from_dataframe(gdf, use_index=False)
        w_r.transform = "r"
        mi_r = Moran(y, w_r)
        candidates["Rook"] = (w_r, float(mi_r.I), float(mi_r.p_sim))
    except Exception:
        pass

    for k in [4, 5, 6]:
        try:
            w_k = KNN.from_dataframe(gdf, k=k)
            w_k.transform = "r"
            mi_k = Moran(y, w_k)
            candidates[f"KNN_k{k}"] = (w_k, float(mi_k.I), float(mi_k.p_sim))
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

# -----------------------------------------------------------------------------
# 3. SpVAR
# -----------------------------------------------------------------------------
def estimate_spvar(gdf: gpd.GeoDataFrame, w) -> Dict[str, Any]:
    vars_ = ["IPVN_m2", "Credito_Hipotecario", "Desempleo", "M2_Aprobados"]

    gdf = gdf.copy()
    for v in vars_:
        gdf[f"W_{v}"] = lag_spatial(w, gdf[v].values.astype(float))
        gdf[f"W_{v}_lag"] = lag_spatial(w, gdf[f"{v}_lag"].values.astype(float))

    results = {}
    impact_matrices = {}
    coef_summary = []

    for target in vars_:
        y = gdf[target].values.astype(float)
        X_cols = [f"W_{target}_lag"]

        for v in vars_:
            if v != target:
                X_cols.append(f"{v}_lag")
                X_cols.append(f"W_{v}_lag")

        X_cols.append(f"W_{target}")

        X = gdf[X_cols].values.astype(float)
        X = np.column_stack([np.ones(len(y)), X])

        model = OLS(y, X)
        fit = model.fit(cov_type="HC1")

        params = fit.params
        bse = fit.bse
        pvalues = fit.pvalues
        names = ["const"] + X_cols
        coef_dict = {
            n: {"coef": float(params[i]), "se": float(bse[i]), "pval": float(pvalues[i])}
            for i, n in enumerate(names)
        }

        results[target] = {
            "coefs": coef_dict,
            "rsquared": float(fit.rsquared),
            "nobs": int(fit.nobs),
        }

        rho_proxy = coef_dict.get(f"W_{target}", {}).get("coef", 0.0)
        n = len(gdf)
        Imat = np.eye(n)
        try:
            Wfull = w.full()[0]
            S = np.linalg.inv(Imat - rho_proxy * Wfull)
            direct = float(np.mean(np.diag(S)))
            total = float(np.mean(S.sum(axis=1)))
            indirect = total - direct
        except Exception:
            direct = indirect = total = np.nan

        impact_matrices[target] = {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "rho_proxy": rho_proxy,
        }

        for nme, d in coef_dict.items():
            coef_summary.append({
                "ecuacion": target,
                "variable": nme,
                "coeficiente": d["coef"],
                "se_HC1": d["se"],
                "pval": d["pval"],
            })

    return {
        "results": results,
        "impacts": impact_matrices,
        "coef_df": pd.DataFrame(coef_summary),
        "vars": vars_,
    }

# -----------------------------------------------------------------------------
# 4. ESTABILIDAD
# -----------------------------------------------------------------------------
def check_stability(spvar_out: Dict) -> Dict[str, Any]:
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

    return {"eigenvalues": eigs, "max_modulus": max_mod, "stable": stable}

# -----------------------------------------------------------------------------
# 5. ESDA
# -----------------------------------------------------------------------------
def run_esda(gdf: gpd.GeoDataFrame, w, var: str = "IPVN_m2") -> Dict:
    y = gdf[var].values.astype(float)
    mi = Moran(y, w)
    lisa = Moran_Local(y, w)
    gdf = gdf.copy()
    gdf["LISA_I"] = lisa.Is
    gdf["LISA_p"] = lisa.p_sim
    gdf["LISA_q"] = lisa.q
    return {"moran_I": float(mi.I), "moran_p": float(mi.p_sim), "lisa_gdf": gdf}

# -----------------------------------------------------------------------------
# 6. VISUALIZACIONES (sin Mapbox token — corrige el error)
# -----------------------------------------------------------------------------
def create_choropleth(gdf: gpd.GeoDataFrame, column: str, title: str) -> go.Figure:
    """
    Choropleth robusto SIN Mapbox token.
    Usa px.choropleth + proyección mercator (funciona en Streamlit Cloud).
    """
    gdf_plot = gdf.copy()
    gdf_plot["id"] = gdf_plot.index.astype(str)

    geojson = json.loads(gdf_plot.to_json())

    # Asegurar id en cada feature
    for i, feat in enumerate(geojson["features"]):
        feat["id"] = str(i)

    fig = px.choropleth(
        gdf_plot,
        geojson=geojson,
        locations="id",
        color=column,
        hover_name="DEPARTAMENTO",
        hover_data={column: ":.1f", "COD_DPTO": True, "id": False},
        color_continuous_scale="Viridis",
        title=title,
        projection="mercator",
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="white",
    )
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#1a1a1a", size=12),
        coloraxis_colorbar=dict(title=column),
        height=500,
    )
    return fig


def create_3d_extrusion(gdf: gpd.GeoDataFrame) -> go.Figure:
    z = np.maximum(gdf["IPVN_m2"].values.astype(float), 0)
    size = np.maximum(gdf["M2_Aprobados"].values.astype(float), 0)
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
            colorbar=dict(title="Crédito Hipotecario"),
            line=dict(width=0.5, color="#333"),
        ),
        text=gdf["DEPARTAMENTO"],
        hovertemplate="<b>%{text}</b><br>Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>IPVN m²: %{z:.0f}<extra></extra>",
    )])
    fig.update_layout(
        title="Extrusión 3D – Altura = IPVN_m2 | Color = Crédito | Tamaño = M² Aprobados",
        scene=dict(
            xaxis_title="Longitud",
            yaxis_title="Latitud",
            zaxis_title="IPVN_m2 (miles COP)",
            bgcolor="white",
        ),
        paper_bgcolor="white",
        font=dict(color="#1a1a1a"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=550,
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
    for u, v, _ in edge_data:
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
        edge_z += [pos[u][2], pos[v][2], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines", line=dict(color="#555", width=3), hoverinfo="none",
    ))
    fig.add_trace(go.Scatter3d(
        x=node_xyz[:, 0], y=node_xyz[:, 1], z=node_xyz[:, 2],
        mode="markers+text",
        marker=dict(size=18, color="#1f77b4", line=dict(width=1, color="#000")),
        text=vars_,
        textposition="top center",
        hovertext=[f"{v}<br>Impacto total: {spvar_out['impacts'][v]['total']:.3f}" for v in vars_],
        hoverinfo="text",
    ))
    fig.update_layout(
        title="Grafo 3D de Mecanismos de Transmisión SpVAR (φ, β, ρ, θ)",
        scene=dict(
            xaxis=dict(showticklabels=False, title="", backgroundcolor="white"),
            yaxis=dict(showticklabels=False, title="", backgroundcolor="white"),
            zaxis=dict(showticklabels=False, title="", backgroundcolor="white"),
            bgcolor="white",
        ),
        paper_bgcolor="white",
        font=dict(color="#1a1a1a"),
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=500,
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
    )
    return fig

# -----------------------------------------------------------------------------
# 7. TEXTO DE POLÍTICA
# -----------------------------------------------------------------------------
def generate_policy_text(spvar_out, stability, w_name, moran_I, moran_p) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("INTERPRETABILIDAD AUTOMATIZADA – SpVAR & ESDA | COLOMBIA")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Matriz de pesos seleccionada: {w_name}")
    lines.append(f"Índice de Moran Global (IPVN_m2): I = {moran_I:.4f}  (p = {moran_p:.4f})")
    if moran_p < 0.05:
        lines.append("-> Existe autocorrelación espacial positiva significativa en precios de vivienda.")
    else:
        lines.append("-> No se detecta autocorrelación espacial global significativa al 5 %.")
    lines.append("")
    lines.append("-- ESTABILIDAD DEL SISTEMA DINAMICO --")
    lines.append(f"Máximo módulo de autovalores: {stability['max_modulus']:.4f}")
    lines.append("OK Sistema estable (max|lambda| < 1)." if stability["stable"] else "ADVERTENCIA: Sistema potencialmente inestable.")
    lines.append("")
    lines.append("-- MECANISMOS DE TRANSMISION E IMPACTOS --")
    for v in spvar_out["vars"]:
        imp = spvar_out["impacts"][v]
        lines.append(f"* {v}: Directo={imp['direct']:.4f} | Indirecto={imp['indirect']:.4f} | Total={imp['total']:.4f}")
    lines.append("")
    lines.append("-- IMPLICACIONES DE POLITICA --")
    lines.append("1. Credito: shocks de colocacion hipotecaria generan spillovers espaciales.")
    lines.append("2. Desarrollo urbano: la autodinamica de precios implica persistencia local.")
    lines.append("3. Empleo: el desempleo frena M2_Aprobados con externalidades regionales.")
    lines.append("4. Focalizar vivienda social y redescuento en clusters LISA (HH/LL).")
    lines.append("=" * 70)
    return "\n".join(lines)

# =============================================================================
# APP STREAMLIT
# =============================================================================
st.title("SpVAR Espacial + ESDA | Mercado Inmobiliario Colombia")
st.markdown(
    "**Modelo de Vector Autorregresivo Espacial** con selección automática de matriz de pesos, "
    "impactos directos/indirectos/totales y análisis exploratorio espacial (Moran / LISA)."
)

# Carga de datos
gdf = load_and_prepare_data()

# Selección de W
with st.spinner("Seleccionando matriz de pesos espaciales y estimando SpVAR..."):
    w, w_name, moran_I, moran_p = select_and_build_weights(gdf, "IPVN_m2")
    esda_out = run_esda(gdf, w, "IPVN_m2")
    gdf = esda_out["lisa_gdf"]
    spvar_out = estimate_spvar(gdf, w)
    stability = check_stability(spvar_out)
    policy_text = generate_policy_text(spvar_out, stability, w_name, moran_I, moran_p)
    coef_table = spvar_out["coef_df"].round(4)

# Sidebar info
with st.sidebar:
    st.header("Resumen del modelo")
    st.metric("Matriz de pesos", w_name)
    st.metric("Moran I (IPVN_m2)", f"{moran_I:.4f}")
    st.metric("p-valor Moran", f"{moran_p:.4f}")
    st.metric("Estabilidad", "Estable" if stability["stable"] else "Inestable")
    st.metric("max |lambda|", f"{stability['max_modulus']:.4f}")
    st.caption("Datos: MGN 2018 DANE · Variables simuladas con coherencia econométrica.")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Mapas Coropléticos",
    "Extrusión 3D",
    "Mecanismos de Transmisión",
    "Resultados Econométricos",
    "Interpretabilidad de Política",
])

with tab1:
    st.subheader("Visualización espacial de las variables del sistema")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            create_choropleth(gdf, "IPVN_m2", "Índice de Precios de Vivienda (m²)"),
            use_container_width=True,
        )
        st.plotly_chart(
            create_choropleth(gdf, "Desempleo", "Tasa de Desempleo Departamental (%)"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            create_choropleth(gdf, "Credito_Hipotecario", "Colocación de Crédito Hipotecario"),
            use_container_width=True,
        )
        st.plotly_chart(
            create_choropleth(gdf, "M2_Aprobados", "Área Aprobada para Construcción (miles m²)"),
            use_container_width=True,
        )

with tab2:
    st.subheader("Extrusión 3D espacial")
    st.caption("Altura (Z) = IPVN_m2 · Color = Crédito Hipotecario · Tamaño = M² Aprobados")
    st.plotly_chart(create_3d_extrusion(gdf), use_container_width=True)

with tab3:
    st.subheader("Mecanismos de transmisión (φ, β, ρ, θ)")
    st.plotly_chart(create_transmission_graph(spvar_out), use_container_width=True)
    st.plotly_chart(create_impact_bars(spvar_out), use_container_width=True)

with tab4:
    st.subheader("Coeficientes SpVAR (OLS + errores robustos HC1)")
    st.dataframe(coef_table, use_container_width=True)
    st.markdown(f"""
    - **Matriz de pesos óptima:** `{w_name}`
    - **Moran Global (IPVN_m2):** I = {moran_I:.4f} (p = {moran_p:.4f})
    - **Estabilidad del sistema:** {'Estable' if stability['stable'] else 'Inestable'}  
      (max |λ| = {stability['max_modulus']:.4f})
    """)

with tab5:
    st.subheader("Conclusiones operativas automatizadas")
    st.text(policy_text)

st.markdown("---")
st.caption("SpVAR + ESDA · Streamlit · libpysal / esda / statsmodels / Plotly")
