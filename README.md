---
title: SpVAR Colombia - Mercado Inmobiliario
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# SpVAR Espacial + ESDA | Mercado Inmobiliario Colombia

Modelo de **Vector Autorregresivo Espacial (SpVAR)** con Análisis Exploratorio de Datos Espaciales (ESDA) aplicado al mercado inmobiliario y dinámica económica de los departamentos de Colombia.

## Características

- **Selección automática de matriz de pesos espaciales** (Queen, Rook, KNN) basada en Índice de Moran Global
- **Estimación SpVAR** con errores robustos HC1
- **Impactos espaciales**: Directos, Indirectos y Totales
- **Auditoría de estabilidad** del sistema dinámico (autovalores)
- **Mapas coropléticos interactivos** (Plotly)
- **Extrusión 3D espacial** de variables
- **Grafo 3D de mecanismos de transmisión** (φ, β, ρ, θ)
- **Panel de interpretabilidad automatizada** orientado a política económica

## Variables del modelo

| Variable | Descripción |
|----------|-------------|
| `IPVN_m2` | Índice de Precios de Vivienda / Valor m² |
| `Credito_Hipotecario` | Colocación de Crédito Hipotecario |
| `Desempleo` | Tasa de Desempleo Departamental |
| `M2_Aprobados` | Área Aprobada para Construcción |

## Datos

- Geometría: Marco Geoestadístico Nacional (MGN) 2018 – DANE
- Variables: simuladas con coherencia econométrica a partir de patrones reales de Colombia

## Stack técnico

- `libpysal` / `esda` – econometría espacial
- `statsmodels` – estimación OLS + HC1
- `geopandas` – manejo de datos geoespaciales
- `plotly` – visualizaciones interactivas
- `gradio` – interfaz web

## Cómo usar

1. El Space carga automáticamente la geometría y estima el modelo al iniciar.
2. Navega por las pestañas:
   - **Mapas Coropléticos**
   - **Extrusión 3D Espacial**
   - **Mecanismos de Transmisión**
   - **Resultados Econométricos**
   - **Interpretabilidad de Política**

## Licencia

MIT
