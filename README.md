---
title: SpVAR Colombia - Mercado Inmobiliario
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
license: mit
---

# SpVAR Espacial + ESDA | Mercado Inmobiliario Colombia

Modelo de **Vector Autorregresivo Espacial (SpVAR)** con Análisis Exploratorio de Datos Espaciales (ESDA) aplicado al mercado inmobiliario y dinámica económica de los departamentos de Colombia.

## Características

- Selección automática de matriz de pesos espaciales (Queen, Rook, KNN)
- Estimación SpVAR con errores robustos HC1
- Impactos espaciales: Directos, Indirectos y Totales
- Auditoría de estabilidad del sistema (autovalores)
- Mapas coropléticos interactivos (**sin token de Mapbox** — corrige el error anterior)
- Extrusión 3D espacial
- Grafo 3D de mecanismos de transmisión (φ, β, ρ, θ)
- Panel de interpretabilidad de política económica

## Variables

| Variable | Descripción |
|----------|-------------|
| IPVN_m2 | Índice de Precios de Vivienda / Valor m² |
| Credito_Hipotecario | Colocación de Crédito Hipotecario |
| Desempleo | Tasa de Desempleo Departamental |
| M2_Aprobados | Área Aprobada para Construcción |

## Stack

- streamlit
- libpysal / esda
- statsmodels
- geopandas
- plotly

## Licencia

MIT
