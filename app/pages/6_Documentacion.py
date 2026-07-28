"""Toda la documentación del proyecto, renderizada dentro de la app — nada queda
solo en un .md que el visitante no va a abrir. ARCHITECTURE.md §11."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

st.title("6 · Documentación")
st.caption("Todo el trabajo del proyecto, sin salir de esta web ni tener que ir a GitHub.")


def render_markdown_file(path: Path, strip_images: bool = True) -> None:
    if not path.exists():
        st.info(f"Falta {path.relative_to(ROOT)}.")
        return
    text = path.read_text(encoding="utf-8")
    if strip_images:
        # las imágenes relativas (reports/figures/...) no cargan servidas así en Streamlit;
        # se quitan de la vista en línea, las figuras ya están en las pestañas 2-5 como gráficos vivos.
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    st.markdown(text)


tab_guia, tab_informe, tab_referencias, tab_tecnica = st.tabs([
    "📘 Guía de usuario (fácil)",
    "📊 Informe completo",
    "🔬 Referencias de calibración",
    "🛠️ Documentación técnica",
])

with tab_guia:
    st.caption("Para cualquiera que vea la app por primera vez: qué es cada pestaña y qué significa cada número.")
    render_markdown_file(ROOT / "docs" / "GUIA_USUARIO.md")

with tab_informe:
    st.caption("El análisis completo con todas las cifras, metodología y limitaciones declaradas.")
    render_markdown_file(ROOT / "reports" / "INFORME.md")

with tab_referencias:
    st.caption("De dónde salió cada rango usado para calibrar los datos sintéticos (también visible en la pestaña 1 · Datos).")
    render_markdown_file(ROOT / "data" / "external" / "referencias_publicas.md")

with tab_tecnica:
    st.markdown(
        "Documentación de diseño técnico: arquitectura del sistema, decisiones metodológicas y "
        "plan de ejecución del proyecto. Pensada para quien va a revisar o replicar el trabajo, "
        "no para el público general de la presentación."
    )
    tech_docs = {
        "ARCHITECTURE.md — diseño técnico completo (brazos, QUBO, protocolo experimental)": ROOT / "ARCHITECTURE.md",
        "PLAN.md — plan de ejecución del proyecto": ROOT / "PLAN.md",
        "README.md — resumen del repositorio": ROOT / "README.md",
        "reports/RESULTS.md — tabla de resultados en bruto": ROOT / "reports" / "RESULTS.md",
    }
    for label, path in tech_docs.items():
        if path.exists():
            st.download_button(
                f"📥 {label}",
                path.read_bytes(),
                file_name=path.name,
                mime="text/markdown",
                key=str(path),
            )
    with st.expander("Ver ARCHITECTURE.md completo aquí mismo"):
        render_markdown_file(ROOT / "ARCHITECTURE.md")
    with st.expander("Ver PLAN.md completo aquí mismo"):
        render_markdown_file(ROOT / "PLAN.md")

st.divider()
st.caption(
    "Código fuente completo (para quien quiera revisar cómo se calculó cada número): "
    "[github.com/0KevinB/tikee-quantum-scoring](https://github.com/0KevinB/tikee-quantum-scoring)"
)
