import streamlit as st
import pandas as pd


def renderizar():
    st.markdown('<div class="secao-titulo">Sobre o Projeto</div>', unsafe_allow_html=True)
    st.markdown("""
    Entregável final do **Desafio 16 – Risco de Crédito** (UNIMAR 1/2026).
    Classifica solicitantes como bom ou mau pagador com base no German Credit Dataset.
    """)

    st.markdown('<div class="secao-titulo">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Fonte:** [German Credit Data – Kaggle](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)
    - **Amostras:** 1.000 · **Distribuição:** 70% good / 30% bad
    - **Variável-alvo:** `Risk` → `bad = 1` (classe positiva)
    """)

    st.markdown('<div class="secao-titulo">Pipeline de Pré-processamento</div>', unsafe_allow_html=True)
    st.markdown("""
    | Tipo | Colunas | Transformação |
    |---|---|---|
    | Numéricas | `Age`, `Credit amount`, `Duration` | `StandardScaler` |
    | Ordinais | `Saving accounts`, `Checking account` | `OrdinalEncoder` |
    | Nominais | `Sex`, `Housing`, `Purpose` | `OneHotEncoder` |
    | Ordinal numérica | `Job` | passthrough (0–3) |
    """)

    st.markdown('<div class="secao-titulo">Modelos Comparados</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Modelo":       ["Logistic Regression", "Random Forest", "Gradient Boosting ✓"],
        "Recall (bad)": ["0.2222", "0.2889", "0.5111"],
        "F1 (bad)":     ["0.3125", "0.4127", "0.5287"],
        "AUC-ROC":      ["0.6459", "0.7562", "0.7192"],
        "AUC-CV":       ["0.6988 ± 0.014", "0.7645 ± 0.031", "0.7521 ± 0.041"],
    }), hide_index=True, use_container_width=True)

    st.markdown("""
    <div class="caixa-info">
        <strong>Por que Gradient Boosting?</strong><br>
        O Random Forest tem AUC maior (0.756 vs 0.719), mas o Gradient Boosting tem
        Recall(bad) 22pp superior (0.51 vs 0.29). No domínio financeiro, um Falso Negativo
        (aprovar inadimplente) tem custo direto para o banco — maximizar o Recall(bad) é prioritário.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="secao-titulo">Métricas do Modelo Final</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:1rem;">
        <div class="metrica-pill">AUC-ROC <span>0.7192</span></div>
        <div class="metrica-pill">Recall(bad) <span>0.5111</span></div>
        <div class="metrica-pill">F1(bad) <span>0.5287</span></div>
        <div class="metrica-pill">Acurácia <span>0.7267</span></div>
        <div class="metrica-pill">Divisão <span>70/15/15</span></div>
        <div class="metrica-pill">CV <span>K-Fold k=5</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="secao-titulo">Discussão Ética</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Viés de gênero:** `Sex` pode gerar discriminação sistêmica. Uso em produção requer avaliação conforme LGPD.
    - **Custo assimétrico:** FN (aprovar inadimplente) = perda financeira; FP (rejeitar bom pagador) = dano ao cliente.
    - **Explicabilidade (LGPD art. 20):** Modelos em produção devem usar SHAP/LIME para auditabilidade.
    - **Data drift:** Reavaliar periodicamente com novos dados.
    """)

    st.markdown('<div class="secao-titulo">Equipe</div>', unsafe_allow_html=True)
    st.markdown("""
    | Aluno | RA |
    |---|---|
    | Renan Gonçalves Rodrigues | 2040669 |
    | Sofhia Kobor Dias | 2095896 |

    Inteligência Artificial · UNIMAR 1/2026
    """)
