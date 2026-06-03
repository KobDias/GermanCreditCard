"""
app.py – Desafio 16 | Risco de Crédito | Grupo 16
Aplicação Streamlit para predição de risco de crédito usando German Credit Dataset.

Uso:
    streamlit run app.py

O app carrega `model/modelo_final.joblib` (pipeline completo: pré-processamento + GradientBoosting).
Se o arquivo não for encontrado, exibe instrução para treinar o notebook primeiro.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditAI – Risco de Crédito",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS customizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Background */
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f1117 100%);
    border: 1px solid #2d3561;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.header-banner h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #e8eaf0;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.header-banner .subtitle {
    color: #8892b0;
    font-size: 0.95rem;
    font-weight: 400;
}
.header-banner .badge {
    display: inline-block;
    background: #2d3561;
    color: #64ffda;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 0.6rem;
}

/* Section titles */
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64ffda;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2a3a;
}

/* Result cards */
.result-good {
    background: linear-gradient(135deg, #0d2117 0%, #0a1f15 100%);
    border: 2px solid #2ecc71;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-bad {
    background: linear-gradient(135deg, #2d0d0d 0%, #1f0a0a 100%);
    border: 2px solid #e74c3c;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.result-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    opacity: 0.85;
}
.result-desc {
    font-size: 0.9rem;
    opacity: 0.7;
    margin-top: 0.8rem;
    line-height: 1.5;
}

/* Info box */
.info-box {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-left: 4px solid #64ffda;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: #8892b0;
    line-height: 1.6;
}

/* Warning box */
.warn-box {
    background: #1a150d;
    border: 1px solid #f39c12;
    border-left: 4px solid #f39c12;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: #c9a84c;
}

/* Error box */
.error-box {
    background: #1a0d0d;
    border: 1px solid #e74c3c;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-size: 0.9rem;
    color: #e87070;
    line-height: 1.6;
}

/* Metric pill */
.metric-pill {
    display: inline-block;
    background: #1a2035;
    border: 1px solid #2d3561;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #a8b2d8;
}
.metric-pill span {
    color: #64ffda;
    font-weight: 600;
}

/* Streamlit component overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #8892b0 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #2d3561, #1a2035);
    color: #64ffda;
    border: 1px solid #2d3561;
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    padding: 0.7rem 2.5rem;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #3d4571, #2a3045);
    border-color: #64ffda;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(100, 255, 218, 0.15);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #8892b0;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #64ffda !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #3a4060;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1a1f2e;
}
</style>
""", unsafe_allow_html=True)


# ── Carregamento do modelo ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Carrega o pipeline completo salvo em model/modelo_final.joblib."""
    model_path = "model/modelo_final.joblib"
    if not os.path.exists(model_path):
        return None, f"Arquivo `{model_path}` não encontrado. Execute o notebook e salve o modelo primeiro."
    try:
        pipeline = joblib.load(model_path)
        return pipeline, None
    except Exception as e:
        return None, f"Erro ao carregar o modelo: {e}"


pipeline, load_error = load_model()


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>💳 CreditAI</h1>
    <div class="subtitle">Avaliação Preditiva de Risco de Crédito</div>
    <div class="badge">German Credit Dataset · Gradient Boosting · Grupo 16</div>
</div>
""", unsafe_allow_html=True)


# ── Alerta se modelo não carregado ──────────────────────────────────────────
if load_error:
    st.markdown(f"""
    <div class="error-box">
        <strong>⚠️ Modelo não encontrado</strong><br><br>
        {load_error}<br><br>
        <strong>Passos:</strong><br>
        1. Execute o notebook <code>notebooks/notebook_atualizado.ipynb</code> do início ao fim<br>
        2. A célula de salvamento criará <code>model/modelo_final.joblib</code><br>
        3. Coloque o arquivo na pasta <code>model/</code> na raiz do projeto<br>
        4. Reinicie o app com <code>streamlit run app.py</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Tabs ────────────────────────────────────────────────────────────────────
tab_pred, tab_info = st.tabs(["🔍 Avaliação de Crédito", "ℹ️ Sobre o Modelo"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDIÇÃO
# ════════════════════════════════════════════════════════════════════════════
with tab_pred:

    st.markdown("""
    <div class="info-box">
        Preencha o perfil do solicitante abaixo. O modelo retorna a probabilidade de
        <strong>risco ruim</strong> (inadimplência) e a classificação final.
        A classe de interesse de negócio é <strong>bad</strong> (risco = 1).
    </div>
    """, unsafe_allow_html=True)

    # ── Formulário ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Dados Financeiros</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        duration = st.number_input(
            "Duração do crédito (meses)",
            min_value=1, max_value=72, value=24, step=1,
            help="Prazo de pagamento em meses"
        )
    with col2:
        credit_amount = st.number_input(
            "Valor do crédito (DM)",
            min_value=100, max_value=20000, value=3000, step=100,
            help="Valor em Deutsche Marks"
        )
    with col3:
        saving_accounts = st.selectbox(
            "Conta poupança",
            options=["none", "little", "moderate", "quite rich", "rich"],
            index=1,
            help="Saldo na conta poupança"
        )

    col4, col5 = st.columns(2)
    with col4:
        checking_account = st.selectbox(
            "Conta corrente",
            options=["none", "little", "moderate", "rich"],
            index=1,
            help="Saldo na conta corrente"
        )
    with col5:
        purpose = st.selectbox(
            "Finalidade do crédito",
            options=["car", "furniture/equipment", "radio/TV", "domestic appliances",
                     "repairs", "education", "business", "vacation/others"],
            index=2,
            help="Para que o crédito será usado"
        )

    st.markdown('<div class="section-title">Dados Pessoais</div>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)

    with col6:
        age = st.number_input(
            "Idade (anos)",
            min_value=18, max_value=90, value=35, step=1
        )
    with col7:
        sex = st.selectbox(
            "Sexo",
            options=["male", "female"],
            index=0,
            help="Usado com cautela ética — ver seção Sobre o Modelo"
        )
    with col8:
        housing = st.selectbox(
            "Moradia",
            options=["own", "free", "rent"],
            index=0,
            help="Situação de moradia"
        )

    job = st.select_slider(
        "Qualificação profissional",
        options=[0, 1, 2, 3],
        value=2,
        format_func=lambda x: {
            0: "0 – Não qualificado / residente",
            1: "1 – Não qualificado / não residente",
            2: "2 – Qualificado",
            3: "3 – Altamente qualificado"
        }[x],
        help="Nível de qualificação do solicitante (0–3)"
    )

    st.markdown("")

    # ── Botão de predição ────────────────────────────────────────────────────
    predict_btn = st.button("🔮 Avaliar Risco de Crédito")

    if predict_btn:
        # Monta o DataFrame com as mesmas colunas do treino
        input_data = pd.DataFrame([{
            "Age":               age,
            "Sex":               sex,
            "Job":               job,
            "Housing":           housing,
            "Saving accounts":   saving_accounts,
            "Checking account":  checking_account,
            "Credit amount":     credit_amount,
            "Duration":          duration,
            "Purpose":           purpose,
        }])

        try:
            prob_bad  = pipeline.predict_proba(input_data)[0][1]
            prob_good = 1 - prob_bad
            prediction = pipeline.predict(input_data)[0]  # 1=bad, 0=good

            # ── Resultado ────────────────────────────────────────────────────
            if prediction == 1:
                st.markdown(f"""
                <div class="result-bad">
                    <div class="result-icon">⚠️</div>
                    <div class="result-label" style="color:#e74c3c;">RISCO RUIM</div>
                    <div class="result-prob">Probabilidade de inadimplência: <strong>{prob_bad:.1%}</strong></div>
                    <div class="result-desc">
                        O modelo classificou este perfil como <strong>alto risco</strong>.<br>
                        Recomenda-se análise adicional antes da concessão do crédito.<br>
                        <em>Threshold padrão: 0.50 · Ajustável conforme política de risco.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-good">
                    <div class="result-icon">✅</div>
                    <div class="result-label" style="color:#2ecc71;">RISCO BOM</div>
                    <div class="result-prob">Probabilidade de inadimplência: <strong>{prob_bad:.1%}</strong></div>
                    <div class="result-desc">
                        O modelo classificou este perfil como <strong>baixo risco</strong>.<br>
                        Perfil compatível com concessão de crédito.<br>
                        <em>Decisão final deve considerar demais critérios institucionais.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Detalhe probabilístico ────────────────────────────────────
            st.markdown("")
            st.markdown('<div class="section-title">Detalhe das Probabilidades</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="Prob. Risco BOM (good)",
                    value=f"{prob_good:.1%}",
                    delta=None
                )
            with col_b:
                st.metric(
                    label="Prob. Risco RUIM (bad)",
                    value=f"{prob_bad:.1%}",
                    delta=None
                )

            st.progress(prob_bad, text=f"Índice de risco de inadimplência: {prob_bad:.1%}")

            # ── Interpretação do perfil ───────────────────────────────────
            st.markdown('<div class="section-title">Fatores de Risco Identificados</div>', unsafe_allow_html=True)
            factors = []
            if duration > 24:
                factors.append("⚠️ Prazo longo (> 24 meses) — aumenta exposição ao risco")
            if credit_amount > 5000:
                factors.append("⚠️ Valor alto (> DM 5.000) — exige maior capacidade de pagamento")
            if checking_account in ["none", "little"]:
                factors.append("⚠️ Conta corrente com saldo baixo — indicador de liquidez reduzida")
            if saving_accounts in ["none", "little"]:
                factors.append("⚠️ Poupança baixa — menor reserva financeira")
            if purpose in ["education", "vacation/others"]:
                factors.append("⚠️ Finalidade de maior risco histórico (educação/outros)")
            if age < 25:
                factors.append("ℹ️ Solicitante jovem — histórico de crédito limitado")

            if factors:
                for f in factors:
                    st.markdown(f"- {f}")
            else:
                st.markdown("✅ Nenhum fator de risco elevado identificado neste perfil.")

            st.markdown("""
            <div class="warn-box">
                ⚖️ <strong>Aviso ético:</strong> Esta predição é baseada em dados históricos e pode
                conter vieses. A variável <em>Sex</em> é incluída apenas para fins acadêmicos.
                Decisões de crédito reais devem seguir a LGPD e normas de não-discriminação.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erro na predição: {e}")
            st.markdown("Verifique se o modelo foi salvo corretamente e se as colunas batem com o pipeline.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: SOBRE O MODELO
# ════════════════════════════════════════════════════════════════════════════
with tab_info:

    st.markdown('<div class="section-title">Sobre o Projeto</div>', unsafe_allow_html=True)
    st.markdown("""
    Esta aplicação é o entregável final do **Desafio 16 – Risco de Crédito** do curso de
    Inteligência Artificial (UNIMAR). O objetivo é classificar solicitantes como **bom** ou
    **mau pagador** com base no German Credit Dataset (UCI / Kaggle).
    """)

    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Fonte:** [German Credit Data – Kaggle](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)
    - **Amostras:** 1.000 solicitantes
    - **Distribuição:** 70% good / 30% bad (desbalanceado)
    - **Variável-alvo:** `Risk` → recodificada como `bad = 1` (classe positiva de negócio)
    """)

    st.markdown('<div class="section-title">Pipeline de Pré-processamento</div>', unsafe_allow_html=True)
    st.markdown("""
    | Tipo de variável | Colunas | Transformação |
    |---|---|---|
    | Numéricas contínuas | `Age`, `Credit amount`, `Duration` | `StandardScaler` |
    | Ordinais | `Saving accounts`, `Checking account` | `OrdinalEncoder` (ordem manual) |
    | Nominais | `Sex`, `Housing`, `Purpose` | `OneHotEncoder` |
    | Ordinal numérica | `Job` | passthrough (0–3) |
    """)

    st.markdown("""
    <div class="info-box">
        <strong>Correção P2:</strong> Na P1, variáveis nominais eram codificadas com <code>LabelEncoder</code>,
        atribuindo ordem artificial. Corrigido para <code>OneHotEncoder</code>, que cria colunas binárias
        independentes sem impor hierarquia.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Modelos Comparados</div>', unsafe_allow_html=True)

    metrics_data = {
        "Modelo": ["Logistic Regression", "Random Forest", "Gradient Boosting ✓"],
        "Acurácia":  ["~0.72", "~0.74", "~0.71"],
        "Recall (bad)": ["~0.45", "~0.29", "~0.51"],
        "F1 (bad)":  ["~0.46", "~0.40", "~0.50"],
        "AUC-ROC":   ["~0.73", "~0.76", "~0.72"],
    }
    st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        <strong>Por que Gradient Boosting?</strong><br>
        Embora o Random Forest tenha AUC ligeiramente maior (~0.756 vs ~0.719), o <strong>Recall(bad)</strong>
        do GBM é 22 pontos percentuais superior (0.51 vs 0.29).<br><br>
        No domínio de crédito, um <strong>Falso Negativo</strong> (aprovar um inadimplente) tem
        custo financeiro direto para o banco. Maximizar o Recall(bad) é prioritário.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Métricas do Modelo Final (Teste)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:1rem;">
        <div class="metric-pill">AUC-ROC <span>~0.72</span></div>
        <div class="metric-pill">Recall(bad) <span>~0.51</span></div>
        <div class="metric-pill">F1(bad) <span>~0.50</span></div>
        <div class="metric-pill">Acurácia <span>~0.71</span></div>
        <div class="metric-pill">Divisão <span>70/15/15</span></div>
        <div class="metric-pill">CV <span>Stratified K-Fold k=5</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Discussão Ética</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Viés de gênero:** A variável `Sex` pode gerar discriminação sistêmica. Em produção, seu uso
      deve ser avaliado conforme **LGPD** e normas de não-discriminação.
    - **Custo assimétrico:** FN (aprovar inadimplente) = perda financeira; FP (rejeitar bom pagador)
      = dano ao cliente. O threshold padrão (0.50) pode ser ajustado via curva ROC.
    - **Explicabilidade (LGPD art. 20):** Decisões automatizadas de crédito devem ser auditáveis.
      Recomenda-se uso de SHAP/LIME para modelos em produção.
    - **Data drift:** Reavaliar periodicamente com novos dados para detectar degradação.
    """)

    st.markdown('<div class="section-title">Equipe</div>', unsafe_allow_html=True)
    st.markdown("""
    | Aluno | RA |
    |---|---|
    | Renan Gonçalves Rodrigues | 2040669 |
    | Sofhia Kobor Dias | 2095896 |

    **Curso:** Inteligência Artificial · UNIMAR 1/2026
    """)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CreditAI · Desafio 16 – Risco de Crédito · Grupo 16 · UNIMAR 2026<br>
    Modelo: GradientBoostingClassifier · Dataset: German Credit (UCI/Kaggle)
</div>
""", unsafe_allow_html=True)
