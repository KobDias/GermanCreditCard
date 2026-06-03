import streamlit as st
from componentes.estilos import aplicar_estilos
from componentes.modelo import carregar_modelo
from visoes import avaliacao, sobre

st.set_page_config(
    page_title="CreditAI – Risco de Crédito",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

aplicar_estilos()

st.markdown("""
<div class="cabecalho">
    <h1>💳 CreditAI</h1>
    <div class="subtitulo">Avaliação Preditiva de Risco de Crédito</div>
    <div class="badge">German Credit Dataset · Gradient Boosting · Grupo 16</div>
</div>
""", unsafe_allow_html=True)

pipeline, erro = carregar_modelo()

if erro:
    st.markdown(f"""
    <div class="caixa-erro">
        <strong>⚠️ Modelo não encontrado</strong><br><br>
        {erro}<br><br>
        Execute o notebook e salve o modelo em <code>model/modelo_final.joblib</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

aba_pred, aba_info = st.tabs(["Avaliação de Crédito", "Sobre o Modelo"])

with aba_pred:
    avaliacao.renderizar(pipeline)

with aba_info:
    sobre.renderizar()

st.markdown("""
<div class="rodape">
    CreditAI · Desafio 16 · Grupo 16 · UNIMAR 2026<br>
    GradientBoostingClassifier · German Credit Dataset (UCI/Kaggle)
</div>
""", unsafe_allow_html=True)
