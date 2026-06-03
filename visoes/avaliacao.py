import streamlit as st
from componentes.modelo import OPCOES, montar_input, identificar_fatores


def renderizar(pipeline):
    st.markdown("""
    <div class="caixa-info">
        Preencha o perfil do solicitante. O modelo retorna a probabilidade de
        <strong>risco ruim</strong> (inadimplência) e a classificação final.
        Classe de interesse: <strong>bad</strong> (risco = 1).
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="secao-titulo">Dados Financeiros</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        duration = st.number_input("Duração do crédito (meses)", min_value=1, max_value=72, value=24, step=1)
    with col2:
        credit_amount = st.number_input("Valor do crédito (DM)", min_value=100, max_value=20000, value=3000, step=100)
    with col3:
        saving_accounts = st.selectbox("Conta poupança", OPCOES["saving_accounts"], index=1)

    col4, col5 = st.columns(2)
    with col4:
        checking_account = st.selectbox("Conta corrente", OPCOES["checking_account"], index=1)
    with col5:
        purpose = st.selectbox("Finalidade do crédito", OPCOES["purpose"], index=2)

    st.markdown('<div class="secao-titulo">Dados Pessoais</div>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    with col6:
        age = st.number_input("Idade (anos)", min_value=18, max_value=90, value=35, step=1)
    with col7:
        sex = st.selectbox("Sexo", OPCOES["sex"], index=0, help="Usado com cautela ética — ver aba Sobre o Modelo")
    with col8:
        housing = st.selectbox("Moradia", OPCOES["housing"], index=0)

    job = st.select_slider(
        "Qualificação profissional",
        options=list(OPCOES["job"].keys()),
        value=2,
        format_func=lambda x: OPCOES["job"][x],
    )

    st.markdown("")

    if st.button("Avaliar Risco de Crédito"):
        entrada = montar_input(age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose)
        try:
            prob_ruim = pipeline.predict_proba(entrada)[0][1]
            prob_bom = 1 - prob_ruim
            predicao = pipeline.predict(entrada)[0]

            if predicao == 1:
                st.markdown(f"""
                <div class="resultado-ruim">
                    <div class="resultado-icone">⚠️</div>
                    <div class="resultado-label" style="color:#e74c3c;">RISCO RUIM</div>
                    <div class="resultado-prob">Probabilidade de inadimplência: <strong>{prob_ruim:.1%}</strong></div>
                    <div class="resultado-desc">
                        Perfil classificado como <strong>alto risco</strong>.<br>
                        Recomenda-se análise adicional antes da concessão.<br>
                        <em>Threshold padrão: 0.50 · Ajustável conforme política de risco.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="resultado-bom">
                    <div class="resultado-icone">✅</div>
                    <div class="resultado-label" style="color:#2ecc71;">RISCO BOM</div>
                    <div class="resultado-prob">Probabilidade de inadimplência: <strong>{prob_ruim:.1%}</strong></div>
                    <div class="resultado-desc">
                        Perfil classificado como <strong>baixo risco</strong>.<br>
                        Compatível com concessão de crédito.<br>
                        <em>Decisão final deve considerar demais critérios institucionais.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown('<div class="secao-titulo">Detalhe das Probabilidades</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Prob. Risco BOM (good)", f"{prob_bom:.1%}")
            with col_b:
                st.metric("Prob. Risco RUIM (bad)", f"{prob_ruim:.1%}")
            st.progress(prob_ruim, text=f"Índice de risco: {prob_ruim:.1%}")

            st.markdown('<div class="secao-titulo">Fatores de Risco Identificados</div>', unsafe_allow_html=True)
            fatores = identificar_fatores(duration, credit_amount, checking_account, saving_accounts, purpose, age)
            if fatores:
                for f in fatores:
                    st.markdown(f"- ⚠️ {f}")
            else:
                st.markdown("✅ Nenhum fator de risco elevado identificado.")

            st.markdown("""
            <div class="caixa-aviso">
                ⚖️ <strong>Aviso ético:</strong> Esta predição é baseada em dados históricos e pode conter vieses.
                A variável <em>Sex</em> é incluída apenas para fins acadêmicos.
                Decisões reais devem seguir a LGPD e normas de não-discriminação.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erro na predição: {e}")
