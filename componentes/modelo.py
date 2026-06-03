import os
import joblib
import streamlit as st
import pandas as pd


CAMINHO_MODELO = "model/modelo_final.joblib"

OPCOES = {
    "saving_accounts":   ["none", "little", "moderate", "quite rich", "rich"],
    "checking_account":  ["none", "little", "moderate", "rich"],
    "purpose":           ["car", "furniture/equipment", "radio/TV", "domestic appliances",
                          "repairs", "education", "business", "vacation/others"],
    "sex":               ["male", "female"],
    "housing":           ["own", "free", "rent"],
    "job":               {0: "0 – Não qualificado / residente",
                          1: "1 – Não qualificado / não residente",
                          2: "2 – Qualificado",
                          3: "3 – Altamente qualificado"},
}


@st.cache_resource
def carregar_modelo():
    if not os.path.exists(CAMINHO_MODELO):
        return None, f"Arquivo `{CAMINHO_MODELO}` não encontrado."
    try:
        return joblib.load(CAMINHO_MODELO), None
    except Exception as e:
        return None, f"Erro ao carregar o modelo: {e}"


def montar_input(age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose):
    return pd.DataFrame([{
        "Age":              age,
        "Sex":              sex,
        "Job":              job,
        "Housing":          housing,
        "Saving accounts":  saving_accounts,
        "Checking account": checking_account,
        "Credit amount":    credit_amount,
        "Duration":         duration,
        "Purpose":          purpose,
    }])


def identificar_fatores(duration, credit_amount, checking_account, saving_accounts, purpose, age):
    fatores = []
    if duration > 24:
        fatores.append("Prazo longo (> 24 meses) — aumenta exposição ao risco")
    if credit_amount > 5000:
        fatores.append("Valor alto (> DM 5.000) — exige maior capacidade de pagamento")
    if checking_account in ["none", "little"]:
        fatores.append("Conta corrente com saldo baixo — indicador de liquidez reduzida")
    if saving_accounts in ["none", "little"]:
        fatores.append("Poupança baixa — menor reserva financeira")
    if purpose in ["education", "vacation/others"]:
        fatores.append("Finalidade de maior risco histórico (educação/outros)")
    if age < 25:
        fatores.append("Solicitante jovem — histórico de crédito limitado")
    return fatores
