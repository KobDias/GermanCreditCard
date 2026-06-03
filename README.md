# CreditAI – Risco de Crédito

Desafio 16 · Classificação Binária · Domínio Financeiro  
Curso de Inteligência Artificial – UNIMAR 1/2026

---

## Integrantes

| Aluno | RA |
|---|---|
| Renan Gonçalves Rodrigues | 2040669 |
| Sofhia Kobor Dias | 2095896 |

---

## Problema

Instituições financeiras precisam avaliar risco de inadimplência antes de conceder crédito. Este projeto treina um classificador binário que, dado o perfil financeiro e demográfico de um solicitante, prediz se ele representa bom risco (pagará) ou mau risco (inadimplente).

**Dataset:** [German Credit Data – Kaggle](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk) · 1.000 amostras · 70% good / 30% bad · variável-alvo `Risk` (bad = 1)

---

## Como Rodar

**App publicado:** [germancreditcard.streamlit.app](https://germancreditcard.streamlit.app)

Para rodar localmente, clone o repositório e instale as dependências:

```bash
git clone https://github.com/KobDias/GermanCreditCard.git
cd GermanCreditCard
pip install -r requirements.txt
```

Execute o notebook para gerar o modelo:

```bash
jupyter notebook notebooks/notebook_atualizado.ipynb
# Kernel > Restart & Run All
# O modelo será salvo em model/modelo_final.joblib
```

Suba o app:

```bash
streamlit run app.py
```

---

## Metodologia

O notebook cobre cinco etapas: análise exploratória (distribuição do target, variáveis numéricas e categóricas por classe de risco), pré-processamento com pipeline tipado por variável (`OneHotEncoder` para nominais, `OrdinalEncoder` para ordinais, `StandardScaler` para numéricas), comparação de três classificadores com Stratified K-Fold (k=5), análise comparativa com feature importance e discussão de negócio, e salvamento do pipeline completo via `joblib`.

**Correções da P1:** `LabelEncoder` → `OneHotEncoder` nas nominais; `StandardScaler` removido das ordinais; métricas reorientadas para `pos_label=1` (classe bad).

---

## Resultados

| Modelo | Recall (bad) | F1 (bad) | AUC-ROC | AUC-CV |
|---|---|---|---|---|
| Logistic Regression | 0.2222 | 0.3125 | 0.6459 | 0.6988 ± 0.014 |
| Random Forest | 0.2889 | 0.4127 | 0.7562 | 0.7645 ± 0.031 |
| **Gradient Boosting** | **0.5111** | **0.5287** | 0.7192 | 0.7521 ± 0.041 |

Modelo final: **GradientBoostingClassifier** (`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`).

O Random Forest tem AUC maior (0.756 vs 0.719), mas o Gradient Boosting tem Recall(bad) 22pp superior (0.51 vs 0.29). No domínio financeiro, um Falso Negativo (aprovar inadimplente) tem custo direto — maximizar Recall(bad) é prioritário.

---

## Estrutura

```
GermanCreditCard/
├── app.py
├── requirements.txt
├── .python-version
├── componentes/
│   ├── estilos.py
│   └── modelo.py
├── visoes/
│   ├── avaliacao.py
│   └── sobre.py
├── model/
│   ├── modelo_final.joblib
│   └── feature_info.joblib
├── notebooks/
│   └── notebook_atualizado.ipynb
├── reports/
│   └── relatorio_atualizado.pdf
└── data/
    └── german_credit_data.csv
```

---

## Conclusão e Ética

O projeto cobre um fluxo completo de ML aplicado a crédito: da análise exploratória ao deploy. A escolha do modelo foi orientada ao negócio, não apenas à métrica de AUC.

A variável `Sex` foi incluída para fins acadêmicos, mas seu uso em produção é problemático — modelos de crédito podem perpetuar discriminação sistêmica. A LGPD (art. 20) exige que decisões automatizadas sejam explicáveis; modelos em produção devem usar SHAP/LIME. O threshold de 0.50 deve ser calibrado conforme a matriz de custos da instituição.
