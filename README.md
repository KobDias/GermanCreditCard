# 💳 CreditAI – Risco de Crédito

> Desafio 16 · Domínio Financeiro · Classificação Binária  
> Curso de Inteligência Artificial – UNIMAR 1/2026

---

## 👥 Integrantes

| Aluno | RA |
|---|---|
| Renan Gonçalves Rodrigues | 2040669 |
| Sofhia Kobor Dias | 2095896 |

---

## 📌 Descrição do Problema

Instituições financeiras precisam avaliar o risco de inadimplência antes de conceder crédito.  
Este projeto constrói um classificador binário que, dado o perfil financeiro e demográfico de um solicitante, prediz se ele representa **bom risco** (pagará o crédito) ou **mau risco** (inadimplente).

---

## 🎯 Objetivo

Desenvolver um modelo preditivo de **risco de crédito** utilizando o German Credit Dataset, aplicando boas práticas de Machine Learning supervisionado, avaliação ética e deploy em aplicação interativa.

---

## 📊 Dataset

- **Nome:** German Credit Data
- **Fonte:** [Kaggle – kabure/german-credit-data-with-risk](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)
- **Amostras:** 1.000 solicitantes
- **Features:** 9 variáveis (Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose)
- **Variável-alvo:** `Risk` → `bad = 1` (classe positiva), `good = 0`
- **Desbalanceamento:** 70% good / 30% bad

> Para obter o arquivo: acesse o link do Kaggle acima, clique em **Download** e salve `german_credit_data.csv` na pasta `data/`.

---

## 🤖 Tipo de Problema de Machine Learning

**Classificação Binária Supervisionada** com dataset desbalanceado.

---

## 🔬 Metodologia

### Etapa 1 – Análise Exploratória (EDA)
- Distribuição da variável-alvo (70/30)
- Histogramas de variáveis numéricas por grupo de risco
- Taxa de inadimplência por categoria (Sex, Housing, Purpose)

### Etapa 2 – Pré-processamento
- Valores ausentes em `Saving accounts` e `Checking account` → preenchidos como `"none"` (sem conta)
- **Nominais** (`Sex`, `Housing`, `Purpose`): `OneHotEncoder` *(correção P2: substituiu LabelEncoder)*
- **Ordinais** (`Saving accounts`, `Checking account`): `OrdinalEncoder` com categorias mapeadas manualmente
- **Numéricas** (`Age`, `Credit amount`, `Duration`): `StandardScaler` *(correção P2: não aplicado sobre categóricas)*
- **Job**: passthrough (ordinal 0–3 já em escala natural)
- Divisão: 70% treino / 15% validação / 15% teste — **estratificada** (mantém proporção de bad)

### Etapa 3 – Comparação de Classificadores
- Validação cruzada: **Stratified K-Fold (k=5)** no conjunto treino+val
- Métricas reportadas para a **classe bad (pos_label=1)** *(correção P2: P1 avaliava a classe good)*
- Curvas ROC/AUC e matrizes de confusão

### Etapa 4 – Análise Comparativa
- Comparação de AUC-ROC e Recall(bad) entre os três modelos
- Feature importance por modelo
- Discussão de negócio e ética (viés de gênero, LGPD, custo assimétrico de erros)

### Etapa 5 – Salvamento e Deploy
- Modelo salvo como **pipeline completo** (pré-processamento + classificador) via `joblib`
- Aplicação Streamlit interativa com deploy no Streamlit Community Cloud

---

## 🏆 Modelos Treinados

| Modelo | Acurácia | Precisão (bad) | Recall (bad) | F1 (bad) | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.50 | ~0.45 | ~0.46 | ~0.73 |
| Random Forest | ~0.74 | ~0.55 | ~0.29 | ~0.40 | ~0.76 |
| **Gradient Boosting** ✓ | ~0.71 | ~0.49 | **~0.51** | **~0.50** | ~0.72 |

---

## 🥇 Modelo Final Escolhido

**GradientBoostingClassifier** (`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`)

**Justificativa:** Embora o Random Forest tenha AUC ligeiramente maior (~0.756 vs ~0.719), o **Recall(bad) do Gradient Boosting é 22 pontos percentuais superior** (0.51 vs 0.29).

No domínio financeiro, o custo de um **Falso Negativo** (aprovar um inadimplente) é significativamente maior que o de um Falso Positivo (rejeitar um bom pagador). O Gradient Boosting minimiza melhor esse risco de negócio.

---

## 📈 Métricas de Avaliação

| Métrica | Valor (teste) | Por quê usada |
|---|---|---|
| **AUC-ROC** | ~0.72 | Métrica principal; robusta ao desbalanceamento |
| Recall(bad) | ~0.51 | Mede capacidade de identificar inadimplentes |
| F1(bad) | ~0.50 | Equilíbrio entre precisão e recall para a classe minoritária |
| Acurácia | ~0.71 | Referência geral (menos informativa com desbalanceamento) |
| AUC-CV | ~0.72 ± σ | Validação cruzada Stratified K-Fold k=5 |

---

## 🗂️ Estrutura dos Arquivos

```
creditai-risco-credito/
│
├── app.py                          # Aplicação Streamlit
├── requirements.txt                # Dependências do projeto
├── README.md                       # Documentação do projeto
│
├── notebooks/
│   └── notebook_atualizado.ipynb   # Notebook revisado (P2)
│
├── model/
│   ├── modelo_final.joblib         # Pipeline completo (preprocessador + GBM)
│   └── feature_info.joblib         # Metadados das features para o app
│
├── reports/
│   └── relatorio_atualizado.pdf    # Relatório final (P2)
│
└── data/
    └── german_credit_data.csv      # Dataset (baixar do Kaggle — link acima)
```

---

## 🛠️ Tecnologias Utilizadas

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| ML | scikit-learn 1.4+ |
| Dados | pandas, numpy |
| Visualização | matplotlib, seaborn |
| Serialização | joblib |
| App | Streamlit 1.32+ |
| Deploy | Streamlit Community Cloud |

---

## ▶️ Como Executar o Notebook

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/creditai-risco-credito.git
cd creditai-risco-credito

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Baixe o dataset (Kaggle) e coloque em data/german_credit_data.csv

# 4. Abra o notebook
jupyter notebook notebooks/notebook_atualizado.ipynb

# 5. Execute todas as células (Kernel > Restart & Run All)
#    O modelo será salvo automaticamente em model/modelo_final.joblib
```

---

## ▶️ Como Executar o App Streamlit Localmente

```bash
# (após executar o notebook e gerar model/modelo_final.joblib)
streamlit run app.py
```

O app abrirá em `http://localhost:8501`.

---

## 🌐 Link do App Publicado

> **[🚀 Acessar o CreditAI no Streamlit Cloud](https://LINK-DO-SEU-APP.streamlit.app)**

*Substitua o link acima pelo URL gerado após o deploy no Streamlit Community Cloud.*

---

## ⚠️ Limitações

- **Dataset histórico e alemão:** Os padrões de crédito da Alemanha nos anos 1990 podem não refletir a realidade brasileira atual.
- **Tamanho reduzido:** 1.000 amostras limitam a robustez estatística do modelo.
- **Desbalanceamento:** 70/30 sem técnicas de rebalanceamento (SMOTE, class_weight) — oportunidade de melhoria.
- **Recall ainda moderado:** ~51% de Recall(bad) significa que ~49% dos inadimplentes ainda são aprovados.
- **Threshold fixo em 0.50:** Em produção, o threshold deve ser calibrado conforme o custo relativo de FN vs FP.
- **Variável `Sex`:** Inclusão acadêmica; uso em produção exige avaliação de impacto diferencial e conformidade com LGPD.

---

## 📝 Conclusão

O projeto demonstra um fluxo completo de Machine Learning aplicado ao domínio de crédito:
análise exploratória, pré-processamento tipado por variável, comparação de três classificadores
com validação cruzada estratificada, e escolha fundamentada em critério de negócio (Recall da
classe de risco). O Gradient Boosting foi selecionado como modelo final por maximizar a
detecção de inadimplentes, mesmo que com pequena perda de AUC. O pipeline completo é
exportado e consumido diretamente pela aplicação Streamlit, garantindo consistência entre
treinamento e inferência.

---

## ⚖️ Nota Ética

Modelos de risco de crédito impactam diretamente o acesso de pessoas a recursos financeiros.
Este projeto inclui discussão sobre viés de gênero, custo assimétrico dos erros, explicabilidade
(LGPD art. 20) e necessidade de monitoramento contínuo. Decisões de crédito reais devem
envolver revisão humana e conformidade regulatória.
