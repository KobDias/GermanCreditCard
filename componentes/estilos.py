import streamlit as st


def aplicar_estilos():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .stApp { background: #0f1117; color: #e8eaf0; }

    .cabecalho {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f1117 100%);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .cabecalho h1 { font-size: 2.2rem; font-weight: 700; color: #e8eaf0; margin: 0 0 0.3rem 0; }
    .cabecalho .subtitulo { color: #8892b0; font-size: 0.95rem; }
    .cabecalho .badge {
        display: inline-block;
        background: #2d3561;
        color: #64ffda;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 0.6rem;
    }

    .secao-titulo {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #64ffda;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e2a3a;
    }

    .resultado-bom {
        background: linear-gradient(135deg, #0d2117, #0a1f15);
        border: 2px solid #2ecc71;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .resultado-ruim {
        background: linear-gradient(135deg, #2d0d0d, #1f0a0a);
        border: 2px solid #e74c3c;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .resultado-icone { font-size: 3rem; margin-bottom: 0.5rem; }
    .resultado-label { font-size: 1.6rem; font-weight: 700; margin: 0.3rem 0; }
    .resultado-prob { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; opacity: 0.85; }
    .resultado-desc { font-size: 0.9rem; opacity: 0.7; margin-top: 0.8rem; line-height: 1.5; }

    .caixa-info {
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
    .caixa-aviso {
        background: #1a150d;
        border: 1px solid #f39c12;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #c9a84c;
    }
    .caixa-erro {
        background: #1a0d0d;
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 0.9rem;
        color: #e87070;
        line-height: 1.6;
    }

    .metrica-pill {
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
    .metrica-pill span { color: #64ffda; font-weight: 600; }

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
        transition: all 0.2s;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #3d4571, #2a3045);
        border-color: #64ffda;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(100, 255, 218, 0.15);
    }

    .stTabs [data-baseweb="tab"] { color: #8892b0; font-family: 'Space Grotesk', sans-serif; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #64ffda !important; }

    .rodape {
        text-align: center;
        color: #3a4060;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1a1f2e;
    }
    </style>
    """, unsafe_allow_html=True)
