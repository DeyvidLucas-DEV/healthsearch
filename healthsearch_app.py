# -*- coding: utf-8 -*-
"""
HealthSearch — Motor de Busca Híbrido (BM25 + Semântico + RRF)
================================================================================
Desafio Integrador — Laboratório Prático 05
UNIPÊ — Centro Universitário de João Pessoa
Disciplina: Tendências em Ciência da Computação (Recuperação de Informação / PLN)
Professor: Me. Ricardo Roberto de Lima

Execução:
    streamlit run healthsearch_app.py

Arquitetura (4 fases exigidas no enunciado):
    Fase 1 — Ingestão do corpus médico e pré-processamento (tokenização/stopwords)
    Fase 2 — Motor léxico Okapi BM25 com k1 e b interativos
    Fase 3 — Motor semântico vetorial (embeddings + similaridade de cosseno)
    Fase 4 — Fusão Reciprocal Rank Fusion (RRF) com peso alfa
    Bônus  — Re-ranking com Cross-Encoder sobre o Top-3 híbrido
================================================================================
"""

import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(page_title="HealthSearch", page_icon="🏥", layout="wide")

K_RRF = 60  # constante de suavização de posição, fixada pelo enunciado


# ==============================================================================
# FASE 1 — INGESTÃO DO CORPUS MÉDICO
# Base hardcode obrigatória (seção 5 do enunciado): 6 diretrizes clínicas.
# ==============================================================================
CORPUS = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "texto": (
            "Pacientes com dor precordial aguda e suspeita de síndrome coronariana "
            "devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."
        ),
    },
    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "texto": (
            "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários "
            "reduz a mortalidade no infarto agudo do miocárdio."
        ),
    },
    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "texto": (
            "A crise hipertensiva severa requer administração de anti-hipertensivos "
            "venosos e monitoramento contínuo da pressão arterial na UTI."
        ),
    },
    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "texto": (
            "O acidente vascular cerebral isquêmico agudo deve ser tratado com "
            "trombolíticos venosos em até quatro horas e meia do início dos sintomas."
        ),
    },
    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "texto": (
            "Parada cardiorrespiratória em adultos exige compressões torácicas "
            "contínuas de alta qualidade e desfibrilação precoce no código azul."
        ),
    },
    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "texto": (
            "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, "
            "recomenda-se a monitorização cardíaca contínua por telemetria."
        ),
    },
]

# Campo efetivamente indexado: título + trecho clínico (prática padrão em RI,
# aumenta o recall sem alterar o conteúdo obrigatório do corpus).
for _d in CORPUS:
    _d["indexavel"] = f"{_d['titulo']} {_d['texto']}"


# ------------------------------------------------------------------------------
# FASE 1 — PRÉ-PROCESSAMENTO
# ------------------------------------------------------------------------------
STOPWORDS_PT = {
    "a", "à", "ao", "aos", "as", "às", "com", "como", "da", "das", "de", "do",
    "dos", "e", "em", "entre", "essa", "esse", "esta", "este", "eu", "foi", "for",
    "há", "isso", "já", "la", "lhe", "mais", "mas", "me", "mesmo", "meu", "muito",
    "na", "nas", "não", "no", "nos", "num", "numa", "o", "os", "ou", "para", "pela",
    "pelo", "por", "qual", "quando", "que", "se", "sem", "ser", "seu", "sua", "só",
    "também", "te", "tem", "um", "uma", "você", "ate", "até", "deve", "devem",
}


def normalizar(texto: str) -> str:
    """Minúsculas + remoção de acentos (NFKD). 'CÓD-ECG' -> 'cod-ecg'."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


# Sufixos removidos pelo stemmer, do mais longo para o mais curto (a ordem
# importa: "-icas" precisa ser testado antes de "-as").
SUFIXOS_PT = (
    "issimo", "issima", "mente", "acoes", "icoes", "encia", "ancia",
    "icas", "icos", "ica", "ico", "cao", "oes", "ais", "eis", "res",
    "ada", "ado", "ida", "ido", "iva", "ivo", "osa", "oso", "ura",
    "ias", "ios", "ia", "io", "es", "as", "os", "a", "o", "e",
)
TAMANHO_MINIMO_RADICAL = 4


def radicalizar(token: str) -> str:
    """Stemming leve por remoção de sufixo (inspirado no RSLP).

    EXTENSÃO da Fase 1: sem isso o BM25 trata 'miocárdio' e 'miocárdica'
    como termos não relacionados, e a consulta 'isquemia miocárdica' não
    recupera a diretriz sobre 'infarto agudo do miocárdio'. O radical mínimo
    de 4 caracteres evita over-stemming (colapsar palavras distintas).
    """
    if len(token) <= TAMANHO_MINIMO_RADICAL or token.isdigit():
        return token
    for sufixo in SUFIXOS_PT:
        if token.endswith(sufixo) and len(token) - len(sufixo) >= TAMANHO_MINIMO_RADICAL:
            return token[: -len(sufixo)]
    return token


def tokenizar(texto: str, stemming: bool = False) -> list:
    """Fase 1: minúsculas, sem acento, sem caractere especial, sem stopwords.

    O hífen é tratado como separador, então 'CÓD-ECG-12D' vira
    ['cod', 'ecg', '12d'] tanto no documento quanto na consulta — o que preserva
    o casamento exato do código clínico no motor léxico.
    """
    texto = normalizar(texto)
    tokens = re.findall(r"[a-z0-9]+", texto)
    tokens = [t for t in tokens if t not in STOPWORDS_PT and len(t) > 1]
    return [radicalizar(t) for t in tokens] if stemming else tokens


# ==============================================================================
# FASE 2 — MOTOR LÉXICO: OKAPI BM25
# ==============================================================================
def buscar_bm25(consulta: str, k1: float, b: float, stemming: bool = False) -> np.ndarray:
    """Retorna o vetor de scores BM25 (um score por documento).

    O índice é reconstruído a cada chamada porque k1 e b são controlados por
    slider — mudá-los altera a própria função de ranking, não só a ordenação.
    """
    corpus_tokens = [tokenizar(d["indexavel"], stemming) for d in CORPUS]
    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    return np.array(bm25.get_scores(tokenizar(consulta, stemming)))


# ==============================================================================
# FASE 3 — MOTOR SEMÂNTICO VETORIAL
# ==============================================================================
# Estratégia em dois níveis, ambos previstos no enunciado:
#   (A) Embeddings densos reais via sentence-transformers, quando disponível.
#   (B) Simulação vetorial documentada, caso o modelo não possa ser carregado.
#
# A simulação (B) constrói um vetor esparso TF-IDF e o EXPANDE com um léxico de
# equivalências médicas (ex.: "ataque cardiaco" -> "infarto miocardio"). É essa
# expansão que reproduz o comportamento essencial do embedding para a prova de
# conceito: recuperar o documento certo mesmo quando a consulta usa o termo
# leigo e o documento usa o termo técnico formal.
# ------------------------------------------------------------------------------
MODELO_ST = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Léxico de equivalências clínicas usado APENAS pela simulação (modo B).
SINONIMOS_CLINICOS = {
    "ataque": ["infarto", "coronariana", "miocardio"],
    "cardiaco": ["coronariana", "miocardio", "cardiaca", "precordial"],
    "coracao": ["cardiaca", "miocardio", "cardiorrespiratoria", "precordial"],
    "infarto": ["coronariana", "miocardio", "precordial", "sindrome"],
    "derrame": ["acidente", "vascular", "cerebral", "isquemico", "avc"],
    "avc": ["acidente", "vascular", "cerebral", "isquemico"],
    "trombose": ["tromboliticos", "isquemico", "vascular"],
    "pressao": ["hipertensiva", "hipertensivos", "arterial"],
    "alta": ["hipertensiva", "severa"],
    "hipertensao": ["hipertensiva", "hipertensivos", "arterial"],
    "parada": ["cardiorrespiratoria", "reanimacao", "desfibrilacao", "rcr"],
    "reanimacao": ["cardiorrespiratoria", "compressoes", "desfibrilacao"],
    "aas": ["acetilsalicilico", "acido", "antiagregantes"],
    "aspirina": ["acetilsalicilico", "acido", "antiagregantes"],
    "remedio": ["farmacologia", "acetilsalicilico", "antiagregantes"],
    "exame": ["eletrocardiograma", "ecg", "diagnostico", "monitorizacao"],
    "ecg": ["eletrocardiograma", "cardiaca", "arritmias"],
    "eletrocardiograma": ["ecg", "arritmias", "precordial"],
    "peito": ["precordial", "toracicas", "dor"],
    "dor": ["precordial", "aguda"],
    "emergencia": ["aguda", "urgencia", "precoce", "imediato"],
    "uti": ["monitoramento", "telemetria", "continua", "intensiva"],
    "monitoramento": ["monitorizacao", "telemetria", "continuo"],
    "arritmia": ["arritmias", "cardiaca", "telemetria"],
}


@st.cache_resource(show_spinner=False)
def carregar_modelo_semantico():
    """Tenta carregar o bi-encoder real. Retorna None se indisponível."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODELO_ST)
    except Exception:
        return None


def _expandir(tokens: list) -> list:
    """Modo B: acrescenta os equivalentes clínicos de cada token da consulta."""
    expandido = list(tokens)
    for t in tokens:
        expandido.extend(SINONIMOS_CLINICOS.get(t, []))
    return expandido


def _vetorizar_simulado(textos: list, vocabulario: list) -> np.ndarray:
    """Modo B: matriz TF-IDF densa, uma linha por texto."""
    idx = {termo: i for i, termo in enumerate(vocabulario)}
    n_docs = len(CORPUS)
    # IDF calculado sobre o corpus (constante entre chamadas)
    docs_tokens = [set(tokenizar(d["indexavel"])) for d in CORPUS]
    idf = np.zeros(len(vocabulario))
    for termo, i in idx.items():
        df = sum(1 for s in docs_tokens if termo in s)
        idf[i] = np.log((n_docs + 1) / (df + 1)) + 1.0

    matriz = np.zeros((len(textos), len(vocabulario)))
    for linha, tokens in enumerate(textos):
        for t in tokens:
            if t in idx:
                matriz[linha, idx[t]] += 1.0
        matriz[linha] *= idf
    return matriz


def similaridade_cosseno(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """cos(a, b) = (a · b) / (||a|| · ||b||), para a contra cada linha de B."""
    na = np.linalg.norm(a)
    nB = np.linalg.norm(B, axis=1)
    denom = na * nB
    denom[denom == 0] = 1e-9
    return (B @ a) / denom


@st.cache_data(show_spinner=False)
def _embeddings_corpus_reais():
    modelo = carregar_modelo_semantico()
    if modelo is None:
        return None
    return modelo.encode([d["indexavel"] for d in CORPUS], normalize_embeddings=True)


def buscar_semantico(consulta: str, forcar_simulacao: bool = False):
    """Retorna (scores_cosseno, rotulo_do_modo_utilizado)."""
    if not forcar_simulacao:
        modelo = carregar_modelo_semantico()
        emb_corpus = _embeddings_corpus_reais()
        if modelo is not None and emb_corpus is not None:
            emb_q = modelo.encode(consulta, normalize_embeddings=True)
            return similaridade_cosseno(np.asarray(emb_q), np.asarray(emb_corpus)), "Embeddings densos reais"

    # Modo B — simulação vetorial documentada
    docs_tokens = [tokenizar(d["indexavel"]) for d in CORPUS]
    vocabulario = sorted({t for toks in docs_tokens for t in toks})
    M_docs = _vetorizar_simulado(docs_tokens, vocabulario)
    v_query = _vetorizar_simulado([_expandir(tokenizar(consulta))], vocabulario)[0]
    return similaridade_cosseno(v_query, M_docs), "Simulação vetorial (TF-IDF + expansão clínica)"


# ==============================================================================
# FASE 4 — FUSÃO RRF (RECIPROCAL RANK FUSION)
# ==============================================================================
def calcular_ranks(scores: np.ndarray) -> np.ndarray:
    """Converte scores em posições 1, 2, 3... (maior score = rank 1).

    Documento com score <= 0 NÃO foi recuperado por aquele motor e recebe
    rank infinito. Isso é essencial para a Fase 4: sem esse tratamento, um
    documento irrelevante ganharia uma posição arbitrária no ranking léxico
    e essa posição falsa entraria na soma do RRF, contaminando a fusão.
    Com rank infinito, o termo vira 1/(k_RRF + inf) = 0 e o motor que nada
    encontrou simplesmente não vota.
    """
    ranks = np.full(len(scores), np.inf, dtype=float)
    validos = np.where(scores > 0)[0]
    if validos.size == 0:
        return ranks
    ordem = validos[np.argsort(-scores[validos], kind="stable")]
    for posicao, indice in enumerate(ordem, start=1):
        ranks[indice] = float(posicao)
    return ranks


def formatar_rank(r: float) -> str:
    """Exibe rank infinito como travessão na interface."""
    return "—" if np.isinf(r) else str(int(r))


def fundir_rrf(rank_bm25: np.ndarray, rank_sem: np.ndarray, alfa: float,
               k_rrf: int = K_RRF) -> np.ndarray:
    """Score_RRF(D) = α·[1/(k + Rank_BM25)] + (1-α)·[1/(k + Rank_Semântico)]

    O RRF combina POSIÇÕES, não scores. Por isso dispensa normalizar as escalas
    heterogêneas do BM25 (ilimitada) e do cosseno (-1 a 1).
    """
    return alfa * (1.0 / (k_rrf + rank_bm25)) + (1.0 - alfa) * (1.0 / (k_rrf + rank_sem))


# ==============================================================================
# BÔNUS — CROSS-ENCODER RE-RANKING
# ==============================================================================
MODELO_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@st.cache_resource(show_spinner=False)
def carregar_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(MODELO_CE)
    except Exception:
        return None


def rerank_cross_encoder(consulta: str, indices_top: list):
    """Re-pontua os Top-N candidatos lendo consulta e documento JUNTOS.

    Diferença conceitual para o bi-encoder da Fase 3: lá, consulta e documento
    viram vetores separados e só depois são comparados. Aqui o par entra junto
    no modelo, que devolve uma nota de relevância direta — mais preciso, porém
    caro demais para rodar sobre o corpus inteiro. Daí a ordem: recupera com
    RRF, depois refina o topo.
    """
    modelo = carregar_cross_encoder()
    if modelo is None:
        return None
    pares = [[consulta, CORPUS[i]["indexavel"]] for i in indices_top]
    return np.asarray(modelo.predict(pares), dtype=float)


# ==============================================================================
# INTERFACE — SIDEBAR DE CALIBRAÇÃO
# ==============================================================================
st.title("🏥 HealthSearch — Motor de Busca Híbrido")
st.caption(
    "BM25 (léxico) + Busca Semântica Vetorial + Reciprocal Rank Fusion · "
    "UNIPÊ · Tendências em Ciência da Computação"
)

with st.sidebar:
    st.header("⚙️ Calibração")

    st.subheader("Fase 2 — BM25")
    k1 = st.slider("k₁ — saturação de frequência", 0.0, 3.0, 1.2, 0.1)
    b = st.slider("b — normalização por comprimento", 0.0, 1.0, 0.75, 0.05)

    st.subheader("Fase 4 — Fusão RRF")
    alfa = st.slider("α — peso do BM25 na fusão", 0.0, 1.0, 0.5, 0.05)
    st.caption(f"α = {alfa:.2f} → {alfa:.0%} léxico / {1 - alfa:.0%} semântico · k_RRF = {K_RRF}")

    st.subheader("Fase 1 — Pré-processamento")
    usar_stem = st.checkbox(
        "Aplicar stemming (radicalização)", value=True,
        help="Reduz 'miocárdica' e 'miocárdio' ao radical 'miocard', permitindo "
             "que o BM25 case variações morfológicas da mesma palavra.",
    )

    st.subheader("Fase 3 — Motor semântico")
    forcar_sim = st.checkbox(
        "Forçar simulação vetorial", value=False,
        help="Ignora o modelo de embeddings e usa TF-IDF com expansão clínica.",
    )

    st.subheader("🎁 Bônus")
    usar_ce = st.checkbox(
        "Cross-Encoder re-ranking (Top-3)", value=False,
        help="Re-ordena os 3 primeiros do ranking híbrido com ms-marco-MiniLM-L-6-v2.",
    )

    st.divider()
    st.caption("**Consultas de demonstração**")
    st.code(
        "ataque cardíaco     → sinônimo leigo\n"
        "infarto             → termo clínico\n"
        "CÓD-ECG-12D         → código exato\n"
        "derrame cerebral    → sinônimo de AVC\n"
        "parada do coração   → sinônimo de PCR\n"
        "AAS 100mg           → sigla + dosagem",
        language=None,
    )

# ==============================================================================
# CONSULTA E EXECUÇÃO DOS TRÊS MOTORES
# ==============================================================================
consulta = st.text_input("🔍 Consulta clínica:", "ataque cardíaco")

if not consulta.strip():
    st.info("Digite uma consulta para iniciar a busca.")
    st.stop()

tokens_query = tokenizar(consulta, usar_stem)
if not tokens_query:
    st.warning("A consulta só contém stopwords ou caracteres descartados.")
    st.stop()

# A primeira execução paga o custo de importar o PyTorch e carregar o modelo
# (~40 s nesta máquina). Depois disso, @st.cache_resource mantém tudo em
# memória e as consultas seguintes respondem instantaneamente.
if "ja_aqueceu" not in st.session_state:
    st.session_state["ja_aqueceu"] = True
    if not forcar_sim:
        st.info(
            "⏳ **Primeira busca:** carregando o modelo de embeddings "
            "(~40 s só desta vez). As consultas seguintes são instantâneas. "
            "Para uma demonstração imediata, marque *Forçar simulação vetorial* "
            "na barra lateral."
        )

with st.spinner("Executando os três motores..."):
    scores_bm25 = buscar_bm25(consulta, k1, b, usar_stem)
    scores_sem, modo_semantico = buscar_semantico(consulta, forcar_sim)
    rank_bm25 = calcular_ranks(scores_bm25)
    rank_sem = calcular_ranks(scores_sem)
    scores_rrf = fundir_rrf(rank_bm25, rank_sem, alfa)
    rank_rrf = calcular_ranks(scores_rrf)

st.caption(f"Tokens da consulta: `{tokens_query}`  ·  Motor semântico: **{modo_semantico}**")

base = pd.DataFrame({
    "ID": [d["id"] for d in CORPUS],
    "Título": [d["titulo"] for d in CORPUS],
    "Score BM25": scores_bm25.round(4),
    "Rank BM25": rank_bm25,
    "Cosseno": scores_sem.round(4),
    "Rank Semântico": rank_sem,
    "Score RRF": scores_rrf.round(6),
    "Rank RRF": rank_rrf,
})

COLS_RANK = ["Rank BM25", "Rank Semântico", "Rank RRF"]


def tabela(df: pd.DataFrame, colunas: list, ordenar_por: str) -> pd.DataFrame:
    """Ordena pelo rank numérico e só então formata inf como travessão."""
    saida = df.sort_values(ordenar_por)[colunas].copy()
    for c in saida.columns:
        if c in COLS_RANK:
            saida[c] = saida[c].map(formatar_rank)
    return saida


def cartao_resultado(i: int, rotulo: str, valor: str, posicao: int):
    """Renderiza um documento do ranking com o trecho clínico destacado."""
    doc = CORPUS[i]
    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{posicao}º · {doc['id']} — {doc['titulo']}**")
        c2.metric(rotulo, valor)
        st.write(doc["texto"])


abas = st.tabs([
    "① Léxico (BM25)",
    "② Semântico (Cosseno)",
    "③ Híbrido (RRF)",
    "④ Matriz Comparativa",
])

# ------------------------------------------------------------------ ABA 1
with abas[0]:
    st.subheader("Motor Léxico — Okapi BM25")
    st.latex(r"score(D,Q)=\sum_{t \in Q} IDF(t)\cdot"
             r"\frac{f(t,D)\cdot(k_1+1)}{f(t,D)+k_1\cdot(1-b+b\cdot\frac{|D|}{avgdl})}")
    st.caption(f"Parâmetros ativos: k₁ = {k1} · b = {b}")

    st.dataframe(
        tabela(base, ["ID", "Título", "Score BM25", "Rank BM25"], "Rank BM25"),
        use_container_width=True, hide_index=True,
    )

    if scores_bm25.max() <= 0:
        st.error(
            "⚠️ **Ponto cego léxico:** nenhum documento contém os termos exatos da "
            "consulta. O BM25 só casa strings — sinônimo clínico ele não alcança."
        )
    else:
        for pos, i in enumerate(np.argsort(-scores_bm25)[:3], 1):
            if scores_bm25[i] > 0:
                cartao_resultado(i, "BM25", f"{scores_bm25[i]:.3f}", pos)

# ------------------------------------------------------------------ ABA 2
with abas[1]:
    st.subheader("Motor Semântico — Similaridade de Cosseno")
    st.latex(r"\cos(\vec{q},\vec{d})=\frac{\vec{q}\cdot\vec{d}}"
             r"{\|\vec{q}\|\;\|\vec{d}\|}")
    st.caption(f"Modo em uso: {modo_semantico}")

    st.dataframe(
        tabela(base, ["ID", "Título", "Cosseno", "Rank Semântico"], "Rank Semântico"),
        use_container_width=True, hide_index=True,
    )

    for pos, i in enumerate(np.argsort(-scores_sem)[:3], 1):
        if scores_sem[i] > 0:
            cartao_resultado(i, "Cosseno", f"{scores_sem[i]:.3f}", pos)

    st.info(
        "⚠️ **Ponto cego semântico:** o vetor captura o *sentido* geral, então "
        "códigos e dosagens exatas (CÓD-ECG-12D, AAS 100mg) se diluem em trechos "
        "genéricos sobre exames ou medicamentos."
    )

# ------------------------------------------------------------------ ABA 3
with abas[2]:
    st.subheader("Motor Híbrido — Reciprocal Rank Fusion")
    st.latex(r"Score_{RRF}(D)=\alpha\cdot\frac{1}{k_{RRF}+Rank_{BM25}(D)}"
             r"+(1-\alpha)\cdot\frac{1}{k_{RRF}+Rank_{Sem}(D)}")
    st.caption(f"α = {alfa:.2f} · k_RRF = {K_RRF}")

    st.dataframe(
        tabela(base, ["ID", "Título", "Rank BM25", "Rank Semântico", "Score RRF",
                      "Rank RRF"], "Rank RRF"),
        use_container_width=True, hide_index=True,
    )
    st.caption("Travessão (—) = documento não recuperado por aquele motor; "
               "sua contribuição no RRF é zero.")

    ordem_rrf = list(np.argsort(-scores_rrf))
    top3 = ordem_rrf[:3]

    if usar_ce:
        with st.spinner("Aplicando Cross-Encoder sobre o Top-3..."):
            notas_ce = rerank_cross_encoder(consulta, top3)
        if notas_ce is None:
            st.warning(
                "Cross-Encoder indisponível (sentence-transformers não instalado). "
                "Exibindo o ranking RRF puro."
            )
        else:
            st.markdown("#### 🎁 Top-3 re-ordenado pelo Cross-Encoder")
            comparativo = pd.DataFrame({
                "ID": [CORPUS[i]["id"] for i in top3],
                "Título": [CORPUS[i]["titulo"] for i in top3],
                "Posição RRF": list(range(1, len(top3) + 1)),
                "Nota Cross-Encoder": notas_ce.round(4),
            })
            nova_ordem = np.argsort(-notas_ce)
            comparativo["Posição Final"] = [
                int(np.where(nova_ordem == i)[0][0]) + 1 for i in range(len(top3))
            ]
            st.dataframe(
                comparativo.sort_values("Posição Final"),
                use_container_width=True, hide_index=True,
            )
            trocou = comparativo["Posição RRF"].tolist() != comparativo.sort_values(
                "Posição Final")["Posição RRF"].tolist()
            st.success("O Cross-Encoder alterou a ordem do Top-3." if trocou
                       else "O Cross-Encoder confirmou a ordem do RRF.")
            top3 = [top3[i] for i in nova_ordem]

    st.markdown("#### Resultado final")
    for pos, i in enumerate(top3, 1):
        cartao_resultado(i, "RRF", f"{scores_rrf[i]:.5f}", pos)

# ------------------------------------------------------------------ ABA 4
with abas[3]:
    st.subheader("Matriz Comparativa dos Três Motores")
    st.dataframe(
        tabela(base, list(base.columns), "Rank RRF"),
        use_container_width=True, hide_index=True,
    )

    st.markdown("#### Comparação de posições (quanto menor a barra, melhor)")
    # Rank infinito é plotado como (N+1) para caber na escala do gráfico.
    grafico = base.set_index("ID")[COLS_RANK].replace(np.inf, len(CORPUS) + 1)
    st.bar_chart(grafico, height=320)
    st.caption(f"Barras com valor {len(CORPUS) + 1} representam 'não recuperado'.")

    st.markdown("#### Métricas de desempenho da busca")
    def primeiro(coluna: str) -> str:
        achados = base.loc[base[coluna] == 1, "ID"]
        return achados.iloc[0] if len(achados) else "—"

    vencedor_bm25 = primeiro("Rank BM25")
    vencedor_sem = primeiro("Rank Semântico")
    vencedor_rrf = primeiro("Rank RRF")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("1º Léxico", vencedor_bm25)
    m2.metric("1º Semântico", vencedor_sem)
    m3.metric("1º Híbrido", vencedor_rrf)
    m4.metric("Docs recuperados", int((scores_bm25 > 0).sum()),
              delta=f"{int((scores_sem > 0).sum())} semântico")

    # Correlação calculada só sobre documentos recuperados pelos DOIS motores.
    mutuos = np.isfinite(rank_bm25) & np.isfinite(rank_sem)
    if mutuos.sum() > 1 and np.std(rank_bm25[mutuos]) > 0 and np.std(rank_sem[mutuos]) > 0:
        concordancia = float(np.corrcoef(rank_bm25[mutuos], rank_sem[mutuos])[0, 1])
        st.metric("Correlação entre os rankings léxico e semântico", f"{concordancia:.3f}")
    else:
        st.metric("Correlação entre os rankings léxico e semântico", "n/d")
    st.caption(
        "Correlação próxima de 1 indica que os dois motores concordam e a fusão "
        "muda pouco. Correlação baixa ou negativa indica visões complementares — "
        "exatamente o cenário em que o RRF agrega valor."
    )

    if vencedor_bm25 != vencedor_sem:
        st.success(
            f"✅ **Divergência detectada:** o léxico elegeu {vencedor_bm25} e o "
            f"semântico elegeu {vencedor_sem}. O RRF arbitrou em favor de "
            f"{vencedor_rrf} combinando as duas posições."
        )
    else:
        st.info(f"Ambos os motores concordam em {vencedor_bm25}; o RRF confirma o resultado.")

st.divider()
st.caption(
    "HealthSearch · Laboratório Prático 05 — Desafio Integrador · "
    "Fases: ① Pré-processamento ② BM25 ③ Vetorial ④ RRF · Bônus: Cross-Encoder"
)
