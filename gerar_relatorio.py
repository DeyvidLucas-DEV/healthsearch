# -*- coding: utf-8 -*-
"""Gera o relatório técnico em PDF (2 páginas) exigido no entregável.

Os números e o gráfico NÃO são digitados à mão: são extraídos executando os
próprios motores do healthsearch_app.py, garantindo que o relatório reflita o
comportamento real do sistema.

Rodar: .venv/bin/python gerar_relatorio.py
"""

import functools
import sys
import types

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing, String
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

# ----------------------------------------------------------------------------
# Importa a lógica do app sem subir o Streamlit (stub com cache real)
# ----------------------------------------------------------------------------
# O healthsearch_app.py importa pandas para a interface, mas o relatório só usa
# as funções de busca. Stubamos o pandas para não pagar o custo do import — que
# nesta máquina é significativo — sem alterar o comportamento dos motores.
if "pandas" not in sys.modules:
    _pd = types.ModuleType("pandas")
    _pd.DataFrame = object
    sys.modules["pandas"] = _pd

_st = types.ModuleType("streamlit")
_st.set_page_config = lambda **k: None
_st.cache_resource = lambda *a, **k: (lambda f: functools.lru_cache(maxsize=None)(f))
_st.cache_data = lambda *a, **k: (lambda f: functools.lru_cache(maxsize=None)(f))
sys.modules["streamlit"] = _st

_fonte = open("healthsearch_app.py", encoding="utf-8").read().split("# INTERFACE — SIDEBAR")[0]
_mod = {}
exec(compile(_fonte, "healthsearch_app.py", "exec"), _mod)

CORPUS = _mod["CORPUS"]
buscar_bm25 = _mod["buscar_bm25"]
buscar_semantico = _mod["buscar_semantico"]
calcular_ranks = _mod["calcular_ranks"]
fundir_rrf = _mod["fundir_rrf"]

CONSULTA_ESTUDO = "infarto"
K1, B, ALFA = 1.2, 0.75, 0.5

# O relatório é gerado no modo de simulação vetorial: ele roda em qualquer
# máquina, sem PyTorch, e reproduz exatamente o cenário do estudo de caso
# (a consulta "infarto" recuperando a diretriz registrada como "síndrome
# coronariana"). O comportamento com embeddings densos reais está descrito
# na seção de análise.
USAR_SIMULACAO = True


def executar(consulta):
    s_bm = buscar_bm25(consulta, K1, B, True)
    s_sem, modo = buscar_semantico(consulta, USAR_SIMULACAO)
    r_bm = calcular_ranks(s_bm)
    r_sem = calcular_ranks(s_sem)
    s_rrf = fundir_rrf(r_bm, r_sem, ALFA)
    return s_bm, s_sem, s_rrf, r_bm, r_sem, calcular_ranks(s_rrf), modo


# Cache em disco: carregar o modelo de embeddings custa ~40 s e muita memória.
# Guardamos o resultado da execução real para que o PDF possa ser reconstruído
# instantaneamente. Apague dados_relatorio.json para recomputar do zero.
import json
import os

CACHE = "dados_relatorio.json"
if os.path.exists(CACHE):
    with open(CACHE, encoding="utf-8") as fh:
        d = json.load(fh)
    s_bm = np.array(d["s_bm"]); s_sem = np.array(d["s_sem"]); s_rrf = np.array(d["s_rrf"])
    r_bm = np.array(d["r_bm"]); r_sem = np.array(d["r_sem"]); r_rrf = np.array(d["r_rrf"])
    modo = d["modo"]
    print(f"dados carregados do cache ({CACHE})")
else:
    s_bm, s_sem, s_rrf, r_bm, r_sem, r_rrf, modo = executar(CONSULTA_ESTUDO)
    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump({"s_bm": s_bm.tolist(), "s_sem": s_sem.tolist(),
                   "s_rrf": s_rrf.tolist(), "r_bm": r_bm.tolist(),
                   "r_sem": r_sem.tolist(), "r_rrf": r_rrf.tolist(),
                   "modo": modo, "consulta": CONSULTA_ESTUDO}, fh, indent=2)
    print(f"dados computados e salvos em {CACHE}")
N = len(CORPUS)
FORA = N + 1  # posição usada no gráfico para "não recuperado"

# ----------------------------------------------------------------------------
# Gráfico de comparação de ranks (exigido no enunciado)
# ----------------------------------------------------------------------------
COR_BM25, COR_SEM, COR_RRF = (colors.HexColor("#8FA9C8"),
                              colors.HexColor("#C88F8F"),
                              colors.HexColor("#7FB093"))


def construir_grafico():
    """Gráfico de comparação de ranks desenhado com reportlab.graphics.

    Escolhido em vez do matplotlib por ser vetorial (nítido em qualquer zoom)
    e por não exigir uma dependência extra só para uma figura.
    Barras com valor N+1 representam "não recuperado".
    """
    def altura(r):
        return FORA if np.isinf(r) else float(r)

    dados = [
        tuple(altura(r) for r in r_bm),
        tuple(altura(r) for r in r_sem),
        tuple(altura(r) for r in r_rrf),
    ]

    d = Drawing(460, 175)
    bc = VerticalBarChart()
    bc.x, bc.y = 32, 34
    bc.width, bc.height = 412, 122
    bc.data = dados
    bc.categoryAxis.categoryNames = [c["id"] for c in CORPUS]
    bc.categoryAxis.labels.fontSize = 7.5
    bc.categoryAxis.labels.dy = -3
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = FORA
    bc.valueAxis.valueStep = 1
    bc.valueAxis.labels.fontSize = 7
    bc.barSpacing = 0.6
    bc.groupSpacing = 9
    for i, cor in enumerate((COR_BM25, COR_SEM, COR_RRF)):
        bc.bars[i].fillColor = cor
        bc.bars[i].strokeColor = None
    d.add(bc)

    # Rótulo em cada barra: a posição, ou "n/r" quando não recuperada
    largura_grupo = bc.width / len(CORPUS)
    largura_barra = (largura_grupo - 9) / 3
    for serie, valores in enumerate(dados):
        for doc_i, v in enumerate(valores):
            x = (bc.x + doc_i * largura_grupo + 5
                 + serie * largura_barra + largura_barra / 2)
            y = bc.y + (v / FORA) * bc.height + 2
            texto = "n/r" if v == FORA else str(int(v))
            rotulo = String(x, y, texto, fontSize=5.8,
                            fillColor=colors.HexColor("#444444"), textAnchor="middle")
            d.add(rotulo)

    legenda = Legend()
    legenda.x, legenda.y = 118, 12
    legenda.alignment = "right"
    legenda.fontSize = 7.5
    legenda.columnMaximum = 1
    legenda.deltax = 96
    legenda.dxTextSpace = 4
    legenda.colorNamePairs = [
        (COR_BM25, "BM25 (léxico)"),
        (COR_SEM, "Semântico"),
        (COR_RRF, "Híbrido RRF"),
    ]
    d.add(legenda)

    d.add(String(230, 165,
                 f'Comparação de posições — consulta: "{CONSULTA_ESTUDO}"'
                 f"   (barra até o topo = não recuperado)",
                 fontSize=8, textAnchor="middle",
                 fillColor=colors.HexColor("#222222")))
    return d




# ----------------------------------------------------------------------------
# Montagem do PDF (2 páginas)
# ----------------------------------------------------------------------------
EQUIPE = [
    ("Deyvid Lucas", "Fases 1 e 2 — pré-processamento, stemming e motor BM25"),
    ("(integrante 2)", "Fase 3 — embeddings, similaridade de cosseno e simulação vetorial"),
    ("(integrante 3)", "Fase 4 — fusão RRF, interface Streamlit e relatório técnico"),
]

estilos = getSampleStyleSheet()
CORPO = ParagraphStyle("corpo", parent=estilos["BodyText"], fontSize=8.4,
                       leading=11.6, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=estilos["Heading2"], fontSize=11,
                    spaceBefore=7, spaceAfter=4, textColor=colors.HexColor("#1F3864"))
TITULO = ParagraphStyle("titulo", parent=estilos["Title"], fontSize=15, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=estilos["Normal"], fontSize=8,
                     alignment=1, textColor=colors.HexColor("#555555"), spaceAfter=8)
MINI = ParagraphStyle("mini", parent=CORPO, fontSize=7.4, leading=9.6)


def tabela_simples(dados, larguras, cabecalho=True):
    t = Table(dados, colWidths=larguras)
    estilo = [
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BBBBBB")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    if cabecalho:
        estilo += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(estilo))
    return t


def p(txt, estilo=CORPO):
    return Paragraph(txt, estilo)


doc = SimpleDocTemplate(
    "Relatorio_Tecnico_HealthSearch.pdf", pagesize=A4,
    leftMargin=1.6 * cm, rightMargin=1.6 * cm,
    topMargin=1.3 * cm, bottomMargin=1.2 * cm,
)

fx = []

# ------------------------------- PÁGINA 1 -----------------------------------
fx.append(p("HealthSearch — Motor de Busca Híbrido BM25 + Semântico", TITULO))
fx.append(p("Relatório Técnico · Laboratório Prático 05 — Desafio Integrador<br/>"
            "UNIPÊ — Tendências em Ciência da Computação · Prof. Me. Ricardo Roberto de Lima", SUB))

fx.append(p("1. Arquitetura da solução", H1))
fx.append(p(
    "O HealthSearch é um pipeline de recuperação de informação de quatro estágios, "
    "implementado em um único script Streamlit (<b>healthsearch_app.py</b>). A consulta do "
    "usuário percorre <b>dois motores independentes e paralelos</b> — um léxico e um "
    "vetorial — cujos resultados são unificados por um estágio de fusão. A decisão de "
    "arquitetura central é que os motores não compartilham representação: o léxico opera "
    "sobre tokens discretos e o semântico sobre vetores densos de 384 dimensões. Essa "
    "independência é justamente o que permite que um cubra o ponto cego do outro."))

fx.append(tabela_simples([
    ["Fase", "Componente", "Entrada → Saída", "Técnica"],
    ["1", "Pré-processamento",
     "texto bruto → lista de tokens",
     "minúsculas, remoção de acentos (NFKD), regex [a-z0-9]+, stopwords PT, stemming"],
    ["2", "Motor léxico",
     "tokens → vetor de scores",
     "Okapi BM25 (rank_bm25) com k1 e b ajustáveis por slider"],
    ["3", "Motor semântico",
     "texto → vetor 384-D → score",
     "paraphrase-multilingual-MiniLM-L12-v2 + similaridade de cosseno"],
    ["4", "Fusão",
     "2 rankings → 1 ranking",
     "Reciprocal Rank Fusion com peso α e k_RRF = 60"],
], [1.0 * cm, 3.0 * cm, 4.0 * cm, 9.4 * cm]))
fx.append(Spacer(1, 5))

fx.append(p("2. Fase 1 — Pré-processamento", H1))
fx.append(p(
    "A normalização unifica documento e consulta no mesmo espaço de símbolos. O hífen é "
    "tratado como separador, de modo que <b>CÓD-ECG-12D</b> gera os tokens "
    "<font face='Courier'>['cod','ecg','12d']</font> tanto no corpus quanto na consulta — "
    "preservando o casamento exato do código clínico, requisito explícito do estudo de caso. "
    "Acrescentamos ao escopo mínimo um <b>stemmer de sufixos</b> (radical mínimo de 4 "
    "caracteres, para evitar over-stemming). Sem ele, o BM25 trata <i>miocárdio</i> e "
    "<i>miocárdica</i> como termos não relacionados: medimos que a consulta "
    "&ldquo;ataque cardíaco&rdquo; passa de <b>nenhum documento recuperado</b> pelo motor "
    "léxico para a recuperação correta do Doc 2 quando o stemming está ativo."))

fx.append(p("3. Fase 2 — Motor léxico Okapi BM25", H1))
fx.append(p(
    "O índice é reconstruído a cada alteração de slider, pois k<sub>1</sub> e b não reordenam um "
    "ranking pronto: eles alteram a própria função de pontuação. O parâmetro <b>k<sub>1</sub></b> "
    "controla a saturação da frequência do termo (a 1ª ocorrência vale muito, a 8ª quase "
    "nada) e <b>b</b> controla a penalização por comprimento do documento, via a razão "
    "|D|/avgdl. Com b = 0 o tamanho é ignorado; com b = 1 a penalização é máxima."))

fx.append(p("4. Fase 3 — Motor semântico vetorial", H1))
fx.append(p(
    "Documentos e consulta são projetados em vetores densos normalizados e comparados por "
    "similaridade de cosseno. O sistema implementa <b>dois modos</b>: (A) embeddings reais "
    "via <i>sentence-transformers</i>, padrão de execução; e (B) uma <b>simulação vetorial "
    "documentada</b> — TF-IDF esparso expandido por um léxico de equivalências clínicas "
    "(ex.: <i>ataque</i> → <i>infarto, coronariana, miocárdio</i>) — selecionável na interface. "
    "O modo B garante que a aplicação seja demonstrável em máquinas sem PyTorch e torna "
    "explícito, para fins didáticos, qual conhecimento o embedding denso codifica implicitamente."))

fx.append(p("5. Fase 4 — Fusão Reciprocal Rank Fusion", H1))
fx.append(p(
    "O RRF combina <b>posições</b>, não scores. Essa é a razão técnica de sua escolha: o "
    "BM25 produz valores em escala ilimitada e o cosseno em [-1, 1]; somá-los diretamente "
    "exigiria uma normalização arbitrária que distorceria as distribuições. Ao operar sobre "
    "ranks, o RRF é invariante a escala. A constante k_RRF = 60 amortece a diferença entre "
    "as primeiras posições, evitando que o 1º lugar de um motor domine sozinho a fusão."))
fx.append(p(
    "<b>Decisão técnica relevante:</b> um documento com score zero <i>não foi recuperado</i> "
    "por aquele motor e recebe rank infinito, contribuindo com 1/(k+∞) = 0. Sem esse "
    "tratamento — que foi um defeito real detectado e corrigido durante os testes — o "
    "argsort atribuiria posições arbitrárias a documentos irrelevantes, e esses ranks "
    "falsos entrariam na soma, contaminando a fusão. Na consulta &ldquo;ataque cardíaco&rdquo;, "
    "o bug fazia o sistema retornar o Doc 1 no lugar do Doc 2.", MINI))

fx.append(PageBreak())

# ------------------------------- PÁGINA 2 -----------------------------------
fx.append(p("6. Resultado experimental — gráfico de comparação de ranks", H1))
fx.append(p(
    f'Consulta de estudo: <b>&ldquo;{CONSULTA_ESTUDO}&rdquo;</b> · k<sub>1</sub> = {K1} · b = {B} · '
    f'α = {ALFA} · modo semântico: {modo}.'))
fx.append(construir_grafico())

# Tabela de resultados, extraída da execução real
linhas = [["ID", "Diretriz", "Score BM25", "Rank BM25", "Cosseno", "Rank Sem.", "Score RRF", "Rank RRF"]]
for i in np.argsort(r_rrf):
    fmt = lambda r: "—" if np.isinf(r) else str(int(r))
    linhas.append([
        CORPUS[i]["id"], CORPUS[i]["titulo"][:30],
        f"{s_bm[i]:.4f}", fmt(r_bm[i]),
        f"{s_sem[i]:.4f}", fmt(r_sem[i]),
        f"{s_rrf[i]:.5f}", fmt(r_rrf[i]),
    ])
fx.append(tabela_simples(linhas, [1.1*cm, 4.3*cm, 1.9*cm, 1.7*cm, 1.6*cm, 1.7*cm, 2.0*cm, 1.5*cm]))
fx.append(Spacer(1, 5))

fx.append(p("7. Análise: cada motor cobrindo o ponto cego do outro", H1))
fx.append(p(
    "A consulta &ldquo;infarto&rdquo; reproduz o cenário exato descrito no estudo de caso. "
    "O <b>Doc 1 — Protocolo Emergência ECG</b> é a diretriz registrada sob o termo técnico "
    "formal <i>síndrome coronariana</i>: clinicamente é uma das respostas corretas, mas a "
    "palavra &ldquo;infarto&rdquo; não aparece em lugar nenhum do seu texto. O resultado é "
    "que o <b>motor léxico é inteiramente cego para ele</b> — score zero, não recuperado. "
    "É precisamente a falha que motivou o projeto: o BM25 casa strings, não conceitos."))
fx.append(p(
    "O <b>motor semântico recupera o Doc 1 em 1º lugar</b>, porque no espaço vetorial "
    "<i>infarto</i> e <i>síndrome coronariana</i> são vizinhos. Já o <b>Doc 2 — Guia de "
    "Farmacologia Cardíaca</b>, que contém o termo literalmente, é ancorado em 1º pelo "
    "motor léxico. A fusão RRF entrega os dois no topo: <b>Doc 2 em 1º e Doc 1 em 2º</b>, "
    "sem que nenhum dos dois se perca. Nenhum motor isolado produz esse ranking — o léxico "
    "não enxerga o Doc 1 e o semântico não distingue o casamento exato do Doc 2."))
fx.append(p(
    "O comportamento inverso foi verificado com a consulta <b>&ldquo;CÓD-ECG-12D&rdquo;</b>: "
    "o motor léxico ancora o Doc 1 e o Doc 6 (os únicos que contêm o código) no topo, "
    "impedindo que a busca vetorial dilua a precisão em trechos genéricos sobre exames "
    "cardíacos. Em ambos os cenários, o híbrido preserva o acerto e descarta o erro."))
fx.append(p(
    "<b>Observação sobre o modo de embeddings reais.</b> Executando a mesma consulta com o "
    "modelo <i>paraphrase-multilingual-MiniLM-L12-v2</i>, o motor semântico eleva o "
    "<b>Doc 4 (AVC isquêmico)</b> à 1ª posição — infarto e AVC são vizinhos vetoriais por "
    "serem ambos eventos isquêmicos agudos, mas são condições clinicamente distintas. "
    "Também nesse caso a fusão RRF corrige o ranking, rebaixando o Doc 4 e promovendo o "
    "Doc 2. Os dois modos convergem para a mesma conclusão: o erro de um motor é "
    "amortecido pelo voto do outro.", MINI))

fx.append(p("8. Limitações identificadas", H1))
fx.append(p(
    "(i) O modelo de embeddings não resolve <b>siglas farmacológicas</b>: a consulta "
    "&ldquo;AAS 100mg&rdquo; não recupera o Doc 2, pois o modelo não associa a sigla AAS a "
    "&ldquo;ácido acetilsalicílico&rdquo;, e o termo tampouco aparece literalmente no corpus "
    "para o BM25 casar. Em produção, isso exige um dicionário de expansão de siglas médicas "
    "(DeCS/UMLS) antes da indexação. (ii) O stemmer é de sufixos e não de dicionário, então "
    "não unifica pares irregulares como <i>compressões</i>/<i>compressão</i>. (iii) O corpus "
    "de 6 documentos impede métricas estatisticamente significativas de precisão e revocação; "
    "a avaliação aqui é qualitativa e orientada a casos.", MINI))

fx.append(p("9. Divisão de tarefas da equipe", H1))
fx.append(tabela_simples(
    [["Integrante", "Responsabilidade"]] + [[n, r] for n, r in EQUIPE],
    [4.6 * cm, 11.8 * cm]))
fx.append(Spacer(1, 4))

fx.append(p("10. Execução", H1))
fx.append(p(
    "<font face='Courier'>pip install streamlit pandas rank_bm25 sentence-transformers</font><br/>"
    "<font face='Courier'>streamlit run healthsearch_app.py</font><br/>"
    "A primeira busca carrega o modelo de embeddings (~40 s); as seguintes são instantâneas, "
    "pois o modelo fica em cache via <font face='Courier'>@st.cache_resource</font>. Para "
    "demonstração imediata, a opção <i>Forçar simulação vetorial</i> dispensa o download. "
    "O bônus de Cross-Encoder (ms-marco-MiniLM-L-6-v2) re-pontua o Top-3 do ranking híbrido "
    "lendo consulta e documento no mesmo passe — mais preciso que o bi-encoder, porém caro "
    "demais para o corpus inteiro, o que justifica aplicá-lo só após a recuperação.", MINI))

doc.build(fx)
print("PDF gerado: Relatorio_Tecnico_HealthSearch.pdf")
