# 🏥 HealthSearch — Motor de Busca Híbrido (BM25 + Semântico + RRF)

Protótipo de motor de busca para um repositório de diretrizes clínicas, que combina
**busca léxica (Okapi BM25)** com **busca semântica vetorial (embeddings)** através de
**Reciprocal Rank Fusion (RRF)**.

> **Laboratório Prático 05 — Desafio Integrador**
> UNIPÊ — Centro Universitário de João Pessoa
> Tendências em Ciência da Computação · Recuperação de Informação / PLN
> Prof. Me. Ricardo Roberto de Lima

---

## O problema

Motores de busca puros falham de duas formas opostas em um corpus médico:

| Abordagem | Onde falha | Exemplo |
|---|---|---|
| **Léxica pura** (BM25) | Não alcança sinônimos | Buscar `ataque cardíaco` não recupera a diretriz registrada como *síndrome coronariana aguda* |
| **Semântica pura** (embeddings) | Dilui termos exatos | Buscar `CÓD-ECG-12D` traz trechos genéricos sobre exames cardíacos |

O HealthSearch executa os dois motores em paralelo e funde os rankings, ficando com o
acerto de cada um.

## Demonstração do argumento

Consulta `infarto`, medida com embeddings reais:

| Documento | Rank BM25 | Rank Semântico | Rank Híbrido |
|---|---|---|---|
| **Doc 2** — Farmacologia Cardíaca (*infarto do miocárdio*) | 1 | 2 | **1** ✅ |
| Doc 4 — AVC Isquêmico | não recuperado | **1** ❌ | 2 |

O motor semântico **erra**: coloca o documento de AVC em primeiro, porque *infarto* e *AVC*
são vizinhos no espaço vetorial (ambos eventos isquêmicos agudos). O motor léxico acerta o
documento, mas é cego para os outros quatro. **A fusão RRF corrige o erro** e mantém a
cobertura.

## Arquitetura

```
consulta
   │
   ├─► Fase 1: pré-processamento (minúsculas, sem acento, stopwords PT, stemming)
   │
   ├─► Fase 2: BM25 ──────────► ranking léxico    ─┐
   │      k₁ = saturação                            │
   │      b  = normalização por tamanho             ├─► Fase 4: RRF ─► ranking final
   │                                                │      α = peso
   └─► Fase 3: embeddings ───► ranking semântico ──┘      k_RRF = 60
          similaridade de cosseno
```

**Fórmula da fusão:**

```
Score_RRF(D) = α · 1/(k_RRF + Rank_BM25(D)) + (1−α) · 1/(k_RRF + Rank_Semântico(D))
```

O RRF combina *posições*, não *scores* — por isso dispensa normalizar as escalas
heterogêneas do BM25 (ilimitada) e do cosseno ([−1, 1]).

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run healthsearch_app.py
```

A primeira busca carrega o modelo de embeddings (~40 s); as seguintes são instantâneas.
Para demonstração imediata, marque **Forçar simulação vetorial** na barra lateral — o app
usa TF-IDF com expansão clínica e dispensa o PyTorch.

## Interface

Barra lateral com os controles de calibração exigidos:

- **k₁** (0.0 – 3.0, padrão 1.2) — saturação de frequência do termo
- **b** (0.0 – 1.0, padrão 0.75) — normalização pelo comprimento do documento
- **α** (0.0 – 1.0, padrão 0.5) — peso do léxico na fusão
- **Stemming** — radicalização (liga/desliga)
- **Cross-Encoder** — re-ranking bônus do Top-3

E quatro abas: `① Léxico` · `② Semântico` · `③ Híbrido RRF` · `④ Matriz Comparativa`.

## Arquivos

| Arquivo | Função |
|---|---|
| `healthsearch_app.py` | **Entregável principal** — aplicação Streamlit completa e auto-contida |
| `calcular_dados.py` | Executa os motores e salva os resultados em JSON |
| `gerar_relatorio.py` | Gera o relatório técnico em PDF e o gráfico de ranks |
| `Relatorio_Tecnico_HealthSearch.pdf` | Relatório técnico (2 páginas) |
| `grafico_ranks.png` | Gráfico de comparação de posições |

## Detalhes de implementação

**Rank infinito para não recuperados.** Documento com score zero não foi recuperado por
aquele motor e recebe rank infinito, contribuindo com `1/(k+∞) = 0`. Sem isso, o `argsort`
atribui posições arbitrárias a documentos irrelevantes e esses ranks falsos contaminam a
fusão — um defeito real detectado e corrigido durante os testes.

**Stemming.** Sem radicalização, o BM25 trata `miocárdio` e `miocárdica` como termos não
relacionados. O stemmer usa remoção de sufixos com radical mínimo de 4 caracteres para
evitar over-stemming.

**Dois modos semânticos.** Embeddings reais (`paraphrase-multilingual-MiniLM-L12-v2`) por
padrão, com queda automática para simulação vetorial documentada quando o PyTorch não está
disponível.
