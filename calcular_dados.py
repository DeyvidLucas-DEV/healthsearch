# -*- coding: utf-8 -*-
"""Executa os motores uma única vez e salva os resultados em JSON.

Separado do gerar_relatorio.py de propósito: este script carrega o PyTorch
(caro em memória) e NADA mais. A renderização do PDF lê o JSON e roda leve.
"""
import functools
import json
import sys
import types

import numpy as np

_st = types.ModuleType("streamlit")
_st.set_page_config = lambda **k: None
_st.cache_resource = lambda *a, **k: (lambda f: functools.lru_cache(maxsize=None)(f))
_st.cache_data = lambda *a, **k: (lambda f: functools.lru_cache(maxsize=None)(f))
sys.modules["streamlit"] = _st

fonte = open("healthsearch_app.py", encoding="utf-8").read().split("# INTERFACE — SIDEBAR")[0]
mod = {}
exec(compile(fonte, "healthsearch_app.py", "exec"), mod)
print("modulo carregado", flush=True)

CONSULTA, K1, B, ALFA = "infarto", 1.2, 0.75, 0.5
s_bm = mod["buscar_bm25"](CONSULTA, K1, B, True)
print("bm25 ok", flush=True)
s_sem, modo = mod["buscar_semantico"](CONSULTA, False)
print("semantico ok:", modo, flush=True)
r_bm = mod["calcular_ranks"](s_bm)
r_sem = mod["calcular_ranks"](s_sem)
s_rrf = mod["fundir_rrf"](r_bm, r_sem, ALFA)
r_rrf = mod["calcular_ranks"](s_rrf)

with open("dados_relatorio.json", "w", encoding="utf-8") as fh:
    json.dump({"consulta": CONSULTA, "modo": modo, "k1": K1, "b": B, "alfa": ALFA,
               "s_bm": s_bm.tolist(), "s_sem": s_sem.tolist(), "s_rrf": s_rrf.tolist(),
               "r_bm": r_bm.tolist(), "r_sem": r_sem.tolist(), "r_rrf": r_rrf.tolist()},
              fh, indent=2)
print("dados_relatorio.json salvo", flush=True)
