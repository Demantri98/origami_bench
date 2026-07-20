"""Load existing tuned shapes and realistic (N, K) pools for each family.

Pure-python (csv only). Used by the dataset generator to (a) know which
``(M, N, K[, q_dtype_w])`` points already exist on gfx950 (so we can exclude
them) and (b) build a pool of realistic weight shapes ``(N, K)`` to sample from.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Iterable

from .families import (
    CONFIG_DIR,
    MODEL_CONFIG_DIR,
    TARGET_GFX,
    Family,
)


def _matching_model_config_files(fam: Family) -> list[str]:
    """model_configs CSVs whose name matches this family (include/exclude)."""
    out = []
    for path in sorted(glob.glob(os.path.join(MODEL_CONFIG_DIR, "*.csv"))):
        name = os.path.basename(path)
        if not all(tok in name for tok in fam.mc_include):
            continue
        if any(tok in name for tok in fam.mc_exclude):
            continue
        out.append(path)
    return out


def family_csv_files(fam: Family) -> list[str]:
    """Main tuned CSV + all matching model_configs CSVs that exist on disk."""
    files = [os.path.join(CONFIG_DIR, fam.tuned_csv)]
    files += _matching_model_config_files(fam)
    return [f for f in files if os.path.exists(f)]


def _iter_rows(path: str) -> Iterable[dict]:
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            yield {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}


def _norm_dtype(tok: str | None) -> str:
    if not tok:
        return ""
    tok = tok.strip()
    # Normalize a few synonyms seen across CSVs.
    if tok in ("int8", "i8"):
        return "torch.int8"
    if tok in ("fp8", "f8", "float8_e4m3fn", "torch.float8_e4m3fnuz"):
        return "torch.float8_e4m3fn"
    return tok


def load_existing(fam: Family, gfx: str = TARGET_GFX) -> set[tuple]:
    """Set of existing keys for ``gfx``.

    Key is ``(M, N, K, q_dtype_w)`` for dtype families, else ``(M, N, K)``.
    Rows without a gfx column (rare/legacy) are treated as belonging to ``gfx``.
    """
    existing: set[tuple] = set()
    for path in family_csv_files(fam):
        for row in _iter_rows(path):
            row_gfx = row.get("gfx")
            if row_gfx and row_gfx != gfx:
                continue
            try:
                m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
            except (KeyError, ValueError, TypeError):
                continue
            if fam.has_dtype:
                dt = _norm_dtype(row.get("q_dtype_w"))
                existing.add((m, n, k, dt))
            else:
                existing.add((m, n, k))
    return existing


def load_nk_pool(fam: Family, gfx: str = TARGET_GFX) -> list[tuple[int, int]]:
    """Realistic ``(N, K)`` weight shapes seen for this family (all gfx).

    We union across gfx (not just the target) because a real weight shape is
    valid regardless of which arch happened to be tuned for it; this widens the
    realistic pool without inventing shapes.
    """
    nk: set[tuple[int, int]] = set()
    for path in family_csv_files(fam):
        for row in _iter_rows(path):
            try:
                nk.add((int(row["N"]), int(row["K"])))
            except (KeyError, ValueError, TypeError):
                continue
    return sorted(nk)


def load_m_pool(fam: Family, gfx: str = TARGET_GFX) -> list[int]:
    """Distinct M values observed for this family (any gfx)."""
    ms: set[int] = set()
    for path in family_csv_files(fam):
        for row in _iter_rows(path):
            try:
                ms.add(int(row["M"]))
            except (KeyError, ValueError, TypeError):
                continue
    return sorted(ms)
