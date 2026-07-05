"""Nomination-aware compound translation (Pirelli C-numbers).

The lap/deg model and the strategy prior are keyed by the RELATIVE labels
SOFT/MEDIUM/HARD, but Pirelli nominates a different physical triple (C1..C6) per
weekend. When a nomination shifts between seasons at the same circuit (Barcelona:
C1/C2/C3 through 2025, C2/C3/C4 in 2026), label-keyed history rates the wrong tyres:
"SOFT" history describes a compound that may now be the MEDIUM.

The fix is translation, not re-estimation: historical evidence at a circuit is mapped
into the TARGET year's label space through the C-numbers before use.

  * Lap rows (physics): a row keeps its C-number; it becomes whatever label that
    C-number carries in the target year. Rows whose C-number is not nominated in the
    target year are DROPPED (clamping would pollute a fit with a different compound;
    shrinkage toward the global fit fills the gap).
  * Strategy sequences (behaviour): labels are mapped through the C-numbers with
    clamp-to-nearest, preserving the sequence SHAPE (stop count, stint structure) even
    when an end compound has no exact target equivalent.

Circuit-years missing from ``config/nominations.yaml`` are left untouched, so the
model degrades gracefully to the old label-keyed behaviour.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yaml

from strategy_sim2.settings import load_settings, resolve_path

_LABELS = ("HARD", "MEDIUM", "SOFT")  # index-aligned with the [hard, medium, soft] yaml rows


@lru_cache(maxsize=4)
def _load(path_str: str) -> dict[tuple[int, str], dict[str, int]]:
    """{(year, circuit) -> {label -> C-number}}."""
    with open(path_str) as f:
        raw = yaml.safe_load(f) or {}
    out: dict[tuple[int, str], dict[str, int]] = {}
    for year, circuits in raw.items():
        for circ, triple in (circuits or {}).items():
            out[(int(year), str(circ))] = {lab: int(c) for lab, c in zip(_LABELS, triple)}
    return out


def load_nominations(cfg: dict | None = None) -> dict[tuple[int, str], dict[str, int]]:
    cfg = cfg or load_settings()
    path = cfg.get("nominations", {}).get("path", "strategy_sim2/config/nominations.yaml")
    return _load(str(resolve_path(path)))


def label_map(source: dict[str, int], target: dict[str, int],
              clamp: bool) -> dict[str, str | None]:
    """Map source-year labels to target-year labels via C-numbers.

    Exact C match wins. Otherwise: ``clamp`` maps to the target label with the nearest
    C-number (ties toward the harder compound — the conservative direction for
    degradation); ``clamp=False`` maps to None (caller drops the row).
    """
    by_c = {c: lab for lab, c in target.items()}
    out: dict[str, str | None] = {}
    for lab, c in source.items():
        if c in by_c:
            out[lab] = by_c[c]
        elif clamp:
            out[lab] = min(target, key=lambda t: (abs(target[t] - c), target[t]))
        else:
            out[lab] = None
    return out


def relabel_laps(laps: pd.DataFrame, target_year: int,
                 cfg: dict | None = None) -> pd.DataFrame:
    """Translate the ``compound`` column of historical laps into the target year's
    label space, circuit by circuit. No-op for circuit-years without a nomination
    entry or whose nomination equals the target's."""
    cfg = cfg or load_settings()
    if not cfg.get("nominations", {}).get("relabel_laps", True) or not len(laps):
        return laps
    noms = load_nominations(cfg)
    pieces, changed = [], False
    for (year, circ), grp in laps.groupby(["year", "circuit"], sort=False):
        src = noms.get((int(year), str(circ)))
        tgt = noms.get((int(target_year), str(circ)))
        if src is None or tgt is None or src == tgt:
            pieces.append(grp)
            continue
        m = label_map(src, tgt, clamp=False)
        grp = grp.copy()
        grp["compound"] = grp["compound"].map(lambda lab: m.get(lab, lab))
        pieces.append(grp[grp["compound"].notna()])
        changed = True
    return pd.concat(pieces, ignore_index=True) if changed else laps


def sequence_relabeler(year: int, circuit: str, target_year: int,
                       cfg: dict | None = None):
    """Callable mapping a compound sequence at (year, circuit) into the target year's
    label space (clamped), or None when no translation applies."""
    cfg = cfg or load_settings()
    if not cfg.get("nominations", {}).get("relabel_prior", True):
        return None
    noms = load_nominations(cfg)
    src = noms.get((int(year), str(circuit)))
    tgt = noms.get((int(target_year), str(circuit)))
    if src is None or tgt is None or src == tgt:
        return None
    m = label_map(src, tgt, clamp=True)

    def _relabel(seq):
        return tuple(m.get(lab, lab) for lab in seq)

    return _relabel
