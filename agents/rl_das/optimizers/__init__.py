"""DE optimizer registry for RL-DAS.

Optimizer implementations:
  - nl_shade_rsp.py  → NL_SHADE_RSP
  - jde21.py         → JDE21
  - madde.py         → MadDE
"""

from __future__ import annotations

from .jde21 import JDE21
from .madde import MadDE
from .nl_shade_rsp import NL_SHADE_RSP

__all__ = ["NL_SHADE_RSP", "JDE21", "MadDE", "get_rldas_portfolio", "_PORTFOLIO"]

_PORTFOLIO: dict[str, type] = {
    "NL_SHADE_RSP": NL_SHADE_RSP,
    "MADDE": MadDE,
    "JDE21": JDE21,
}


def get_rldas_portfolio(names: list[str] | None = None) -> list:
    """Return instantiated RL-DAS optimizer objects.

    Parameters
    ----------
    names:
        Optimizer names. Defaults to ``["NL_SHADE_RSP", "MADDE", "JDE21"]``.
        Valid names: ``"NL_SHADE_RSP"``, ``"MADDE"``, ``"JDE21"``.
    """
    if names is None:
        names = ["NL_SHADE_RSP", "MADDE", "JDE21"]
    unknown = [n for n in names if n not in _PORTFOLIO]
    if unknown:
        raise ValueError(
            f"Unknown RL-DAS optimizer(s): {unknown}. Valid choices: {list(_PORTFOLIO)}"
        )
    return [_PORTFOLIO[n]() for n in names]
