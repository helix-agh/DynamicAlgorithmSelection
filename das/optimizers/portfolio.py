"""Portfolio of available sub-optimizers.

Add new algorithm classes here and they become selectable via the CLI.
"""

from das.optimizers.PSO import SPSO, SPSOL, IPSO, CPSO
from das.optimizers.DE import JDE21, MADDE, NL_SHADE_RSP
from das.optimizers.ES import CMAES, LMCMAES
from das.optimizers.BO import GPBO_EI, GPBO_UCB

PORTFOLIO: dict = {
    "SPSO": SPSO,
    "SPSOL": SPSOL,
    "IPSO": IPSO,
    "CPSO": CPSO,
    "JDE21": JDE21,
    "MADDE": MADDE,
    "NL_SHADE_RSP": NL_SHADE_RSP,
    "CMAES": CMAES,
    "LMCMAES": LMCMAES,
    "GPBO_EI": GPBO_EI,
    "GPBO_UCB": GPBO_UCB,
}


def get_portfolio(names: list[str]) -> list:
    """Return optimizer classes for the given names.

    Raises ValueError for any unknown name so errors surface early.
    """
    unknown = [n for n in names if n not in PORTFOLIO]
    if unknown:
        raise ValueError(
            f"Unknown optimizer(s): {unknown}. Available: {list(PORTFOLIO)}"
        )
    return [PORTFOLIO[n] for n in names]
