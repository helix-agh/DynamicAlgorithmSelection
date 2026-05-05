"""Shared training utilities."""

import json
import os

from das.env.das_env import DASEnv


def load_global_optima(path: str = "bbob_optima.jsonl") -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: v for line in f for k, v in json.loads(line).items()}


def make_das_env(problem_ids: list[str], optimizers: list, cfg: dict):
    """Return a zero-argument factory for use with SB3's make_vec_env."""

    def _init():
        import cocoex as _cx

        _cx.utilities.MiniPrint()
        _suite = _cx.Suite("bbob", "", "")
        return DASEnv(
            problem_ids=problem_ids,
            suite=_suite,
            optimizers=optimizers,
            fe_multiplier=cfg["fe_multiplier"],
            n_checkpoints=cfg["n_checkpoints"],
            checkpoint_division_base=cfg["cdb"],
            reward_option=cfg["reward_option"],
            n_individuals=cfg["n_individuals"],
            seed=cfg.get("seed"),
        )

    return _init


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
