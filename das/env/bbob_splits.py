"""BBOB problem-set constants and train/test/CV split helpers."""

from itertools import product

import numpy as np

ALL_DIMS = [2, 3, 5, 10, 20, 40]
ALL_FUNCTIONS = set(range(1, 25))
INSTANCE_IDS = [1, 2, 3, 4, 5, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
EASY_TRAIN_FUNCTIONS = {4, *range(6, 15), 18, 19, 20, 22, 23, 24}


def build_problem_ids(
    functions: set[int],
    dims: list[int],
    instances: list[int] | None = None,
) -> list[str]:
    insts = instances if instances is not None else INSTANCE_IDS
    return [
        f"bbob_f{f:03d}_i{i:02d}_d{d:02d}"
        for i, f, d in product(insts, sorted(functions), dims)
    ]


def get_train_test_split(mode: str, dims: list[int]) -> tuple[list[str], list[str]]:
    """Return (train_ids, test_ids) for the given split mode and dimensions.

    Modes:
      easy   – train on easy BBOB functions, test on hard ones
      hard   – inverse of easy
      random – random 2/3 / 1/3 split on all problem IDs
    """
    if mode == "easy":
        return (
            build_problem_ids(EASY_TRAIN_FUNCTIONS, dims),
            build_problem_ids(ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, dims),
        )
    if mode == "hard":
        return (
            build_problem_ids(ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, dims),
            build_problem_ids(EASY_TRAIN_FUNCTIONS, dims),
        )
    # random 2/3 – 1/3 split
    all_ids = build_problem_ids(ALL_FUNCTIONS, dims)
    rng = np.random.default_rng()
    rng.shuffle(all_ids)
    split = 2 * len(all_ids) // 3
    return all_ids[:split], all_ids[split:]


def get_cv_folds(
    cv_mode: str, dims: list[int], seed: int = 0, n_folds: int = 3
) -> list[tuple[list[str], list[str], str]]:
    """Return (train_ids, test_ids, fold_tag) for each CV fold.

    LOIO: instance IDs are shuffled and split into n_folds groups;
          each fold tests on one group and trains on the rest.
    LOPO: BBOB functions are shuffled and split into n_folds groups;
          each fold tests on all problems from one group of functions.
    """
    rng = np.random.default_rng(seed)
    folds = []

    if cv_mode == "LOIO":
        insts = list(INSTANCE_IDS)
        rng.shuffle(insts)
        chunk = len(insts) // n_folds
        for i in range(n_folds):
            test_insts = insts[i * chunk : (i + 1) * chunk]
            train_insts = [inst for inst in insts if inst not in set(test_insts)]
            folds.append(
                (
                    build_problem_ids(ALL_FUNCTIONS, dims, train_insts),
                    build_problem_ids(ALL_FUNCTIONS, dims, test_insts),
                    f"loio{i}",
                )
            )
    else:  # LOPO
        fns = list(ALL_FUNCTIONS)
        rng.shuffle(fns)
        chunk = len(fns) // n_folds
        for i in range(n_folds):
            test_fns = set(fns[i * chunk : (i + 1) * chunk])
            train_fns = ALL_FUNCTIONS - test_fns
            folds.append(
                (
                    build_problem_ids(train_fns, dims),
                    build_problem_ids(test_fns, dims),
                    f"lopo{i}",
                )
            )

    return folds
