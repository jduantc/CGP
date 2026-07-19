# External dependencies

This project builds on two lab tools that are not vendored into this repo
(they're independently developed, and one is 300+ MB with third-party
example data). Clone them alongside this repo and add them to your
`PYTHONPATH`.

## fges-py

https://github.com/eberharf/fges-py, commit `961188a` (the version this
project was developed against).

fges-py provides the FGES causal discovery search and SEM/BDeu scoring used
by `cgp/fges_runner.py`. It has no license file upstream and is not
published to PyPI, so it can't be pip-installed — clone it and put its
root directory on `PYTHONPATH`:

```sh
git clone https://github.com/eberharf/fges-py.git
cd fges-py && git checkout 961188a
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

Apply `fges-py-checkpoint-score.patch` (a one-line change exposing the
final score in `FGES`'s checkpoint dict, which `cgp/fges_runner.py` reads):

```sh
git apply /path/to/this/repo/external/fges-py-checkpoint-score.patch
```

## py-tetrad

https://github.com/cmu-phil/py-tetrad, commit `2ae7061` (MIT licensed, has
a `setup.py`). Used only for generating synthetic SEM data via TETRAD
during data-generation experiments (not required to run `cgp/` or
`scripts/run_cgp_demo.py` against existing data files). Follow py-tetrad's
own README for JVM/JPype and Tetrad-jar setup, then apply
`py-tetrad-simulateSEM.patch`, which adds a `simulateSEM(edgelist, means,
coefs, vars, ...)` helper to `pytetrad/tools/simulate.py` for simulating
data from a specific DAG with specific edge coefficients (rather than
py-tetrad's built-in random-DAG simulators):

```sh
git clone https://github.com/cmu-phil/py-tetrad.git
cd py-tetrad && git checkout 2ae7061
git apply /path/to/this/repo/external/py-tetrad-simulateSEM.patch
pip install -e .
```
