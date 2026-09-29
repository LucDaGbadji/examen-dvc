# Examen DVC et Dagshub

Reproducible modelling workflow built with DVC and hosted on DagsHub. The
objective is to model the silica concentration of the final product of a
mineral flotation process from its operational parameters.

## Data

- Source: https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv
- Location: `data/raw_data/raw.csv`, tracked by DVC through `data/raw_data/raw.csv.dvc`
- Size: 1817 observations, with no missing values and no duplicates
- Target: `silica_concentrate`, the last column of the dataset
- Features: `ave_flot_air_flow`, `ave_flot_level`, `iron_feed`, `starch_flow`,
  `amina_flow`, `ore_pulp_flow`, `ore_pulp_pH`, `ore_pulp_density`

The file also contains a `date` column. It is a timestamp identifier and not an
operational parameter of the process, so the splitting stage removes every
non-numeric column.

The features span very heterogeneous scales, from the order of one unit for
`ore_pulp_density` to several thousands for `starch_flow`. They are therefore
standardised before modelling.

## Repository structure

```
.
├── data
│   ├── raw_data            raw dataset (DVC-tracked)
│   ├── processed_data      train/test splits and scaled features (DVC outputs)
│   └── prediction.csv      test-set predictions (DVC output)
├── metrics
│   └── scores.json         evaluation metrics (Git-tracked)
├── models
│   ├── best_params.pkl     best hyper-parameters found by the grid search
│   └── gbr_model.pkl       trained model
├── src
│   ├── data
│   │   ├── data_split.py
│   │   └── normalize.py
│   └── models
│       ├── grid_search.py
│       ├── training.py
│       └── evaluate.py
├── params.yaml             parameters of the split, grid search and model
├── requirements.txt        pinned Python dependencies
├── dvc.yaml                pipeline definition
└── dvc.lock                locked state of the last pipeline run
```

## Pipeline

| Stage        | Script                      | Main outputs                                             |
|--------------|-----------------------------|----------------------------------------------------------|
| `split`      | `src/data/data_split.py`    | `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`  |
| `normalize`  | `src/data/normalize.py`     | `X_train_scaled.csv`, `X_test_scaled.csv`                 |
| `gridsearch` | `src/models/grid_search.py` | `models/best_params.pkl`                                  |
| `training`   | `src/models/training.py`    | `models/gbr_model.pkl`                                    |
| `evaluate`   | `src/models/evaluate.py`    | `data/prediction.csv`, `metrics/scores.json`              |

1. **split** — removes the non-numeric columns, separates the target and
   performs a train/test split. The test size and the random seed are read
   from `params.yaml`.
2. **normalize** — fits a `StandardScaler` on the training features only, then
   applies it to both sets, which avoids any data leakage.
3. **gridsearch** — runs `GridSearchCV` on a `GradientBoostingRegressor` with
   cross-validation. The grid, the number of folds and the scoring metric are
   read from `params.yaml`.
4. **training** — fits the final model on the whole training set with the best
   parameters found by the grid search.
5. **evaluate** — predicts on the scaled test set, then writes the predictions
   and the evaluation metrics.

## Modelling choices

The linear correlations between the operational parameters and the target
remain weak, the strongest being close to 0.23 in absolute value. A linear
model would therefore capture little signal, which motivates the choice of a
`GradientBoostingRegressor`, able to model non-linear effects and interactions
between parameters.

## Parameters

`params.yaml` holds three sections:

- `split`: `test_size`, `random_state`
- `grid_search`: `cv`, `scoring`, `param_grid` (`n_estimators`,
  `learning_rate`, `max_depth`)
- `model`: `random_state`

Changing a value in this file makes DVC re-run only the stages that depend on
it.

## Metrics

`metrics/scores.json` contains `mse`, `rmse`, `mae` and `r2`, computed on the
test set. They can be displayed with:

```bash
dvc metrics show
```

## Remote storage and review access

The DVC remote `origin` points to the DagsHub storage of this repository and
its URL is declared in `.dvc/config`. The access credentials are kept in
`.dvc/config.local`, which is ignored by Git.

The repository is shared with https://dagshub.com/licence.pedago as a
collaborator with read-only access rights.

## Reproduction

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Configure the DagsHub credentials locally, then retrieve the data and re-run
the workflow:

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <DAGSHUB_USER>
dvc remote modify origin --local password <DAGSHUB_TOKEN>

dvc pull
dvc repro
dvc dag
```

`dvc repro` only re-runs the stages whose dependencies, parameters or code have
changed.
