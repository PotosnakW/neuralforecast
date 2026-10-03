"""Aggregate per-seed forecasts.csv into one results.csv per experiment.

Same metric as training/forecast_error.py (MAE/RMSE over forecast windows whose
input window has at least one observed value), but it knows where each
experiment's runs actually live after the Sept-2026 ciexcl fix, so the table
notebooks can read a single results.csv per experiment with no overlay logic:

  infini_{gate}_t5tiny  =  {gate}_ciexcl_fix/  all 21 datasets, both arms (post-fix)
                         + {gate}/             13 original datasets, *_ciincl only
                                               (its *_ciexcl columns are pre-fix, stale)
  everything else       =  the shared folder as-is

Writes OUT_DIR/{experiment}/results.csv with columns
  dataset, {col}_{mae,rmse}_{mean,sd}, {col}_n_seeds
and never touches the source folders.

Usage (one experiment per call so slurm can fan out; see build_results.sbatch):
  python build_results.py --experiment vanilla_t5tiny
  python build_results.py --list
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

SHARED = '/zfsauton/project/public/dhowarth/icml26_mvts/icml_exp_results'
SCRATCH = '/zfsauton/scratch/wpotosna/neuralforecast_mica/exp_results'
OUT_DIR = '/zfsauton/scratch/wpotosna/neuralforecast_mica/results_csv'
TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training')

ORIGINAL_DATASETS = [
    'simglucose', 'iowa_ihop_smex_windspeed', 'iowa_plows_windspeed',
    'M_DENSE_H', 'M_DENSE_D', 'jena_weather_H', 'jena_weather_D',
    'ett1_H', 'ett1_D', 'ett1_W', 'ett2_H', 'ett2_D', 'ett2_W',
]
NEW_DATASETS = [
    'covid_deaths', 'LOOP_SEATTLE_D',
    'solar_H', 'solar_D', 'solar_W',
    'electricity_H', 'electricity_D', 'electricity_W',
]
ALL_DATASETS = ORIGINAL_DATASETS + NEW_DATASETS

ID_COLS = {'unique_id', 'ds', 'cutoff', 'y', 'available_mask'}
SEEDS = [1, 2, 3, 4, 5]


def src(root, folder, datasets=ALL_DATASETS, cols=None, seeds=SEEDS, fallback=None,
        rename=None):
    """One place runs live. `cols` is a regex a model column must match to be kept;
    `fallback` is another folder under the same root tried when a seed is missing;
    `rename` maps a column name as written in forecasts.csv to its correct alias."""
    return dict(root=root, folder=folder, datasets=datasets, cols=cols, seeds=seeds,
                fallback=fallback, rename=rename or {})


GATES = [
    'infini_t5tiny',
    'infini_channelwise_t5tiny',
    'infini_layerwise_t5tiny',
    'infini_layerwise_channelwise_t5tiny',
    'infini_mlpmixer_t5tiny',
    'infini_mlpquerymixer_t5tiny',
]

# The January infini_t5tiny runs (trained 2026-01-20, before the alias was fixed in
# c5ec1d7) wrote PatchTST's shared-beta gate under the channelwise alias. The
# checkpoints confirm it is the shared gate (channelwise_beta=False), and the
# forecasts differ from infini_channelwise_t5tiny's, so it is only a mislabel.
JAN_ALIAS_FIX = {
    'infini_t5tiny': {'AutoPatchTSTMultivariate_infini_channelwise_ciincl':
                      'AutoPatchTSTMultivariate_infini_ciincl'},
}

# Sources are listed in precedence order: a (dataset, column) is taken from the
# first source that has it.
EXPERIMENTS = {
    **{g: [src(SHARED, f'{g}_ciexcl_fix'),
           src(SHARED, g, datasets=ORIGINAL_DATASETS, cols=r'_ciincl$',
               rename=JAN_ALIAS_FIX.get(g))]
       for g in GATES},

    'infini_poolmean_t5tiny': [src(SHARED, 'infini_poolmean_t5tiny')],
    'infini_poolmean_layerwise_t5tiny': [src(SHARED, 'infini_poolmean_layerwise_t5tiny')],
    'infini_poolmean_mlpquerymixer_t5tiny': [src(SHARED, 'infini_poolmean_mlpquerymixer_t5tiny')],

    'vanilla_t5tiny': [src(SHARED, 'vanilla_t5tiny')],
    'vanilla_pca_t5tiny': [src(SHARED, 'vanilla_pca_t5tiny')],
    'vanilla_t5tiny_ishm3': [src(SHARED, 'vanilla_t5tiny_ishm3')],
    'infini_mlpquerymixer_t5tiny_ishm3': [src(SHARED, 'infini_mlpquerymixer_t5tiny_ishm3')],
    'infini_mlpquerymixer_t5tiny_static_weights': [src(SHARED, 'infini_mlpquerymixer_t5tiny_static_weights')],
    'infini_mlpquerymixer_t5tiny_dynamic_weights': [src(SHARED, 'infini_mlpquerymixer_t5tiny_dynamic_weights')],

    'itransformer_baseline': [src(SHARED, 'itransformer_baseline')],
    'crossformer_baseline': [src(SHARED, 'crossformer_baseline')],
    'timerxl_baseline': [src(SHARED, 'timerxl_baseline')],
    'tsmixer_baseline': [src(SHARED, 'tsmixer_baseline')],
    'timemixer_baseline': [src(SHARED, 'timemixer_baseline')],
    'multivariateMLP_baseline': [src(SHARED, 'multivariateMLP_baseline')],

    # Deterministic / inference-only: one seed.
    'statsforecast': [src(SHARED, 'statsforecast', seeds=[1], fallback='statsforecast_refit_false')],
    'chronos2.0_baseline': [src(SHARED, 'chronos2.0_baseline_wpotosna', seeds=[1])],
    'toto2_baseline': [src(SHARED, 'toto2_baseline', seeds=[1])],
    'toto2_313m_baseline': [src(SHARED, 'toto2_313m_baseline', seeds=[1])],
    'timesfm3_baseline': [src(SHARED, 'timesfm3_baseline', seeds=[1])],
}


# ---------------------------------------------------------------- availability mask

def _load_dataset(dataset):
    """(df, h) exactly as forecast_error.py loads them."""
    sys.path.insert(0, TRAINING_DIR)
    from experiment_datasets import get_datasets

    cwd = os.getcwd()
    os.chdir(TRAINING_DIR)  # loaders use '../datasets/...' relative paths
    try:
        if dataset in ('iowa_ihop_smex_windspeed', 'iowa_plows_windspeed'):
            fname = {'iowa_ihop_smex_windspeed': 'preprocessed_iowa_ihop_smex02_dataset.csv',
                     'iowa_plows_windspeed': 'preprocessed_iowa_plows_dataset.csv'}[dataset]
            df = pd.read_csv(f'../datasets/{fname}')
            df.ds = pd.to_datetime(df.ds, format='%Y-%m-%d %H:%M:%S')
            df = (df.groupby('unique_id').resample('5min', on='ds')
                    .agg({'y': 'mean', 'available_mask': 'max'}).reset_index())
            return df, 24

        class Args:
            pass
        args = Args()
        # results folders use '_' where GIFT-Eval names use '/' (ett1_H -> ett1/H)
        args.dataset_name = re.sub(r'_(H|D|W)$', r'/\1', dataset)
        df, h, _, _, _ = get_datasets(args)
        return df, h
    finally:
        os.chdir(cwd)


def availability_mask(dataset):
    """Per (unique_id, cutoff): number of observed points in the trailing 2h window,
    or None if the dataset is fully observed. Cached, since every experiment needs it."""
    cache = os.path.join(OUT_DIR, '_av_masks', f'{dataset}.parquet')
    none_marker = cache + '.none'
    if os.path.exists(none_marker):
        return None
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df, h = _load_dataset(dataset)
    if 'available_mask' not in df.columns or (df['available_mask'] == 1).all():
        open(none_marker, 'w').close()
        return None

    df = df.sort_values(['unique_id', 'ds'])
    av = pd.DataFrame({
        'unique_id': df['unique_id'].astype(str).values,
        'cutoff': pd.to_datetime(df['ds']).values,
        'sum_av_mask': (df.groupby('unique_id')['available_mask']
                          .transform(lambda s: s.rolling(window=2 * h, min_periods=1).sum())
                          .values),
    })
    tmp = f'{cache}.{os.getpid()}.tmp'  # concurrent array tasks may race on the cache
    av.to_parquet(tmp, index=False)
    os.replace(tmp, cache)
    return av


# ---------------------------------------------------------------- evaluation

def _run_dir(root, folder, dataset, seed):
    hits = sorted(glob.glob(f'{root}/{folder}/{dataset}/rs{seed}_ishm*_h*/forecasts.csv'))
    return hits[0] if hits else None


def evaluate_forecasts(path, av_mask, col_regex=None):
    """{model_column: (mae, rmse)} for one forecasts.csv."""
    header = pd.read_csv(path, nrows=0).columns
    model_cols = [c for c in header if c not in ID_COLS
                  and (col_regex is None or re.search(col_regex, c))]
    if not model_cols:
        return {}

    usecols = ['unique_id', 'cutoff', 'y'] + model_cols
    df = pd.read_csv(path, usecols=usecols, engine='pyarrow')
    if av_mask is not None:
        df['unique_id'] = df['unique_id'].astype(str)
        df['cutoff'] = pd.to_datetime(df['cutoff'])
        df = df.merge(av_mask, on=['unique_id', 'cutoff'], how='left')
        df = df[df['sum_av_mask'] > 0]

    y = df['y'].to_numpy(dtype=float)
    out = {}
    for c in model_cols:
        err = y - df[c].to_numpy(dtype=float)
        out[c] = (np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2)))
    return out


def build_experiment(name):
    rows = []
    for dataset in ALL_DATASETS:
        sources = [s for s in EXPERIMENTS[name] if dataset in s['datasets']]
        if not sources:
            continue
        av_mask = None
        mask_loaded = False

        per_seed = {}  # column -> {seed: (mae, rmse)}
        for s in sources:
            for seed in s['seeds']:
                path = _run_dir(s['root'], s['folder'], dataset, seed)
                if path is None and s['fallback']:
                    path = _run_dir(s['root'], s['fallback'], dataset, seed)
                if path is None:
                    continue
                if not mask_loaded:
                    av_mask, mask_loaded = availability_mask(dataset), True
                print(f'  {dataset} rs{seed}  {os.path.relpath(path, s["root"])}', flush=True)
                for col, metrics in evaluate_forecasts(path, av_mask, s['cols']).items():
                    col = s['rename'].get(col, col)
                    per_seed.setdefault(col, {}).setdefault(seed, metrics)  # first source wins

        if not per_seed:
            print(f'  {dataset}: no runs found')
            continue
        row = {'dataset': dataset}
        for col, by_seed in per_seed.items():
            vals = np.array(list(by_seed.values()))  # (n_seeds, 2)
            for j, metric in enumerate(['mae', 'rmse']):
                row[f'{col}_{metric}_mean'] = vals[:, j].mean()
                row[f'{col}_{metric}_sd'] = vals[:, j].std(ddof=1) if len(vals) > 1 else np.nan
            row[f'{col}_n_seeds'] = len(vals)
        rows.append(row)

    out = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, name, 'results.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f'wrote {path}  ({len(out)} datasets)')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='name in EXPERIMENTS, or an integer index (slurm array)')
    parser.add_argument('--list', action='store_true', help='print experiment names with indices')
    args = parser.parse_args()

    names = list(EXPERIMENTS)
    if args.list:
        for i, n in enumerate(names):
            print(i, n)
        sys.exit()
    name = names[int(args.experiment)] if args.experiment.isdigit() else args.experiment
    print(f'### {name}')
    build_experiment(name)
