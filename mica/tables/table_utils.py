"""Shared code for the per-table notebooks in this folder.

Each notebook lists its columns as (label, model_key) pairs; MODELS maps a key to
the experiment results.csv and model column it comes from. The results.csv files
are built by build_results.py, which already resolves the ciexcl fix (post-fix
*_ciexcl_fix runs first, January runs only for 13-dataset ciincl) -- so nothing
here needs to know which folder a number came from.
"""
import os
import re

import numpy as np
import pandas as pd

RESULTS_DIR = '/zfsauton/scratch/wpotosna/neuralforecast_mica/results_csv'
TABLES_DIR = '/zfsauton/scratch/wpotosna/neuralforecast_mica/paper_tables'

# Row order shared by every table; datasets with no data in a table are dropped.
DATASETS = [
    'simglucose',
    'covid_deaths',
    'iowa_ihop_smex_windspeed',
    'iowa_plows_windspeed',
    'jena_weather_H',
    'jena_weather_D',
    'M_DENSE_H',
    'M_DENSE_D',
    'LOOP_SEATTLE_D',
    'ett1_H',
    'ett1_D',
    'ett1_W',
    'ett2_H',
    'ett2_D',
    'ett2_W',
    'solar_H',
    'solar_D',
    'solar_W',
    'electricity_H',
    'electricity_D',
    'electricity_W',
]

DATASET_MAPPING = {
    'covid_deaths': ('COVID Deaths', 'D'),
    'simglucose': ('Simglucose', '5min'),
    'iowa_ihop_smex_windspeed': ('Iowa IHOP SMEX02', '5min'),
    'iowa_plows_windspeed': ('Iowa PLOWS', '5min'),
    'jena_weather_D': ('Jena Weather', 'D'),
    'jena_weather_H': ('Jena Weather', 'H'),
    'M_DENSE_D': ('M-DENSE', 'D'),
    'M_DENSE_H': ('M-DENSE', 'H'),
    'LOOP_SEATTLE_H': ('Loop-Seattle', 'H'),
    'LOOP_SEATTLE_D': ('Loop-Seattle', 'D'),
    'ett1_H': ('ETT1', 'H'),
    'ett1_D': ('ETT1', 'D'),
    'ett1_W': ('ETT1', 'W'),
    'ett2_H': ('ETT2', 'H'),
    'ett2_D': ('ETT2', 'D'),
    'ett2_W': ('ETT2', 'W'),
    'solar_H': ('Solar', 'H'),
    'solar_D': ('Solar', 'D'),
    'solar_W': ('Solar', 'W'),
    'electricity_H': ('Electricity', 'H'),
    'electricity_D': ('Electricity', 'D'),
    'electricity_W': ('Electricity', 'W'),
}


# ---------------------------------------------------------------- model registry
# key -> (experiment folder under RESULTS_DIR, model column prefix in results.csv)

BACKBONES = {'moment': 'AutoMOMENT', 'patchtst': 'AutoPatchTSTMultivariate'}

# gate name -> (experiment, alias infix). Aliases are inconsistent across gates
# (mixers have no 'infini_' prefix); see training/experiment_models.py.
GATES = {
    'shared': ('infini_t5tiny', 'infini'),
    'channelwise': ('infini_channelwise_t5tiny', 'infini_channelwise'),
    'layerwise': ('infini_layerwise_t5tiny', 'infini_layerwise'),
    'layerwise_channelwise': ('infini_layerwise_channelwise_t5tiny', 'infini_layerwise_channelwise'),
    'mlp': ('infini_mlpmixer_t5tiny', 'mlpmixer'),
    'mlpquery': ('infini_mlpquerymixer_t5tiny', 'mlpquerymixer'),
}

# Mean-pool memory (A_global = (1/C) sum_c V^(c)); ciincl-only by construction.
POOLMEAN_GATES = {
    'shared': ('infini_poolmean_t5tiny', 'infini_poolmean'),
    'layerwise': ('infini_poolmean_layerwise_t5tiny', 'infini_poolmean_layerwise'),
    'mlpquery': ('infini_poolmean_mlpquerymixer_t5tiny', 'poolmean_mlpquerymixer'),
}

MODELS = {}
for bb, p in BACKBONES.items():
    MODELS[bb] = ('vanilla_t5tiny', f'{p}_vanilla')
    MODELS[f'{bb}_hm'] = ('vanilla_t5tiny', f'{p}_vanilla_headmixer')
    MODELS[f'{bb}_pca'] = ('vanilla_pca_t5tiny', f'{p}_vanilla_pca')
    MODELS[f'{bb}_ishm3'] = ('vanilla_t5tiny_ishm3', f'{p}_vanilla')
    MODELS[f'{bb}_mlpquery_incl_ishm3'] = ('infini_mlpquerymixer_t5tiny_ishm3', f'{p}_mlpquerymixer_ciincl')
    for gate, (exp, alias) in GATES.items():
        for arm in ('incl', 'excl'):
            MODELS[f'{bb}_{gate}_{arm}'] = (exp, f'{p}_{alias}_ci{arm}')
    for gate, (exp, alias) in POOLMEAN_GATES.items():
        MODELS[f'{bb}_{gate}_pm'] = (exp, f'{p}_{alias}')

MODELS.update({
    # channel-weighting variants only exist for PatchTST
    'patchtst_mlpquery_sw': ('infini_mlpquerymixer_t5tiny_static_weights',
                             'AutoPatchTSTMultivariate_mlpquerymixer_sw_ciincl'),
    'patchtst_mlpquery_dw': ('infini_mlpquerymixer_t5tiny_dynamic_weights',
                             'AutoPatchTSTMultivariate_mlpquerymixer_dw_ciincl'),

    'itransformer': ('itransformer_baseline', 'AutoiTransformer_multivariate'),
    'itransformer_t5': ('itransformer_baseline', 'AutoiTransformerT5_multivariate'),
    'crossformer': ('crossformer_baseline', 'AutoCrossformer_multivariate'),
    'timerxl': ('timerxl_baseline', 'AutoTimerXL_multivariate'),
    'tsmixer': ('tsmixer_baseline', 'AutoTSMixer_multivariate'),
    'timemixer': ('timemixer_baseline', 'AutoTimeMixer_multivariate'),
    'mlp': ('multivariateMLP_baseline', 'AutoMLPMultivariate_multivariate'),
    'autoets': ('statsforecast', 'AutoETS'),
    'chronos2': ('chronos2.0_baseline', 'Chronos_multivariate'),
    'toto2': ('toto2_baseline', 'Toto2_zeroshot'),
    'toto2_313m': ('toto2_313m_baseline', 'Toto2_313m_zeroshot'),
    'timesfm3': ('timesfm3_baseline', 'TimesFM3_zeroshot'),
})

# Inference-only / deterministic models: one seed, no sd column to fill.
SINGLE_SEED = {'autoets', 'chronos2', 'toto2', 'toto2_313m', 'timesfm3'}
N_SEEDS = 5


# ---------------------------------------------------------------- loading

_cache = {}


def load_results(experiment):
    if experiment not in _cache:
        path = os.path.join(RESULTS_DIR, experiment, 'results.csv')
        _cache[experiment] = pd.read_csv(path).set_index('dataset')
    return _cache[experiment]


def collect(columns, metric='mae', datasets=DATASETS):
    """Numeric (means, sds, n_seeds) frames, one column per label, rows = datasets."""
    means, sds, ns = {}, {}, {}
    for label, key in columns:
        exp, alias = MODELS[key]
        df = load_results(exp).reindex(datasets)
        for out, col in ((means, f'{alias}_{metric}_mean'),
                         (sds, f'{alias}_{metric}_sd'),
                         (ns, f'{alias}_n_seeds')):
            if col not in df.columns:
                raise KeyError(f'{label}: {col} not in {exp}/results.csv')
            out[label] = df[col]
    means, sds, ns = (pd.DataFrame(d, index=pd.Index(datasets, name='dataset'))
                      for d in (means, sds, ns))
    return means, sds, ns


def report_gaps(ns, columns):
    """Print cells with no runs, or fewer seeds than the model should have."""
    single = {label for label, key in columns if key in SINGLE_SEED}
    lines = []
    for label in ns.columns:
        expected = 1 if label in single else N_SEEDS
        for ds, n in ns[label].items():
            if pd.isna(n):
                lines.append(f'  {label:<28} {ds:<26} missing')
            elif n < expected:
                lines.append(f'  {label:<28} {ds:<26} {int(n)}/{expected} seeds')
    print('Gaps:' if lines else 'No gaps: every cell has all seeds.')
    print('\n'.join(lines))


# ---------------------------------------------------------------- formatting

def format_table(means, sds, improvement_pairs=(), decimals=3):
    """LaTeX-ready table: best mean per row bold, second underlined, and for each
    (baseline_label, label) pair, `label` coloured blue where it beats the baseline.
    Missing cells are '-'. Output columns: Dataset, Frequency, then label, label_se."""
    fmt = f'{{:.{decimals}f}}'
    out = pd.DataFrame(index=means.index)

    beats = pd.DataFrame(False, index=means.index, columns=means.columns)
    for base, label in improvement_pairs:
        beats[label] = means[label] < means[base]

    for label in means.columns:
        cells, se_cells = [], []
        for ds in means.index:
            row = means.loc[ds].dropna().sort_values()
            v = means.at[ds, label]
            if pd.isna(v):
                cells.append('-')
            else:
                s = fmt.format(v)
                if v == row.iloc[0]:
                    s = f'\\textbf{{{s}}}'
                elif len(row) > 1 and v == row.iloc[1]:
                    s = f'\\underline{{{s}}}'
                if beats.at[ds, label]:
                    s = f'\\textcolor{{blue}}{{{s}}}'
                cells.append(s)
            sd = sds.at[ds, label]
            se_cells.append('-' if pd.isna(sd) else fmt.format(sd))
        out[label] = cells
        out[f'{label}_se'] = se_cells

    names = [DATASET_MAPPING.get(d, (d, '-')) for d in means.index]
    display_names = [n for n, _ in names]
    # Only show the dataset name on its first row (multirow-style).
    display_names = [n if i == 0 or n != display_names[i - 1] else ''
                     for i, n in enumerate(display_names)]
    out.insert(0, 'Frequency', [f for _, f in names])
    out.insert(0, 'Dataset', display_names)
    return out.reset_index(drop=True)


def make_table(columns, metric='mae', datasets=DATASETS, improvement_pairs=(),
               require=()):
    """collect -> drop rows -> report gaps -> format. Rows with no data at all are
    dropped, as are rows missing any label in `require` (for ablations that only ran
    on some datasets). Returns (formatted_table, numeric_means)."""
    means, sds, ns = collect(columns, metric, datasets)
    keep = means.notna().any(axis=1) & means[list(require)].notna().all(axis=1)
    means, sds, ns = means[keep], sds[keep], ns[keep]
    report_gaps(ns, columns)
    return format_table(means, sds, improvement_pairs), means


def save_table(df, name):
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, f'{name}.csv')
    df.to_csv(path, index=False)
    print(f'wrote {path}')
    return path


def average_ranks(means, complete_only=True):
    """Mean rank per model (1 = best). With complete_only, only datasets where every
    model has a result are ranked, so missing cells can't flatter or penalise anyone."""
    m = means.dropna() if complete_only else means
    dropped = sorted(set(means.index) - set(m.index))
    if dropped:
        print(f'Ranked on {len(m)} datasets; dropped (incomplete): {", ".join(dropped)}')
    return m.rank(axis=1, method='min').mean(axis=0).sort_values().round(3)
