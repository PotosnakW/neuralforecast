"""Shared code for the figure notebooks in this folder.

Results come from the same results.csv files as the tables (built by
../tables/build_results.py, see ../tables/table_utils.py for MODELS / collect), so
figures and tables always agree. Compute costs come from the computational-study
CSVs written by ../tables/run_all_tables.sh and computational_parameter_study_mica.py.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tables'))
from table_utils import DATASETS, MODELS, collect, load_results  # noqa: E402,F401

COST_DIR = '/zfsauton/scratch/wpotosna/neuralforecast_mica/table_results'

COLOR_PALETTE = ['#C0D6CA', '#78ACA8', '#2D6B8F', '#235796',
                 '#E7C4C0', '#E3A39A', '#CA6F6A', '#7B3841',
                 '#D5BC67', '#20425B', '#E77A5B', '#9C9DB2']


def load_costs(name):
    """A computational-study CSV from COST_DIR, indexed by 'Model Variant'.

    get_table() blanks the model name on repeated rows (multirow style) and older
    merges duplicated every row, so forward-fill the name and drop duplicates."""
    df = pd.read_csv(os.path.join(COST_DIR, name))
    df['model'] = df['model'].ffill()
    df['variant'] = df['variant'].fillna('')
    df = df.drop_duplicates(subset=['model', 'variant'], keep='first')
    df.index = (df['model'] + ' ' + df['variant']).str.strip()
    return df


def pct_improvement(base, other):
    """Per-dataset % MAE improvement of `other` over `base` (positive = better)."""
    return ((base - other) / base * 100).dropna()
