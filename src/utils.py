import pandas as pd
import numpy as np
import re

_FIGURE_FONT_FAMILY = None


def configure_matplotlib_fonts():
    global _FIGURE_FONT_FAMILY

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if _FIGURE_FONT_FAMILY is None:
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in ['Times New Roman', 'Times', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']:
            if candidate in available_fonts:
                _FIGURE_FONT_FAMILY = candidate
                break
        else:
            _FIGURE_FONT_FAMILY = 'serif'

    serif_stack = [
        _FIGURE_FONT_FAMILY,
        'Times New Roman',
        'Times',
        'Nimbus Roman',
        'Liberation Serif',
        'DejaVu Serif',
    ]
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': serif_stack,
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False,
    })
    return _FIGURE_FONT_FAMILY


def get_clean_name(name):

    clean = str(name).replace("<", "_").replace(">", "_")
    clean = re.sub(r'[\\/:*?"|]', '_', clean)
    return clean.replace(" ", "_")

def normalize(series):

    denom = series.max() - series.min()
    if denom == 0: return series * 0.0
    return (series - series.min()) / denom

def calc_months_diff(d1, d2):

    if pd.isna(d1) or pd.isna(d2): return 0
    return (d1.year - d2.year) * 12 + d1.month - d2.month
