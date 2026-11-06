# src/preprocessor.py

import pandas as pd
import numpy as np
import re
from dataclasses import dataclass

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def _clean_cell(self, val):
        if pd.isna(val) or val == '': return np.nan
        val_str = str(val).strip()
        
        if '[T]' in val_str: return np.nan
        
        match = re.search(r'([-+]?\d*\.\d+|\d+)', val_str)
        if match: return float(match.group(1))
        return np.nan

    def load_and_process(self, file_path):
        df = pd.read_csv(file_path)
        cut_col = self.config['cut_off_column']
        
        try:
            idx = next(i for i, c in enumerate(df.columns) if str(c).lower() == cut_col.lower())
            df = df.iloc[:, :idx]
        except StopIteration: pass
        
        df.iloc[:, 0] = df.iloc[:, 0].fillna('Unknown')
        bench_names = df.iloc[:, 0].values
        df_data = df.iloc[:, 1:].copy()
        
        valid_row_indices = []
        for i, name in enumerate(bench_names):
            name_str = str(name).strip().lower()
            if "date" in name_str or "release" in name_str or name_str == "nan" or name_str == "unknown":
                continue
            valid_row_indices.append(i)
        
        df_data = df_data.iloc[valid_row_indices].reset_index(drop=True)
        bench_names = bench_names[valid_row_indices]

        df_numeric = df_data.apply(lambda col: col.map(self._clean_cell))
        
        df_norm = df_numeric.copy()
        tags, sf = self.config['tags'], self.config['safety_factor']
        
        for i in range(len(df_numeric)):
            name, row = str(bench_names[i]), df_numeric.iloc[i]
            valid = row.dropna()
            if valid.empty: continue
            max_v = valid.max()
            

            if tags['lib_0_1'] in name: 
                df_norm.iloc[i] = (1 - row) * 100

            elif tags['lib_standard'] in name:
                if max_v > 100: df_norm.iloc[i] = (1 - row / (max_v * sf)) * 100
                else: df_norm.iloc[i] = 100 - row

            elif tags['unbounded_hib'].lower() in name.lower():
                df_norm.iloc[i] = (row / (max_v * sf)) * 100

            else:
                if 0 < max_v <= 1.05: df_norm.iloc[i] = row * 100
                else: df_norm.iloc[i] = row
                
        return df_norm, bench_names


@dataclass
class MasterSourceData:
    benchmark_names: pd.Index
    model_columns: list
    metadata_columns: list
    model_dates: pd.Series
    disc_scores: pd.DataFrame
    sat_scores: pd.DataFrame
    availability: pd.DataFrame


class MasterDataPreprocessor:
    SCORE_PATTERN = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)'

    def __init__(self, config):
        self.config = config
        self.score_re = re.compile(self.SCORE_PATTERN)
        self.slash_re = re.compile(
            rf'^\s*({self.SCORE_PATTERN})\s*/\s*({self.SCORE_PATTERN})\s*(?:\[T\])?\s*$',
            flags=re.IGNORECASE
        )
        self.numeric_re = re.compile(
            rf'^\s*({self.SCORE_PATTERN})\s*%?\s*(?:\[T\])?\s*$',
            flags=re.IGNORECASE
        )

    def load_and_process_master(self, file_path):
        df = pd.read_csv(file_path)
        first_col = df.columns[0]
        model_cols, metadata_cols = self._split_columns(df.columns)

        row_names = df[first_col].fillna('').astype(str).str.strip()
        row_names_lower = row_names.str.lower()

        release_mask = row_names_lower.isin({'release date', 'release_date'})
        if release_mask.any():
            raw_model_dates = df.loc[release_mask, model_cols].iloc[0]
            blank_date_mask = (
                raw_model_dates.isna()
                | raw_model_dates.astype(str).str.strip().eq('')
            )
            model_dates = pd.to_datetime(
                raw_model_dates,
                errors='coerce',
                format='mixed'
            )
            invalid_date_mask = ~blank_date_mask & model_dates.isna()
            if invalid_date_mask.any():
                invalid_values = [
                    f"{model_name}: {raw_model_dates.loc[model_name]!r}"
                    for model_name in raw_model_dates.index[invalid_date_mask]
                ]
                raise ValueError(
                    "Non-empty model release dates could not be parsed: "
                    + "; ".join(invalid_values)
                )
        else:
            model_dates = pd.Series(pd.NaT, index=model_cols)

        metadata_row_mask = row_names_lower.isin({'vendor', 'release date', 'release_date'})
        invalid_name_mask = row_names_lower.isin({'', 'nan', 'unknown'})
        bench_df = df.loc[~metadata_row_mask & ~invalid_name_mask].copy()
        bench_names = pd.Index(bench_df[first_col].astype(str).str.strip(), name='Benchmark')
        raw_scores = bench_df[model_cols].reset_index(drop=True)

        disc_numeric = raw_scores.apply(lambda col: col.map(self._clean_disc_cell))
        sat_numeric = raw_scores.apply(lambda col: col.map(self._clean_sat_cell))
        availability = raw_scores.apply(lambda col: col.map(self._is_evaluated_cell)).astype(bool)

        disc_numeric.index = bench_names
        sat_numeric.index = bench_names
        availability.index = bench_names

        disc_scores = self._normalize_matrix(disc_numeric)
        sat_scores = self._normalize_matrix(sat_numeric)

        return MasterSourceData(
            benchmark_names=bench_names,
            model_columns=list(model_cols),
            metadata_columns=list(metadata_cols),
            model_dates=model_dates,
            disc_scores=disc_scores,
            sat_scores=sat_scores,
            availability=availability
        )

    def _split_columns(self, columns):
        columns = list(columns)
        cut_col = self.config.get('cut_off_column', 'GitHub')

        try:
            cut_idx = next(i for i, c in enumerate(columns) if str(c).lower() == cut_col.lower())
        except StopIteration:
            metadata_names = {
                'github', 'huggingface', 'modelscope', 'benchmark_date',
                'gh_likes', 'gh_forks', 'hf_likes', 'hf_downloads',
                'ms_likes', 'ms_downloads', 'hf_issue'
            }
            metadata_indices = [i for i, c in enumerate(columns) if str(c).lower() in metadata_names]
            cut_idx = min(metadata_indices) if metadata_indices else len(columns)

        return columns[1:cut_idx], columns[cut_idx:]

    def _clean_disc_cell(self, val):
        val_str = self._to_str(val)
        if val_str is None:
            return np.nan

        slash_match = self.slash_re.match(val_str)
        if slash_match:
            return float(slash_match.group(1))

        if '[t]' in val_str.lower():
            return np.nan

        match = self.score_re.search(val_str)
        return float(match.group(0)) if match else np.nan

    def _clean_sat_cell(self, val):
        val_str = self._to_str(val)
        if val_str is None:
            return np.nan

        slash_match = self.slash_re.match(val_str)
        if slash_match:
            return float(slash_match.group(2))

        val_str = re.sub(r'\s*\[T\]\s*$', '', val_str, flags=re.IGNORECASE)
        match = self.score_re.search(val_str)
        return float(match.group(0)) if match else np.nan

    def _is_evaluated_cell(self, val):
        val_str = self._to_str(val)
        if val_str is None:
            return False

        return bool(self.slash_re.match(val_str) or self.numeric_re.match(val_str))

    def _to_str(self, val):
        if pd.isna(val):
            return None
        val_str = str(val).strip()
        if val_str == '':
            return None
        return val_str

    def _normalize_matrix(self, df_numeric):
        df_norm = df_numeric.copy()
        tags = self.config['tags']
        sf = self.config['safety_factor']

        for bench_name in df_numeric.index:
            row = df_numeric.loc[bench_name]
            valid = row.dropna()
            if valid.empty:
                continue

            name = str(bench_name)
            max_v = valid.max()

            if tags['lib_0_1'] in name:
                df_norm.loc[bench_name] = (1 - row) * 100
            elif tags['lib_standard'] in name:
                if max_v > 100:
                    df_norm.loc[bench_name] = (1 - row / (max_v * sf)) * 100
                else:
                    df_norm.loc[bench_name] = 100 - row
            elif tags['unbounded_hib'].lower() in name.lower():
                df_norm.loc[bench_name] = (row / (max_v * sf)) * 100
            else:
                if 0 < max_v <= 1.05:
                    df_norm.loc[bench_name] = row * 100
                else:
                    df_norm.loc[bench_name] = row

        return df_norm
