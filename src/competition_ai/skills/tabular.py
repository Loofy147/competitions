from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class TabularSkill:
    """
    Skill: Tabular Data Engineering.
    Synthesized from multiple high-performing kernels.
    """
    @staticmethod
    def simple_impute(df: pd.DataFrame) -> pd.DataFrame:
        """Simple imputation: mode for categorical, median for numerical."""
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def label_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Applies label encoding to specified columns."""
        df = df.copy()
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    @staticmethod
    def union_find_groups(df: pd.DataFrame, key_columns: List[str]) -> pd.Series:
        """Identifies connected components using Union-Find on multiple keys."""
        n = len(df)
        parent = list(range(n))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra

        pos = {idx: i for i, idx in enumerate(df.index)}
        for col in key_columns:
            seen = {}
            for idx, val in df[col].items():
                if pd.isna(val): continue
                i = pos[idx]
                if val in seen: union(seen[val], i)
                else: seen[val] = i
        return pd.Series([find(pos[idx]) for idx in df.index], index=df.index)

    @staticmethod
    def loo_group_rate(df: pd.DataFrame, group_col: str, target_col: str, mask_col: str) -> pd.Series:
        """Calculates Leave-One-Out survival rate within groups."""
        # Note: target_col must have NaNs for test set
        members = df[df[mask_col] == 1]
        gsum = members.groupby(group_col)[target_col].sum()
        gcnt = members.groupby(group_col)[target_col].count()

        out = np.full(len(df), np.nan)
        g_vals = df[group_col].values
        t_vals = df[target_col].values
        m_vals = df[mask_col].values

        for i in range(len(df)):
            s, c = gsum.get(g_vals[i], 0.0), gcnt.get(g_vals[i], 0)
            if m_vals[i] == 1 and not np.isnan(t_vals[i]):
                s, c = s - t_vals[i], c - 1
            out[i] = (s / c) if c > 0 else np.nan
        return pd.Series(out, index=df.index)
