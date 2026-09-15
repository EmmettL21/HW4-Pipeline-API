"""Custom transformer for the NL MVP voting pipeline.

Kept in its own module so both the build script and the FastAPI server
import the exact same class when pickling/unpickling the fitted pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

STAT_COLS = [
    "WAR",
    "R",
    "H",
    "HR",
    "RBI",
    "SB",
    "BB",
    "batting_avg",
    "onbase_perc",
    "slugging_perc",
    "onbase_plus_slugging",
]

YEAR_COL = "Year"


class LeagueSeasonAdjuster(BaseEstimator, TransformerMixin):
    """Converts raw batting stats into season-relative z-scores.

    NL offensive levels shift a lot across 2005-2025 (steroid-era hangover,
    the launch-angle revolution, humidors, the juiced/deadened ball, the
    2020 shortened season, etc.), so a raw HR or OPS total isn't directly
    comparable across seasons. This transformer learns each training
    season's per-stat mean and standard deviation at fit time and reports
    every row as "how many standard deviations above/below that season's
    field it is" - i.e. state that only exists after fitting on real
    season-by-season data, not something a freshly constructed transformer
    could reproduce.

    A season absent from the training data (e.g. a future season at
    inference time) falls back to the pooled across-all-seasons mean/std
    computed at fit time.
    """

    def __init__(self, stat_cols=None, year_col=YEAR_COL):
        # __init__ only assigns constructor args - no computation here.
        self.stat_cols = stat_cols
        self.year_col = year_col

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = self.stat_cols if self.stat_cols is not None else STAT_COLS

        season_stats = {}
        for year, group in X.groupby(self.year_col):
            std = group[cols].std(ddof=0).replace(0, 1.0)
            season_stats[year] = {"mean": group[cols].mean(), "std": std}
        self.season_stats_ = season_stats

        global_std = X[cols].std(ddof=0).replace(0, 1.0)
        self.global_mean_ = X[cols].mean()
        self.global_std_ = global_std
        self.feature_names_out_ = [f"{c}_season_z" for c in cols]
        return self

    def transform(self, X):
        check_is_fitted(self, "season_stats_")
        cols = self.stat_cols if self.stat_cols is not None else STAT_COLS
        X = pd.DataFrame(X)

        means = []
        stds = []
        for year in X[self.year_col]:
            stats = self.season_stats_.get(year)
            if stats is None:
                means.append(self.global_mean_)
                stds.append(self.global_std_)
            else:
                means.append(stats["mean"])
                stds.append(stats["std"])

        mean_df = pd.DataFrame(means, index=X.index)[cols]
        std_df = pd.DataFrame(stds, index=X.index)[cols]
        z = (X[cols].reset_index(drop=True) - mean_df.reset_index(drop=True)) / std_df.reset_index(drop=True)
        return z.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return np.array(self.feature_names_out_, dtype=object)
