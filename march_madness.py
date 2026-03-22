"""
March Madness ML Bracket Predictor 2026 run 3/16/26

includes isotonic calibration on output probabilities (Brier score improvement),
ensemble with XGBoost + LightGBM + Logistic Regression for average probablity,
150 parameter combo search

Output:
  - submission_2026.csv
  - model_feature_importance_{M/W}.png
  - hyperparam_search_results.csv
  - bracket_{year}_simulated_{M/W}.csv
  - bracket_{year}_rounds_{M/W}.txt
  - team_name_crosswalk_{M/W}.csv
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from itertools import combinations
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR       = "data"
PREDICT_YEAR   = 2026
TRAIN_UP_TO    = 2025
BACKTEST_YEARS = list(range(2018, 2026))
N_SIMULATIONS  = 10_000
N_PARAM_SEARCH = 150
ELO_K          = 20
ELO_START      = 1500

# Ensemble weights: how much each model contributes to final probability
# XGBoost is strongest, LightGBM adds diversity, LogReg adds calibration stability
ENSEMBLE_WEIGHTS = {"xgb": 0.45, "lgbm": 0.35, "logreg": 0.20}

# Barttorvik CSV path
BARTTORVIK_MENS   = "mens_all_team_results.csv"
BARTTORVIK_WOMENS = None
BT_FEATURE_COLS = [
    "adjoe", "adjde", "barthag", "adjt",
    "sos", "ncsos", "consos",
    "elite SOS", "elite noncon SOS",
    "Opp OE", "Opp DE",
    "Con Adj OE", "Con Adj DE",
    "Qual O", "Qual D", "Qual Barthag", "Qual Games",
    "WAB",
    "Conf Win%",
    "ConOE", "ConDE",
]

M_FILES = {
    "seeds":    "MNCAATourneySeeds.csv",
    "results":  "MNCAATourneyCompactResults.csv",
    "detailed": "MRegularSeasonDetailedResults.csv",
    "teams":    "MTeams.csv",
    "massey":   "MMasseyOrdinals.csv",
    "conf":     "MTeamConferences.csv",
    "coaches":  "MTeamCoaches.csv",
}

W_FILES = {
    "seeds":    "WNCAATourneySeeds.csv",
    "results":  "WNCAATourneyCompactResults.csv",
    "detailed": "WRegularSeasonDetailedResults.csv",
    "teams":    "WTeams.csv",
    "massey":   None,
    "conf":     "WTeamConferences.csv",
    "coaches":  "WTeamCoaches.csv",
}

GENDER_FILES = {"M": M_FILES, "W": W_FILES}

SAMPLE_SUB_STAGE2 = os.path.join(DATA_DIR, "SampleSubmissionStage2.csv")
SAMPLE_SUB_STAGE1 = os.path.join(DATA_DIR, "SampleSubmissionStage1.csv")

MANUAL_NAME_MAP = {
    "UConn":                  "Connecticut",
    "USC":                    "Southern California",
    "UTEP":                   "Texas El Paso",
    "LSU":                    "Louisiana St",
    "SMU":                    "Southern Methodist",
    "UCF":                    "Central Florida",
    "UNC":                    "North Carolina",
    "UNC Wilmington":         "NC Wilmington",
    "UNC Greensboro":         "NC Greensboro",
    "UNC Asheville":          "NC Asheville",
    "UNC A&T":                "NC A&T",
    "UNLV":                   "Nevada Las Vegas",
    "VCU":                    "Virginia Commonwealth",
    "UMBC":                   "MD Baltimore County",
    "UMass":                  "Massachusetts",
    "UMass Lowell":           "Massachusetts Lowell",
    "Miami FL":               "Miami FL",
    "Miami OH":               "Miami OH",
    "St. John's":             "St John's",
    "Saint Mary's":           "St Mary's CA",
    "Saint Joseph's":         "St Joseph's PA",
    "Saint Peter's":          "St Peter's",
    "Saint Louis":            "Saint Louis",
    "Saint Bonaventure":      "St Bonaventure",
    "Saint Francis":          "St Francis PA",
    "LIU":                    "Long Island University",
    "ETSU":                   "East Tennessee St",
    "East Tennessee St.":     "East Tennessee St",
    "SIU Edwardsville":       "SIU Edwardsville",
    "Southern Illinois":      "Southern Illinois",
    "UTSA":                   "UT San Antonio",
    "UT Arlington":           "UT Arlington",
    "UT Martin":              "UT Martin",
    "Texas A&M Corpus Chris": "Texas A&M Corpus Chr",
    "FIU":                    "Florida Intl",
    "NIU":                    "Northern Illinois",
    "BYU":                    "Brigham Young",
    "TCU":                    "Texas Christian",
    "Ole Miss":               "Mississippi",
    "UCSB":                   "UC Santa Barbara",
    "Cal St. Bakersfield":    "CS Bakersfield",
    "Cal St. Fullerton":      "CS Fullerton",
    "Cal St. Northridge":     "CS Northridge",
    "Loyola Chicago":         "Loyola-Chicago",
    "Loyola Marymount":       "Loyola Marymount",
    "Loyola MD":              "Loyola MD",
    "Detroit":                "Detroit Mercy",
    "Little Rock":            "Ark Little Rock",
    "Arkansas Pine Bluff":    "Ark Pine Bluff",
    "East Texas A&M":         "Texas A&M Commerce",
    "Purdue Fort Wayne":      "Purdue Fort Wayne",
    "IU Indianapolis":        "Indiana Upui",
    "Queens":                 "Queens NC",
    "Stonehill":              "Stonehill",
    "Le Moyne":               "Le Moyne",
    "Lindenwood":             "Lindenwood",
    "Southern Indiana":       "Southern Indiana",
    "Texas A&M Commerce":     "Texas A&M Commerce",
    "Chicago St.":            "Chicago St",
    "Pitt":                   "Pittsburgh",
}

def normalize_name(name):
    n = name.lower().strip()
    n = n.replace(".", "").replace("\u2019", "'").replace("\u2018", "'")
    return n

def build_name_crosswalk(bt_df, kaggle_teams_df, gender):
    kaggle_names = dict(zip(kaggle_teams_df["TeamName"], kaggle_teams_df["TeamID"]))
    bt_names = bt_df["team"].unique().tolist()

    mapping = {}
    matched_log = []

    for bt_name in bt_names:
        if bt_name in MANUAL_NAME_MAP:
            target = MANUAL_NAME_MAP[bt_name]
            if target in kaggle_names:
                mapping[bt_name] = kaggle_names[target]
                matched_log.append((bt_name, target, kaggle_names[target], "manual", 1.0))
                continue

        if bt_name in kaggle_names:
            mapping[bt_name] = kaggle_names[bt_name]
            matched_log.append((bt_name, bt_name, kaggle_names[bt_name], "exact", 1.0))
            continue

        bt_norm = normalize_name(bt_name)
        for kn, kid in kaggle_names.items():
            if normalize_name(kn) == bt_norm:
                mapping[bt_name] = kid
                matched_log.append((bt_name, kn, kid, "normalized", 1.0))
                break
        if bt_name in mapping:
            continue

        best_score, best_kn = 0, None
        for kn in kaggle_names:
            score = SequenceMatcher(None, bt_norm, normalize_name(kn)).ratio()
            if score > best_score:
                best_score, best_kn = score, kn
        if best_score >= 0.75:
            mapping[bt_name] = kaggle_names[best_kn]
            matched_log.append((bt_name, best_kn, kaggle_names[best_kn], "fuzzy", best_score))
        else:
            matched_log.append((bt_name, best_kn, None, "UNMATCHED", best_score))

    log_df = pd.DataFrame(matched_log, columns=["bt_name", "kaggle_name", "TeamID", "method", "score"])
    log_df.to_csv(f"team_name_crosswalk_{gender}.csv", index=False)

    n_matched = sum(1 for v in mapping.values() if v is not None)
    n_unmatched = len(bt_names) - n_matched
    print(f"    Name crosswalk: {n_matched} matched, {n_unmatched} unmatched")
    if n_unmatched > 0:
        unmatched = log_df[log_df.method == "UNMATCHED"]
        for _, row in unmatched.head(10).iterrows():
            print(f"      \u26a0 '{row.bt_name}' \u2192 best guess '{row.kaggle_name}' (score={row.score:.2f})")

    return mapping


def load_barttorvik(bt_path, kaggle_teams_df, gender):
    if bt_path is None or not os.path.exists(bt_path):
        print(f"    Barttorvik file not found: {bt_path}")
        return None

    bt = pd.read_csv(bt_path)

    if "Fun Rk, adjt" in bt.columns and "adjt" in bt.columns:
        bt["adjt_merged"] = bt["adjt"].fillna(bt["Fun Rk, adjt"])
        bt["adjt"] = bt["adjt_merged"]
        bt.drop(columns=["adjt_merged", "Fun Rk, adjt"], inplace=True, errors="ignore")

    bt.drop(columns=["Fun Rk", "gender"], inplace=True, errors="ignore")
    bt = bt.rename(columns={"year": "Season"})

    name_map = build_name_crosswalk(bt, kaggle_teams_df, gender)

    bt["TeamID"] = bt["team"].map(name_map)
    bt = bt.dropna(subset=["TeamID"])
    bt["TeamID"] = bt["TeamID"].astype(int)

    available_features = [c for c in BT_FEATURE_COLS if c in bt.columns]
    bt_features = bt[["Season", "TeamID"] + available_features].copy()
    bt_features = bt_features.rename(columns={c: f"bt_{c}" for c in available_features})

    print(f"    Barttorvik features loaded: {len(available_features)} columns, "
          f"{len(bt_features)} team-seasons")

    return bt_features

STAT_COLS = ["FGM","FGA","FGM3","FGA3","FTM","FTA","OR","DR","Ast","TO","Stl","Blk","PF"]


def load_gender_data(gender):
    files = GENDER_FILES[gender]
    def path(key):
        if files[key] is None:
            return None
        return os.path.join(DATA_DIR, files[key])

    data = {}
    data["seeds"]    = pd.read_csv(path("seeds"))
    data["results"]  = pd.read_csv(path("results"))
    data["detailed"] = pd.read_csv(path("detailed"))
    data["teams"]    = pd.read_csv(path("teams"))

    data["seeds"]["SeedNum"] = data["seeds"]["Seed"].str.extract(r"(\d+)").astype(int)
    data["seeds"]["Region"]  = data["seeds"]["Seed"].str[0]

    massey_path = path("massey")
    conf_path   = path("conf")
    coaches_path = path("coaches")

    data["use_massey"]  = massey_path is not None and os.path.exists(massey_path)
    data["use_conf"]    = conf_path is not None and os.path.exists(conf_path)
    data["use_coaches"] = coaches_path is not None and os.path.exists(coaches_path)

    if data["use_massey"]:
        data["massey"] = pd.read_csv(massey_path)
    if data["use_conf"]:
        data["conf"] = pd.read_csv(conf_path)
    if data["use_coaches"]:
        data["coaches"] = pd.read_csv(coaches_path)

    return data


def compute_coach_experience(coaches_df, tourney_results_df):
    """
    Compute coach tournament experience features for each team-season.
    Returns DataFrame with Season, TeamID, coach_tourney_games, coach_tourney_wins.
    """
    # Figure out which coach was coaching each team at season end (max DayNum)
    # Coaches file has Season, TeamID, FirstDayNum, LastDayNum, CoachName
    # We want the coach active at DayNum=154 (tournament time)
    coaches = coaches_df.copy()
    season_coaches = coaches[coaches["LastDayNum"] >= 132].copy()
    season_coaches = season_coaches.sort_values("LastDayNum").groupby(
        ["Season", "TeamID"]
    ).last().reset_index()[["Season", "TeamID", "CoachName"]]

    # Build tournament game history per coach
    tourney = tourney_results_df.copy()

    # Merge coach for winner
    tourney_w = tourney.merge(
        season_coaches.rename(columns={"TeamID": "WTeamID", "CoachName": "WCoach"}),
        on=["Season", "WTeamID"], how="left"
    )
    # Merge coach for loser
    tourney_wl = tourney_w.merge(
        season_coaches.rename(columns={"TeamID": "LTeamID", "CoachName": "LCoach"}),
        on=["Season", "LTeamID"], how="left"
    )
    coach_records = []

    # Winning coach: played a game, won it
    for _, row in tourney_wl.iterrows():
        if pd.notna(row.get("WCoach")):
            coach_records.append({
                "Season": row["Season"], "CoachName": row["WCoach"],
                "Games": 1, "Wins": 1
            })
        if pd.notna(row.get("LCoach")):
            coach_records.append({
                "Season": row["Season"], "CoachName": row["LCoach"],
                "Games": 1, "Wins": 0
            })

    if not coach_records:
        return pd.DataFrame(columns=["Season", "TeamID", "coach_tourney_games",
                                      "coach_tourney_wins", "coach_tourney_win_pct"])

    cr = pd.DataFrame(coach_records)
    coach_season = cr.groupby(["CoachName", "Season"]).agg(
        Games=("Games", "sum"), Wins=("Wins", "sum")
    ).reset_index().sort_values(["CoachName", "Season"])

    coach_season["cum_games"] = coach_season.groupby("CoachName")["Games"].cumsum().shift(1)
    coach_season["cum_wins"]  = coach_season.groupby("CoachName")["Wins"].cumsum().shift(1)

    all_coach_seasons = season_coaches[["Season", "TeamID", "CoachName"]].copy()
    coach_cumulative = coach_season[["CoachName", "Season", "cum_games", "cum_wins"]].copy()

    # Merge for each team-season, look up coach, then look up coach's prior experience
    result = all_coach_seasons.merge(coach_cumulative, on=["CoachName", "Season"], how="left")
    result["cum_games"] = result["cum_games"].fillna(0)
    result["cum_wins"]  = result["cum_wins"].fillna(0)
    result["coach_tourney_win_pct"] = (
        result["cum_wins"] / result["cum_games"].replace(0, np.nan)
    ).fillna(0)

    result = result.rename(columns={
        "cum_games": "coach_tourney_games",
        "cum_wins":  "coach_tourney_wins",
    })[["Season", "TeamID", "coach_tourney_games", "coach_tourney_wins",
        "coach_tourney_win_pct"]]

    return result


def compute_team_stats(detailed):
    records = []
    for prefix, opp in [("W", "L"), ("L", "W")]:
        col_map = {f"{prefix}{s}": s for s in STAT_COLS}
        col_map[f"{prefix}Score"] = "Score"
        col_map[f"{opp}Score"]    = "OppScore"
        df = detailed[["Season", "DayNum", f"{prefix}TeamID"] + list(col_map.keys())].copy()
        df = df.rename(columns={f"{prefix}TeamID": "TeamID", **col_map})
        df["Win"] = 1 if prefix == "W" else 0
        records.append(df)

    g = pd.concat(records, ignore_index=True).sort_values(["Season", "TeamID", "DayNum"])

    g["FGPct"]  = g["FGM"]  / g["FGA"].replace(0, np.nan)
    g["FG3Pct"] = g["FGM3"] / g["FGA3"].replace(0, np.nan)
    g["FTPct"]  = g["FTM"]  / g["FTA"].replace(0, np.nan)
    g["Margin"] = g["Score"] - g["OppScore"]
    g["Poss"]   = g["FGA"] - g["OR"] + g["TO"] + 0.475 * g["FTA"]
    g["OffEff"] = 100 * g["Score"]    / g["Poss"].replace(0, np.nan)
    g["DefEff"] = 100 * g["OppScore"] / g["Poss"].replace(0, np.nan)
    g["NetEff"] = g["OffEff"] - g["DefEff"]

    all_feat = STAT_COLS + [
        "Score", "OppScore", "Win", "FGPct", "FG3Pct", "FTPct",
        "Margin", "Poss", "OffEff", "DefEff", "NetEff",
    ]

    full = g.groupby(["Season", "TeamID"])[all_feat].mean().reset_index()

    late = (g[g.DayNum > 110]
            .groupby(["Season", "TeamID"])[all_feat]
            .mean().reset_index()
            .rename(columns={c: f"late_{c}" for c in all_feat}))

    def last_n(group, n):
        return group.tail(n)[["Win", "Margin", "OffEff", "DefEff"]].mean()

    form5  = g.groupby(["Season", "TeamID"]).apply(lambda x: last_n(x, 5)).reset_index()
    form10 = g.groupby(["Season", "TeamID"]).apply(lambda x: last_n(x, 10)).reset_index()
    form5  = form5.rename(columns={"Win": "last5_win", "Margin": "last5_margin",
                                    "OffEff": "last5_off", "DefEff": "last5_def"})
    form10 = form10.rename(columns={"Win": "last10_win", "Margin": "last10_margin",
                                     "OffEff": "last10_off", "DefEff": "last10_def"})

    result = (full
              .merge(late,   on=["Season", "TeamID"], how="left")
              .merge(form5,  on=["Season", "TeamID"], how="left")
              .merge(form10, on=["Season", "TeamID"], how="left"))
    return result


def compute_elo(detailed, K=ELO_K, start=ELO_START):
    records = []
    for season, sg in detailed.groupby("Season"):
        elo = defaultdict(lambda: start)
        for _, row in sg.sort_values("DayNum").iterrows():
            a, b = row["WTeamID"], row["LTeamID"]
            ea = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
            elo[a] += K * (1 - ea)
            elo[b] += K * (0 - (1 - ea))
        for tid, rating in elo.items():
            records.append({"Season": season, "TeamID": tid, "Elo": rating})
    return pd.DataFrame(records)


def build_team_features(data, bt_features=None):
    team_stats = compute_team_stats(data["detailed"])

    elo_df = compute_elo(data["detailed"])
    team_stats = team_stats.merge(elo_df, on=["Season", "TeamID"], how="left")
    team_stats["Elo"] = team_stats["Elo"].fillna(ELO_START)

    if data["use_massey"]:
        massey_avg = (data["massey"][data["massey"].RankingDayNum >= 120]
                      .groupby(["Season", "TeamID"])["OrdinalRank"]
                      .mean().reset_index()
                      .rename(columns={"OrdinalRank": "AvgRank"}))
        team_stats = team_stats.merge(massey_avg, on=["Season", "TeamID"], how="left")
        team_stats["AvgRank"] = team_stats["AvgRank"].fillna(200)

    if data["use_conf"]:
        conf_df = data["conf"]
        conf_stats = team_stats.merge(
            conf_df[["Season", "TeamID", "ConfAbbrev"]],
            on=["Season", "TeamID"], how="left"
        )
        conf_agg = (conf_stats.groupby(["Season", "ConfAbbrev"])
                    .agg(conf_margin=("Margin", "mean"),
                         conf_win_pct=("Win", "mean"),
                         conf_net_eff=("NetEff", "mean")).reset_index())
        team_stats = (team_stats
                      .merge(conf_df[["Season", "TeamID", "ConfAbbrev"]],
                             on=["Season", "TeamID"], how="left")
                      .merge(conf_agg, on=["Season", "ConfAbbrev"], how="left")
                      .drop(columns=["ConfAbbrev"]))

    if data["use_coaches"]:
        coach_exp = compute_coach_experience(data["coaches"], data["results"])
        if not coach_exp.empty:
            team_stats = team_stats.merge(coach_exp, on=["Season", "TeamID"], how="left")
            team_stats["coach_tourney_games"]   = team_stats["coach_tourney_games"].fillna(0)
            team_stats["coach_tourney_wins"]     = team_stats["coach_tourney_wins"].fillna(0)
            team_stats["coach_tourney_win_pct"]  = team_stats["coach_tourney_win_pct"].fillna(0)

    if bt_features is not None:
        pre_cols = len(team_stats.columns)
        team_stats = team_stats.merge(bt_features, on=["Season", "TeamID"], how="left")
        post_cols = len(team_stats.columns)
        n_bt = post_cols - pre_cols

        if "bt_adjoe" in team_stats.columns and "bt_adjde" in team_stats.columns:
            team_stats["bt_net_eff"] = team_stats["bt_adjoe"] - team_stats["bt_adjde"]
        if "bt_Qual O" in team_stats.columns and "bt_Qual D" in team_stats.columns:
            team_stats["bt_qual_net"] = team_stats["bt_Qual O"] - team_stats["bt_Qual D"]
        if "bt_Con Adj OE" in team_stats.columns and "bt_Con Adj DE" in team_stats.columns:
            team_stats["bt_con_net"] = team_stats["bt_Con Adj OE"] - team_stats["bt_Con Adj DE"]

        bt_cols = [c for c in team_stats.columns if c.startswith("bt_")]
        for c in bt_cols:
            team_stats[c] = team_stats[c].fillna(team_stats[c].median())

    return team_stats


def get_stat_feature_cols(team_stats):
    return [c for c in team_stats.columns if c not in ["Season", "TeamID"]]


def make_matchup_row(t1, t2, season, team_stats, seeds_df, stat_feature_cols):
    s1 = team_stats[(team_stats.Season == season) & (team_stats.TeamID == t1)]
    s2 = team_stats[(team_stats.Season == season) & (team_stats.TeamID == t2)]
    if s1.empty or s2.empty:
        return None

    diff  = s1[stat_feature_cols].values[0] - s2[stat_feature_cols].values[0]
    feats = {f"diff_{c}": float(v) for c, v in zip(stat_feature_cols, diff)}

    sd    = seeds_df[seeds_df.Season == season]
    seed1 = sd[sd.TeamID == t1]["SeedNum"].values
    seed2 = sd[sd.TeamID == t2]["SeedNum"].values
    seed1 = int(seed1[0]) if len(seed1) else 8
    seed2 = int(seed2[0]) if len(seed2) else 8

    feats["seed_diff"]  = float(seed1 - seed2)
    feats["seed_sum"]   = float(seed1 + seed2)
    feats["seed_ratio"] = float(seed1) / max(float(seed2), 1)

    elo_diff = feats.get("diff_Elo", 0.0)
    net_diff = feats.get("diff_NetEff", 0.0)
    mar_diff = feats.get("diff_Margin", 0.0)
    feats["interact_seed_elo"]   = feats["seed_diff"] * elo_diff
    feats["interact_seed_net"]   = feats["seed_diff"] * net_diff
    feats["interact_margin_elo"] = mar_diff * elo_diff

    bt_net = feats.get("diff_bt_net_eff", 0.0)
    bt_wab = feats.get("diff_bt_WAB", 0.0)
    bt_barthag = feats.get("diff_bt_barthag", 0.0)
    feats["interact_seed_bt_net"]     = feats["seed_diff"] * bt_net
    feats["interact_seed_bt_wab"]     = feats["seed_diff"] * bt_wab
    feats["interact_bt_barthag_elo"]  = bt_barthag * elo_diff

    coach_diff = feats.get("diff_coach_tourney_games", 0.0)
    feats["interact_seed_coach_exp"]  = feats["seed_diff"] * coach_diff

    return feats


def get_col_names(detailed, team_stats, seeds):
    stat_feature_cols = get_stat_feature_cols(team_stats)
    for season in sorted(detailed.Season.unique()):
        for _, row in detailed[detailed.Season == season].head(200).iterrows():
            sample = make_matchup_row(
                row["WTeamID"], row["LTeamID"],
                row["Season"], team_stats, seeds, stat_feature_cols
            )
            if sample:
                return list(sample.keys())
    return None


def build_dataset(tourney_df, team_stats, seeds, col_names, stat_feature_cols,
                  recent_weight=True):
    rows, labels, weights = [], [], []
    min_yr = tourney_df.Season.min()
    max_yr = tourney_df.Season.max()

    for _, row in tourney_df.iterrows():
        winner, loser = row["WTeamID"], row["LTeamID"]
        season = row["Season"]
        sw = (1.0 + 2.0 * (season - min_yr) / max(max_yr - min_yr, 1)
              if recent_weight else 1.0)

        feats_win = make_matchup_row(winner, loser, season, team_stats, seeds,
                                     stat_feature_cols)
        if feats_win is None:
            continue
        rows.append(list(feats_win.values()))
        labels.append(1)
        weights.append(sw)

        feats_lose = make_matchup_row(loser, winner, season, team_stats, seeds,
                                      stat_feature_cols)
        if feats_lose is None:
            continue
        rows.append(list(feats_lose.values()))
        labels.append(0)
        weights.append(sw)

    X = pd.DataFrame(rows, columns=col_names)
    return X, np.array(labels), np.array(weights)



gender_data = {}
bt_paths = {"M": BARTTORVIK_MENS, "W": BARTTORVIK_WOMENS}

for g in ["M", "W"]:
    gd = load_gender_data(g)

    bt_path = bt_paths[g]
    bt_features = None
    if bt_path:
        bt_features = load_barttorvik(bt_path, gd["teams"], g)

    gd["team_stats"] = build_team_features(gd, bt_features)
    gd["stat_feature_cols"] = get_stat_feature_cols(gd["team_stats"])
    gd["col_names"] = get_col_names(gd["detailed"], gd["team_stats"], gd["seeds"])

    if gd["col_names"] is None:
        print(f"  [{g}] WARNING: Could not determine feature columns. Skipping.")
        continue

    gender_data[g] = gd

    n_bt = len([c for c in gd["col_names"] if "bt_" in c])
    n_coach = len([c for c in gd["col_names"] if "coach" in c])
    print(f"  [{g}] Total matchup features: {len(gd['col_names'])} "
          f"({n_bt} Barttorvik, {n_coach} coach)")


param_grid = {
    "n_estimators":     [200, 300, 400, 500, 600, 800],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma":            [0, 0.1, 0.2, 0.5],
}

sampled_params = list(ParameterSampler(param_grid, n_iter=N_PARAM_SEARCH, random_state=42))

all_search_results = []
best_by_gender = {}

for g, gd in gender_data.items():
    print(f"\n{'='*60}")
    print(f"  [{g}] Hyperparameter search ({BACKTEST_YEARS[0]}\u2013{BACKTEST_YEARS[-1]})")
    print(f"  [{g}] {N_PARAM_SEARCH} combos, optimizing Brier score")
    print(f"{'='*60}")

    tourney = gd["results"]
    ts      = gd["team_stats"]
    sd      = gd["seeds"]
    cn      = gd["col_names"]
    sfc     = gd["stat_feature_cols"]

    search_results = []

    for i, params in enumerate(sampled_params):
        year_briers = []
        for val_year in BACKTEST_YEARS:
            train_df = tourney[tourney.Season < val_year]
            val_df   = tourney[tourney.Season == val_year]
            if len(val_df) == 0:
                continue

            Xtr, ytr, wtr = build_dataset(train_df, ts, sd, cn, sfc)
            if Xtr.empty:
                continue

            m = XGBClassifier(**params, eval_metric="logloss",
                              random_state=42, verbosity=0)
            m.fit(Xtr, ytr, sample_weight=wtr)

            brier_scores = []
            for _, row in val_df.iterrows():
                winner, loser = row["WTeamID"], row["LTeamID"]
                t1, t2 = min(winner, loser), max(winner, loser)
                label = 1 if t1 == winner else 0

                feats = make_matchup_row(t1, t2, val_year, ts, sd, sfc)
                if feats is None:
                    continue
                prob = m.predict_proba(
                    pd.DataFrame([list(feats.values())], columns=cn)
                )[0][1]
                brier_scores.append((prob - label) ** 2)

            if brier_scores:
                year_briers.append(np.mean(brier_scores))

        if year_briers:
            mean_brier = np.mean(year_briers)
            std_brier  = np.std(year_briers)
            search_results.append({
                **params,
                "gender":     g,
                "mean_brier": mean_brier,
                "std_brier":  std_brier,
                "scores":     year_briers,
            })
            if (i + 1) % 10 == 0:
                best_so_far = min(search_results, key=lambda x: x["mean_brier"])
                print(f"  [{g}] [{i+1}/{N_PARAM_SEARCH}] best Brier: "
                      f"{best_so_far['mean_brier']:.4f}")

    best = min(search_results, key=lambda x: x["mean_brier"])
    best_params = {k: best[k] for k in param_grid.keys()}
    best_by_gender[g] = {
        "params": best_params, "scores": best["scores"],
        "mean_brier": best["mean_brier"], "std_brier": best["std_brier"],
    }

    print(f"\n  [{g}] Best XGB Brier: {best['mean_brier']:.4f} \u00b1 {best['std_brier']:.4f}")
    print(f"  [{g}] Best params: {best_params}")
    print(f"  [{g}] Year-by-year Brier:")
    for yr, bs in zip(BACKTEST_YEARS, best["scores"]):
        print(f"        {yr}: {bs:.4f}")

    all_search_results.extend(search_results)

search_df = pd.DataFrame([
    {k: r[k] for k in list(param_grid.keys()) + ["gender", "mean_brier", "std_brier"]}
    for r in all_search_results
]).sort_values("mean_brier")
search_df.to_csv("hyperparam_search_results.csv", index=False)
print("\nSaved: hyperparam_search_results.csv")


ensemble_models = {}

for g, gd in gender_data.items():
    print(f"\n[{g}] Training ensemble on all data up to {TRAIN_UP_TO}...")

    tourney = gd["results"]
    ts      = gd["team_stats"]
    sd      = gd["seeds"]
    cn      = gd["col_names"]
    sfc     = gd["stat_feature_cols"]
    bp      = best_by_gender[g]["params"]

    tourney_train = tourney[tourney.Season <= TRAIN_UP_TO]
    X, y, w = build_dataset(tourney_train, ts, sd, cn, sfc)

    models_g = {}

    xgb_base = XGBClassifier(**bp, eval_metric="logloss", random_state=42, verbosity=0)
    xgb_cal = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
    xgb_cal.fit(X, y, sample_weight=w)
    models_g["xgb"] = xgb_cal

    # Also train uncalibrated for feature importance plot
    xgb_raw = XGBClassifier(**bp, eval_metric="logloss", random_state=42, verbosity=0)
    xgb_raw.fit(X, y, sample_weight=w)

    print(f"  [{g}] Training LightGBM...")
    lgbm_params = {
        "n_estimators":     bp.get("n_estimators", 400),
        "max_depth":        bp.get("max_depth", 4),
        "learning_rate":    bp.get("learning_rate", 0.05),
        "subsample":        bp.get("subsample", 0.8),
        "colsample_bytree": bp.get("colsample_bytree", 0.8),
        "min_child_weight": bp.get("min_child_weight", 3),
        "num_leaves":       31,
        "random_state":     42,
        "verbosity":        -1,
    }
    lgbm_base = LGBMClassifier(**lgbm_params)
    lgbm_cal = CalibratedClassifierCV(lgbm_base, method="isotonic", cv=5)
    lgbm_cal.fit(X, y, sample_weight=w)
    models_g["lgbm"] = lgbm_cal

    # log reg
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=cn)
    logreg_base = LogisticRegression(
        C=1.0, max_iter=2000, random_state=42, solver="lbfgs"
    )
    logreg_cal = CalibratedClassifierCV(logreg_base, method="isotonic", cv=5)
    logreg_cal.fit(X_scaled, y, sample_weight=w)
    models_g["logreg"] = logreg_cal
    models_g["scaler"] = scaler

    ensemble_models[g] = models_g

    importance = pd.Series(xgb_raw.feature_importances_, index=cn).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    top30 = importance.head(30)
    colors = ["#e74c3c" if "bt_" in f else "#3498db" if "coach" in f else "steelblue"
              for f in top30.index]
    top30.plot(kind="barh", ax=ax, color=colors)
    ax.invert_yaxis()
    title = f"Top 30 Features \u2014 {'Mens' if g == 'M' else 'Womens'} (v6 Ensemble)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Barttorvik"),
        Patch(facecolor="#3498db", label="Coach Exp"),
        Patch(facecolor="steelblue", label="Kaggle/Computed"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fname = f"model_feature_importance_{g}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [{g}] Saved: {fname}")

    print(f"  [{g}] Top 10 features:")
    for feat, imp in importance.head(10).items():
        src = "\u2605 BT" if "bt_" in feat else "\u2605 Coach" if "coach" in feat else ""
        print(f"        {feat:40s} {imp:.4f} {src}")



def ensemble_predict(X_df, models_g, col_names):
    """
    Get ensemble probability: weighted average of calibrated XGB, LGBM, LogReg.
    """
    probs = {}

    # XGBoost
    probs["xgb"] = models_g["xgb"].predict_proba(X_df)[:, 1]
    # LightGBM
    probs["lgbm"] = models_g["lgbm"].predict_proba(X_df)[:, 1]
    # Logistic Regression (needs scalng))
    X_scaled = pd.DataFrame(
        models_g["scaler"].transform(X_df), columns=col_names
    )
    probs["logreg"] = models_g["logreg"].predict_proba(X_scaled)[:, 1]
    # Weighted average
    final = (ENSEMBLE_WEIGHTS["xgb"]    * probs["xgb"] +
             ENSEMBLE_WEIGHTS["lgbm"]   * probs["lgbm"] +
             ENSEMBLE_WEIGHTS["logreg"] * probs["logreg"])

    return final

print(f"\n{'='*60}")
print(f"  ENSEMBLE BACKTEST ({BACKTEST_YEARS[0]}\u2013{BACKTEST_YEARS[-1]})")
print(f"{'='*60}")

for g, gd in gender_data.items():
    tourney = gd["results"]
    ts      = gd["team_stats"]
    sd      = gd["seeds"]
    cn      = gd["col_names"]
    sfc     = gd["stat_feature_cols"]
    bp      = best_by_gender[g]["params"]

    year_briers_ensemble = []
    year_briers_xgb_only = []

    for val_year in BACKTEST_YEARS:
        train_df = tourney[tourney.Season < val_year]
        val_df   = tourney[tourney.Season == val_year]
        if len(val_df) == 0:
            continue

        Xtr, ytr, wtr = build_dataset(train_df, ts, sd, cn, sfc)
        if Xtr.empty:
            continue

        # Train mini-ensemble for this fold
        xgb_m = CalibratedClassifierCV(
            XGBClassifier(**bp, eval_metric="logloss", random_state=42, verbosity=0),
            method="isotonic", cv=3
        )
        xgb_m.fit(Xtr, ytr, sample_weight=wtr)

        lgbm_params_bt = {
            "n_estimators": bp.get("n_estimators", 400),
            "max_depth": bp.get("max_depth", 4),
            "learning_rate": bp.get("learning_rate", 0.05),
            "subsample": bp.get("subsample", 0.8),
            "colsample_bytree": bp.get("colsample_bytree", 0.8),
            "num_leaves": 31, "random_state": 42, "verbosity": -1,
        }
        lgbm_m = CalibratedClassifierCV(
            LGBMClassifier(**lgbm_params_bt), method="isotonic", cv=3
        )
        lgbm_m.fit(Xtr, ytr, sample_weight=wtr)

        scaler_bt = StandardScaler()
        Xtr_sc = pd.DataFrame(scaler_bt.fit_transform(Xtr), columns=cn)
        lr_m = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=2000, random_state=42),
            method="isotonic", cv=3
        )
        lr_m.fit(Xtr_sc, ytr, sample_weight=wtr)

        brier_ens, brier_xgb = [], []
        for _, row in val_df.iterrows():
            winner, loser = row["WTeamID"], row["LTeamID"]
            t1, t2 = min(winner, loser), max(winner, loser)
            label = 1 if t1 == winner else 0

            feats = make_matchup_row(t1, t2, val_year, ts, sd, sfc)
            if feats is None:
                continue
            Xv = pd.DataFrame([list(feats.values())], columns=cn)

            # XGB only
            p_xgb = xgb_m.predict_proba(Xv)[0][1]
            brier_xgb.append((p_xgb - label) ** 2)

            # Ensemble
            p_lgbm = lgbm_m.predict_proba(Xv)[0][1]
            Xv_sc = pd.DataFrame(scaler_bt.transform(Xv), columns=cn)
            p_lr = lr_m.predict_proba(Xv_sc)[0][1]
            p_ens = (ENSEMBLE_WEIGHTS["xgb"] * p_xgb +
                     ENSEMBLE_WEIGHTS["lgbm"] * p_lgbm +
                     ENSEMBLE_WEIGHTS["logreg"] * p_lr)
            brier_ens.append((p_ens - label) ** 2)

        if brier_ens:
            year_briers_ensemble.append(np.mean(brier_ens))
            year_briers_xgb_only.append(np.mean(brier_xgb))

    if year_briers_ensemble:
        print(f"\n  [{g}] Year-by-year comparison:")
        print(f"        {'Year':>6}  {'XGB+Cal':>10}  {'Ensemble':>10}  {'Diff':>8}")
        for yr, bx, be in zip(BACKTEST_YEARS, year_briers_xgb_only, year_briers_ensemble):
            diff = be - bx
            arrow = "\u2193" if diff < 0 else "\u2191"
            print(f"        {yr:>6}  {bx:>10.4f}  {be:>10.4f}  {diff:>+8.4f} {arrow}")

        mean_xgb = np.mean(year_briers_xgb_only)
        mean_ens = np.mean(year_briers_ensemble)
        diff = mean_ens - mean_xgb
        print(f"        {'MEAN':>6}  {mean_xgb:>10.4f}  {mean_ens:>10.4f}  {diff:>+8.4f}")



print(f"\nGenerating all-pairs predictions for {PREDICT_YEAR}...")

submission_rows = []

for g, gd in gender_data.items():
    ts  = gd["team_stats"]
    sd  = gd["seeds"]
    cn  = gd["col_names"]
    sfc = gd["stat_feature_cols"]
    mg  = ensemble_models[g]

    year_teams = sorted(ts[ts.Season == PREDICT_YEAR]["TeamID"].unique().tolist())

    print(f"  [{g}] Teams with stats: {len(year_teams)}")
    print(f"  [{g}] Matchups: {len(year_teams) * (len(year_teams) - 1) // 2}")

    n_predicted = 0
    batch_rows = []
    batch_ids = []

    for t1, t2 in combinations(year_teams, 2):
        feats = make_matchup_row(t1, t2, PREDICT_YEAR, ts, sd, sfc)
        if feats is None:
            submission_rows.append({"ID": f"{PREDICT_YEAR}_{t1}_{t2}", "Pred": 0.5})
        else:
            batch_rows.append(list(feats.values()))
            batch_ids.append(f"{PREDICT_YEAR}_{t1}_{t2}")
        n_predicted += 1

    if batch_rows:
        X_batch = pd.DataFrame(batch_rows, columns=cn)
        probs = ensemble_predict(X_batch, mg, cn)
        for gid, p in zip(batch_ids, probs):
            submission_rows.append({"ID": gid, "Pred": float(p)})

    print(f"  [{g}] Predictions generated: {n_predicted}")

submission = pd.DataFrame(submission_rows)

sample_path = (SAMPLE_SUB_STAGE2 if os.path.exists(SAMPLE_SUB_STAGE2)
               else SAMPLE_SUB_STAGE1 if os.path.exists(SAMPLE_SUB_STAGE1)
               else None)

if sample_path:
    print(f"\nAligning with: {sample_path}")
    sample = pd.read_csv(sample_path)
    expected_ids = set(sample["ID"])
    our_ids      = set(submission["ID"])

    matched = expected_ids & our_ids
    missing = expected_ids - our_ids
    extra   = our_ids - expected_ids

    print(f"  Matched:  {len(matched)}")
    print(f"  Missing:  {len(missing)} (filling with 0.5)")
    print(f"  Extra:    {len(extra)} (dropping)")

    if missing:
        missing_rows = pd.DataFrame({"ID": list(missing), "Pred": 0.5})
        submission = pd.concat([submission, missing_rows], ignore_index=True)

    submission = submission[submission["ID"].isin(expected_ids)]
else:
    print("\n  No sample submission found \u2014 outputting all computed matchups")

submission = submission.sort_values("ID").reset_index(drop=True)
submission.to_csv("submission_2026.csv", index=False)
print(f"\nSaved: submission_2026.csv ({len(submission)} rows)")


#blind test
for g, gd in gender_data.items():
    tourney = gd["results"]
    ts      = gd["team_stats"]
    sd      = gd["seeds"]
    cn      = gd["col_names"]
    sfc     = gd["stat_feature_cols"]
    mg      = ensemble_models[g]

    test_year = TRAIN_UP_TO
    actual = tourney[tourney.Season == test_year]
    if actual.empty:
        continue

    print(f"\n[{g}] Test on {test_year} (note: included in training)")
    correct, total = 0, 0
    brier_scores = []

    for _, row in actual.iterrows():
        winner, loser = row["WTeamID"], row["LTeamID"]
        t1, t2 = min(winner, loser), max(winner, loser)
        label = 1 if t1 == winner else 0

        feats = make_matchup_row(t1, t2, test_year, ts, sd, sfc)
        if feats is None:
            continue
        Xv = pd.DataFrame([list(feats.values())], columns=cn)
        prob = float(ensemble_predict(Xv, mg, cn)[0])

        pred = 1 if prob >= 0.5 else 0
        correct += int(pred == label)
        total += 1
        brier_scores.append((prob - label) ** 2)

    if total:
        print(f"  [{g}] Accuracy: {correct}/{total} ({correct/total:.1%})")
        print(f"  [{g}] Brier: {np.mean(brier_scores):.4f}")



print(f"\n{'='*60}")
print(f"  MONTE CARLO BRACKET SIMULATION ({N_SIMULATIONS:,} runs)")
print(f"{'='*60}")

REGION_MAP   = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
ROUND1_PAIRS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

for g, gd in gender_data.items():
    label = "MEN'S" if g == "M" else "WOMEN'S"
    print(f"\n  \u2500\u2500 {label} TOURNAMENT \u2500\u2500")

    ts    = gd["team_stats"]
    sd    = gd["seeds"]
    cn    = gd["col_names"]
    sfc   = gd["stat_feature_cols"]
    teams = gd["teams"]
    mg    = ensemble_models[g]

    year_seeds = sd[sd.Season == PREDICT_YEAR].copy()
    if year_seeds.empty:
        print(f"  [{g}] No seed data for {PREDICT_YEAR} yet (appears after Selection Sunday)")
        continue

    year_seeds = year_seeds.merge(teams[["TeamID", "TeamName"]], on="TeamID", how="left")

    tourney_teams = year_seeds["TeamID"].unique().tolist()
    prob_cache = {}
    # Batch compute all pairwise probabilities
    pair_rows, pair_keys = [], []
    for t_a in tourney_teams:
        for t_b in tourney_teams:
            if t_a < t_b and (t_a, t_b) not in prob_cache:
                feats = make_matchup_row(t_a, t_b, PREDICT_YEAR, ts, sd, sfc)
                if feats is None:
                    prob_cache[(t_a, t_b)] = 0.5
                    prob_cache[(t_b, t_a)] = 0.5
                else:
                    pair_rows.append(list(feats.values()))
                    pair_keys.append((t_a, t_b))

    if pair_rows:
        X_pairs = pd.DataFrame(pair_rows, columns=cn)
        pair_probs = ensemble_predict(X_pairs, mg, cn)
        for (ta, tb), p in zip(pair_keys, pair_probs):
            prob_cache[(ta, tb)] = float(p)
            prob_cache[(tb, ta)] = 1 - float(p)

    def simulate_bracket():
        final_four = {}
        for rc, rn in REGION_MAP.items():
            reg = year_seeds[year_seeds.Region == rc]
            s2t = dict(zip(reg.SeedNum, reg.TeamID))
            matchups = [(s2t[s1], s2t[s2]) for s1, s2 in ROUND1_PAIRS
                        if s1 in s2t and s2 in s2t]
            nr = []
            for _ in range(4):
                if not matchups:
                    break
                nr = []
                for ta, tb in matchups:
                    p = prob_cache.get((ta, tb), 0.5)
                    nr.append(ta if np.random.random() < p else tb)
                matchups = [(nr[i], nr[i+1]) for i in range(0, len(nr)-1, 2)]
            if nr:
                final_four[rn] = nr[0]

        ff = list(final_four.values())
        if len(ff) < 4:
            return None
        semi = []
        for ta, tb in [(ff[0], ff[1]), (ff[2], ff[3])]:
            p = prob_cache.get((ta, tb), 0.5)
            semi.append(ta if np.random.random() < p else tb)
        p = prob_cache.get((semi[0], semi[1]), 0.5)
        return semi[0] if np.random.random() < p else semi[1]

    champ_counts = defaultdict(int)
    for _ in range(N_SIMULATIONS):
        c = simulate_bracket()
        if c:
            champ_counts[c] += 1

    champ_df = pd.DataFrame([
        {"Gender": g,
         "TeamID": tid,
         "TeamName": teams[teams.TeamID == tid]["TeamName"].values[0]
                     if len(teams[teams.TeamID == tid]) else str(tid),
         "ChampionPct": cnt / N_SIMULATIONS * 100}
        for tid, cnt in champ_counts.items()
    ]).sort_values("ChampionPct", ascending=False).reset_index(drop=True)

    print(f"\n  Top 10 champions from {N_SIMULATIONS:,} simulations:")
    for _, row in champ_df.head(10).iterrows():
        print(f"     {row['TeamName']:<25} {row['ChampionPct']:.1f}%")

    champ_df.to_csv(f"bracket_{PREDICT_YEAR}_simulated_{g}.csv", index=False)
    print(f"  Saved: bracket_{PREDICT_YEAR}_simulated_{g}.csv")

    bracket_output = []
    final_four_det = {}

    for rc, rn in REGION_MAP.items():
        reg = year_seeds[year_seeds.Region == rc]
        if reg.empty:
            continue
        s2t = dict(zip(reg.SeedNum, reg.TeamID))
        bracket_output.append(f"\n{'='*40}\n  {rn.upper()} REGION\n{'='*40}")
        matchups = [(s2t[s1], s2t[s2]) for s1, s2 in ROUND1_PAIRS
                    if s1 in s2t and s2 in s2t]
        nr = []
        for rnd in ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]:
            if not matchups:
                break
            bracket_output.append(f"\n  {rnd}:")
            nr = []
            for ta, tb in matchups:
                n1 = teams[teams.TeamID == ta]["TeamName"].values
                n2 = teams[teams.TeamID == tb]["TeamName"].values
                n1 = n1[0] if len(n1) else str(ta)
                n2 = n2[0] if len(n2) else str(tb)
                p  = prob_cache.get((ta, tb), 0.5)
                w  = ta if p >= 0.5 else tb
                wn = n1 if w == ta else n2
                wp = p if w == ta else 1 - p
                bracket_output.append(f"    {n1} vs {n2}  \u2192  {wn} ({wp:.0%})")
                nr.append(w)
            matchups = [(nr[i], nr[i+1]) for i in range(0, len(nr)-1, 2)]
        if nr:
            final_four_det[rn] = nr[0]

    if len(final_four_det) >= 4:
        bracket_output.append(f"\n{'='*40}\n  FINAL FOUR\n{'='*40}\n")
        ff = list(final_four_det.values())
        semi_winners = []
        for ta, tb in [(ff[0], ff[1]), (ff[2], ff[3])]:
            n1 = teams[teams.TeamID == ta]["TeamName"].values[0]
            n2 = teams[teams.TeamID == tb]["TeamName"].values[0]
            p  = prob_cache.get((ta, tb), 0.5)
            w  = ta if p >= 0.5 else tb
            wn = n1 if w == ta else n2
            wp = p if w == ta else 1 - p
            bracket_output.append(f"  {n1} vs {n2}  \u2192  {wn} ({wp:.0%})")
            semi_winners.append(w)

        bracket_output.append(f"\n{'='*40}\n  NATIONAL CHAMPIONSHIP\n{'='*40}\n")
        ta, tb = semi_winners
        n1 = teams[teams.TeamID == ta]["TeamName"].values[0]
        n2 = teams[teams.TeamID == tb]["TeamName"].values[0]
        p  = prob_cache.get((ta, tb), 0.5)
        w  = ta if p >= 0.5 else tb
        wn = n1 if w == ta else n2
        wp = p if w == ta else 1 - p
        bracket_output.append(f"  {n1} vs {n2}  \u2192  \U0001f3c6 {wn} ({wp:.0%})")

    fname = f"bracket_{PREDICT_YEAR}_rounds_{g}.txt"
    bp_info = best_by_gender[g]
    with open(fname, "w") as f:
        f.write(f"MARCH MADNESS {PREDICT_YEAR} \u2014 {'MENS' if g == 'M' else 'WOMENS'}\n")
        f.write(f"Model: v6 Ensemble (XGB + LGBM + LogReg) + Isotonic Calibration\n")
        f.write(f"Weights: {ENSEMBLE_WEIGHTS}\n")
        f.write(f"Trained up to {TRAIN_UP_TO}, tuned on "
                f"{BACKTEST_YEARS[0]}\u2013{BACKTEST_YEARS[-1]}\n")
        f.write(f"Best XGB params: {bp_info['params']}\n")
        f.write(f"Best XGB Brier: {bp_info['mean_brier']:.4f}\n\n")
        f.write("\n".join(bracket_output))
    print(f"  Saved: {fname}")


