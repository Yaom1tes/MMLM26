"""
March Madness ML Bracket Predictor 2026

Pipeline:
1. Build features: efficiency, Elo, form, seeds, interactions
2. Backtest 2018-2024: RandomizedSearchCV to find best hyperparameters
3. Train final model on all data up to 2024 with best params
4. Blind test on 2025
5. Monte Carlo bracket simulation (10,000 runs)

Output:
    - model_feature_importance.png
    - bracket_2025_simulated.csv
    - bracket_2025_rounds.txt
    - accuracy_2025.txt
    - hyperparam_search_results.csv
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────
SEEDS_PATH    = "data/MNCAATourneySeeds.csv"
RESULTS_PATH  = "data/MNCAATourneyCompactResults.csv"
DETAILED_PATH = "data/MRegularSeasonDetailedResults.csv"
TEAMS_PATH    = "data/MTeams.csv"
MASSEY_PATH   = "data/MMasseyOrdinals.csv"
CONF_PATH     = "data/MTeamConferences.csv"
PREDICT_YEAR  = 2025
TRAIN_UP_TO   = 2024
BACKTEST_YEARS = list(range(2018, 2025))   # years used for hyperparam search
N_SIMULATIONS  = 10000
N_PARAM_SEARCH = 80                        # number of random param combos to try
ELO_K          = 20
ELO_START      = 1500

print("Loading data...")
seeds    = pd.read_csv(SEEDS_PATH)
tourney  = pd.read_csv(RESULTS_PATH)
detailed = pd.read_csv(DETAILED_PATH)
teams    = pd.read_csv(TEAMS_PATH)

use_massey = os.path.exists(MASSEY_PATH)
use_conf   = os.path.exists(CONF_PATH)
if use_massey: massey = pd.read_csv(MASSEY_PATH)
if use_conf:   conf_df = pd.read_csv(CONF_PATH)
print(f"  Massey ordinals: {'yes' if use_massey else 'no'}")
print(f"  Conference data: {'yes' if use_conf else 'no'}")

seeds["SeedNum"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
seeds["Region"]  = seeds["Seed"].str[0]

print("Building team features...")

stat_cols = ["FGM","FGA","FGM3","FGA3","FTM","FTA","OR","DR","Ast","TO","Stl","Blk","PF"]

def compute_team_stats(detailed):
    records = []
    for prefix, opp in [("W","L"), ("L","W")]:
        col_map = {f"{prefix}{s}": s for s in stat_cols}
        col_map[f"{prefix}Score"] = "Score"
        col_map[f"{opp}Score"]    = "OppScore"
        df = detailed[["Season","DayNum",f"{prefix}TeamID"] + list(col_map.keys())].copy()
        df = df.rename(columns={f"{prefix}TeamID": "TeamID", **col_map})
        df["Win"] = 1 if prefix == "W" else 0
        records.append(df)

    g = pd.concat(records, ignore_index=True).sort_values(["Season","TeamID","DayNum"])

    g["FGPct"]  = g["FGM"]  / g["FGA"].replace(0, np.nan)
    g["FG3Pct"] = g["FGM3"] / g["FGA3"].replace(0, np.nan)
    g["FTPct"]  = g["FTM"]  / g["FTA"].replace(0, np.nan)
    g["Margin"] = g["Score"] - g["OppScore"]
    g["Poss"]   = g["FGA"] - g["OR"] + g["TO"] + 0.475 * g["FTA"]
    g["OffEff"] = 100 * g["Score"]    / g["Poss"].replace(0, np.nan)
    g["DefEff"] = 100 * g["OppScore"] / g["Poss"].replace(0, np.nan)
    g["NetEff"] = g["OffEff"] - g["DefEff"]

    all_feat = stat_cols + ["Score","OppScore","Win","FGPct","FG3Pct","FTPct",
                             "Margin","Poss","OffEff","DefEff","NetEff"]

    full = g.groupby(["Season","TeamID"])[all_feat].mean().reset_index()

    late = (g[g.DayNum > 110]
            .groupby(["Season","TeamID"])[all_feat]
            .mean().reset_index()
            .rename(columns={c: f"late_{c}" for c in all_feat}))

    def last_n(group, n):
        return group.tail(n)[["Win","Margin","OffEff","DefEff"]].mean()

    form5  = g.groupby(["Season","TeamID"]).apply(lambda x: last_n(x, 5)).reset_index()
    form10 = g.groupby(["Season","TeamID"]).apply(lambda x: last_n(x, 10)).reset_index()
    form5  = form5.rename(columns={"Win":"last5_win","Margin":"last5_margin",
                                    "OffEff":"last5_off","DefEff":"last5_def"})
    form10 = form10.rename(columns={"Win":"last10_win","Margin":"last10_margin",
                                     "OffEff":"last10_off","DefEff":"last10_def"})

    result = (full
              .merge(late,   on=["Season","TeamID"], how="left")
              .merge(form5,  on=["Season","TeamID"], how="left")
              .merge(form10, on=["Season","TeamID"], how="left"))
    return result

team_stats = compute_team_stats(detailed)

print("Computing Elo ratings...")

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

elo_df     = compute_elo(detailed)
team_stats = team_stats.merge(elo_df, on=["Season","TeamID"], how="left")
team_stats["Elo"] = team_stats["Elo"].fillna(ELO_START)

if use_massey:
    massey_avg = (massey[massey.RankingDayNum >= 120]
                  .groupby(["Season","TeamID"])["OrdinalRank"]
                  .mean().reset_index()
                  .rename(columns={"OrdinalRank": "AvgRank"}))
    team_stats = team_stats.merge(massey_avg, on=["Season","TeamID"], how="left")
    team_stats["AvgRank"] = team_stats["AvgRank"].fillna(200)

if use_conf:
    conf_stats = team_stats.merge(conf_df[["Season","TeamID","ConfAbbrev"]],
                                   on=["Season","TeamID"], how="left")
    conf_agg = (conf_stats.groupby(["Season","ConfAbbrev"])
                .agg(conf_margin=("Margin","mean"),
                     conf_win_pct=("Win","mean"),
                     conf_net_eff=("NetEff","mean")).reset_index())
    team_stats = (team_stats
                  .merge(conf_df[["Season","TeamID","ConfAbbrev"]], on=["Season","TeamID"], how="left")
                  .merge(conf_agg, on=["Season","ConfAbbrev"], how="left")
                  .drop(columns=["ConfAbbrev"]))

stat_feature_cols = [c for c in team_stats.columns if c not in ["Season","TeamID"]]

def make_matchup_row(t1, t2, season, team_stats, seeds_df):
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

    return feats

# get column names from a sample row
col_names = None
for _, row in detailed[detailed.Season == 2010].head(200).iterrows():
    sample = make_matchup_row(row["WTeamID"], row["LTeamID"], 2010, team_stats, seeds)
    if sample:
        col_names = list(sample.keys())
        break

def build_dataset(tourney_df, team_stats, seeds, col_names, recent_weight=True, flip=True):
    """Build X, y, sample_weights from tournament games.
    For each game, always adds two rows:
      - winner as t1 → label 1
      - loser as t1  → label 0 (features recomputed in reverse, not just negated)
    flip=False for validation uses only one row per game (winner as t1, label 1)
    but that means ~50% baseline — so we always use both rows, just with real feature directions.
    """
    rows, labels, weights = [], [], []
    min_yr = tourney_df.Season.min()
    max_yr = tourney_df.Season.max()

    for _, row in tourney_df.iterrows():
        winner, loser = row["WTeamID"], row["LTeamID"]
        season = row["Season"]
        sw = (1.0 + 2.0 * (season - min_yr) / max(max_yr - min_yr, 1)
              if recent_weight else 1.0)

        feats_win = make_matchup_row(winner, loser, season, team_stats, seeds)
        if feats_win is None: continue
        rows.append(list(feats_win.values()))
        labels.append(1)
        weights.append(sw)

        feats_lose = make_matchup_row(loser, winner, season, team_stats, seeds)
        if feats_lose is None: continue
        rows.append(list(feats_lose.values()))
        labels.append(0)
        weights.append(sw)

    X = pd.DataFrame(rows, columns=col_names)
    return X, np.array(labels), np.array(weights)

print(f"\nSearching hyperparameters across {BACKTEST_YEARS[0]}–{BACKTEST_YEARS[-1]}...")
print(f"  Trying {N_PARAM_SEARCH} random combinations...\n")

param_grid = {
    "n_estimators":      [200, 300, 400, 500, 600, 800],
    "max_depth":         [3, 4, 5, 6],
    "learning_rate":     [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":  [1, 3, 5, 7],
    "gamma":             [0, 0.1, 0.2, 0.5],
}

sampled_params = list(ParameterSampler(param_grid, n_iter=N_PARAM_SEARCH, random_state=42))

search_results = []

for i, params in enumerate(sampled_params):
    year_scores = []
    for val_year in BACKTEST_YEARS:
        train_df = tourney[tourney.Season < val_year]
        val_df   = tourney[tourney.Season == val_year]
        if len(val_df) == 0: continue

        Xtr, ytr, wtr = build_dataset(train_df, team_stats, seeds, col_names, flip=True)
        Xval, yval, _ = build_dataset(val_df,   team_stats, seeds, col_names, flip=False)
        if Xval.empty: continue

        m = XGBClassifier(**params, eval_metric="logloss",
                          random_state=42, verbosity=0)
        m.fit(Xtr, ytr, sample_weight=wtr)
        acc = (m.predict(Xval) == yval).mean()
        year_scores.append(acc)

    if year_scores:
        mean_acc = np.mean(year_scores)
        std_acc  = np.std(year_scores)
        search_results.append({**params,
                                "mean_acc": mean_acc,
                                "std_acc":  std_acc,
                                "scores":   year_scores})
        if (i + 1) % 10 == 0:
            best_so_far = max(search_results, key=lambda x: x["mean_acc"])
            print(f"  [{i+1}/{N_PARAM_SEARCH}] best so far: {best_so_far['mean_acc']:.3f}")

best = max(search_results, key=lambda x: x["mean_acc"])
best_params = {k: best[k] for k in param_grid.keys()}
print(f"\n  Best mean accuracy: {best['mean_acc']:.3f} ± {best['std_acc']:.3f}")
print(f"  Best params: {best_params}")

search_df = pd.DataFrame([{k: r[k] for k in list(param_grid.keys()) + ["mean_acc","std_acc"]}
                           for r in search_results]).sort_values("mean_acc", ascending=False)
search_df.to_csv("hyperparam_search_results.csv", index=False)
print("Saved: hyperparam_search_results.csv")

print(f"\n  Year-by-year accuracy with best params:")
for yr, acc in zip(BACKTEST_YEARS, best["scores"]):
    print(f"    {yr}: {acc:.3f}")

print(f"\nTraining final model on all data up to {TRAIN_UP_TO}...")

tourney_train = tourney[tourney.Season <= TRAIN_UP_TO]
X, y, w = build_dataset(tourney_train, team_stats, seeds, col_names)
print(f"  Training samples: {len(X)}")

model = XGBClassifier(**best_params, eval_metric="logloss", random_state=42, verbosity=0)
model.fit(X, y, sample_weight=w)

importance = pd.Series(model.feature_importances_, index=col_names).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 8))
importance.head(25).plot(kind="barh", ax=ax, color="steelblue")
ax.invert_yaxis()
ax.set_title("Top 25 Features (v3 — tuned model)", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("model_feature_importance.png", dpi=150)
plt.close()
print("Saved: model_feature_importance.png")

year_seeds = seeds[seeds.Season == PREDICT_YEAR].copy()
year_stats = team_stats[team_stats.Season == PREDICT_YEAR].copy()

if year_seeds.empty:
    print(f"No seed data for {PREDICT_YEAR}. Exiting.")
    exit()

year_seeds = year_seeds.merge(teams[["TeamID","TeamName"]], on="TeamID", how="left")

def predict_prob(t1, t2):
    feats = make_matchup_row(t1, t2, PREDICT_YEAR, year_stats, seeds)
    if feats is None: return 0.5
    return model.predict_proba(pd.DataFrame([list(feats.values())], columns=col_names))[0][1]

# pre-cache all matchup probs
tourney_teams = year_seeds["TeamID"].unique().tolist()
prob_cache = {}
print("\nPre-computing matchup probabilities...")
for t1 in tourney_teams:
    for t2 in tourney_teams:
        if t1 != t2 and (t1, t2) not in prob_cache:
            p = predict_prob(t1, t2)
            prob_cache[(t1, t2)] = p
            prob_cache[(t2, t1)] = 1 - p

print(f"Simulating {N_SIMULATIONS:,} brackets...")

region_map   = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
round1_pairs = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

def simulate_bracket(year_seeds, region_map, round1_pairs, prob_cache):
    final_four = {}
    for rc, rn in region_map.items():
        reg = year_seeds[year_seeds.Region == rc]
        s2t = dict(zip(reg.SeedNum, reg.TeamID))
        matchups = [(s2t[s1], s2t[s2]) for s1, s2 in round1_pairs
                    if s1 in s2t and s2 in s2t]
        nr = []
        for _ in range(4):
            if not matchups: break
            nr = []
            for t1, t2 in matchups:
                p = prob_cache.get((t1, t2), 0.5)
                nr.append(t1 if np.random.random() < p else t2)
            matchups = [(nr[i], nr[i+1]) for i in range(0, len(nr)-1, 2)]
        if nr: final_four[rn] = nr[0]

    ff = list(final_four.values())
    if len(ff) < 4: return None
    semi = []
    for t1, t2 in [(ff[0], ff[1]), (ff[2], ff[3])]:
        p = prob_cache.get((t1, t2), 0.5)
        semi.append(t1 if np.random.random() < p else t2)
    p = prob_cache.get((semi[0], semi[1]), 0.5)
    return semi[0] if np.random.random() < p else semi[1]

champ_counts = defaultdict(int)
for _ in range(N_SIMULATIONS):
    c = simulate_bracket(year_seeds, region_map, round1_pairs, prob_cache)
    if c: champ_counts[c] += 1

champ_df = pd.DataFrame([
    {"TeamID": tid,
     "TeamName": teams[teams.TeamID == tid]["TeamName"].values[0]
                 if len(teams[teams.TeamID == tid]) else str(tid),
     "ChampionPct": cnt / N_SIMULATIONS * 100}
    for tid, cnt in champ_counts.items()
]).sort_values("ChampionPct", ascending=False).reset_index(drop=True)

print(f"\n🏆 Top 10 champions from {N_SIMULATIONS:,} simulations:")
for _, row in champ_df.head(10).iterrows():
    print(f"   {row['TeamName']:<25} {row['ChampionPct']:.1f}%")

bracket_output, all_results, final_four_det = [], [], {}

for rc, rn in region_map.items():
    reg = year_seeds[year_seeds.Region == rc]
    if reg.empty: continue
    s2t = dict(zip(reg.SeedNum, reg.TeamID))
    bracket_output.append(f"\n{'='*40}\n  {rn.upper()} REGION\n{'='*40}")
    matchups = [(s2t[s1], s2t[s2]) for s1, s2 in round1_pairs if s1 in s2t and s2 in s2t]
    nr = []
    for rnd in ["Round of 64","Round of 32","Sweet 16","Elite 8"]:
        if not matchups: break
        bracket_output.append(f"\n  {rnd}:")
        nr = []
        for t1, t2 in matchups:
            n1 = teams[teams.TeamID == t1]["TeamName"].values
            n2 = teams[teams.TeamID == t2]["TeamName"].values
            n1 = n1[0] if len(n1) else str(t1)
            n2 = n2[0] if len(n2) else str(t2)
            p  = prob_cache.get((t1, t2), 0.5)
            w  = t1 if p >= 0.5 else t2
            wn = n1 if w == t1 else n2
            wp = p if w == t1 else 1 - p
            bracket_output.append(f"    {n1} vs {n2}  →  {wn} ({wp:.0%})")
            all_results.append({"Round": rnd, "Region": rn, "Team1": n1, "Team2": n2,
                                 "PredictedWinner": wn, "WinProbability": round(wp, 3)})
            nr.append(w)
        matchups = [(nr[i], nr[i+1]) for i in range(0, len(nr)-1, 2)]
    if nr: final_four_det[rn] = nr[0]

bracket_output.append(f"\n{'='*40}\n  FINAL FOUR\n{'='*40}\n")
ff = list(final_four_det.values())
ffr = list(final_four_det.keys())
semi_winners = []
for i, (t1, t2) in enumerate([(ff[0], ff[1]), (ff[2], ff[3])]):
    n1 = teams[teams.TeamID == t1]["TeamName"].values[0]
    n2 = teams[teams.TeamID == t2]["TeamName"].values[0]
    p  = prob_cache.get((t1, t2), 0.5)
    w  = t1 if p >= 0.5 else t2
    wn = n1 if w == t1 else n2
    wp = p if w == t1 else 1 - p
    bracket_output.append(f"  {n1} vs {n2}  →  {wn} ({wp:.0%})")
    semi_winners.append(w)

bracket_output.append(f"\n{'='*40}\n  NATIONAL CHAMPIONSHIP\n{'='*40}\n")
t1, t2 = semi_winners
n1 = teams[teams.TeamID == t1]["TeamName"].values[0]
n2 = teams[teams.TeamID == t2]["TeamName"].values[0]
p  = prob_cache.get((t1, t2), 0.5)
w  = t1 if p >= 0.5 else t2
wn = n1 if w == t1 else n2
wp = p if w == t1 else 1 - p
sim_pct = champ_df[champ_df.TeamID == w]["ChampionPct"].values
sim_pct = sim_pct[0] if len(sim_pct) else 0
bracket_output.append(f"  {n1} vs {n2}  →  🏆 {wn} ({wp:.0%})")
print(f"\n🏆 Deterministic champion: {wn}")
print(f"   Win prob vs opponent:    {wp:.0%}")
print(f"   Simulation win rate:     {sim_pct:.1f}%")

pd.DataFrame(all_results).to_csv("bracket_2025.csv", index=False)
champ_df.to_csv("bracket_2025_simulated.csv", index=False)
print("\nSaved: bracket_2025.csv")
print("Saved: bracket_2025_simulated.csv")

with open("bracket_2025_rounds.txt", "w") as f:
    f.write(f"MARCH MADNESS {PREDICT_YEAR} — v3 PREDICTIONS\n")
    f.write(f"Trained up to {TRAIN_UP_TO}, tuned on {BACKTEST_YEARS[0]}–{BACKTEST_YEARS[-1]}\n")
    f.write(f"Best params: {best_params}\n")
    f.write(f"Best mean CV accuracy: {best['mean_acc']:.3f}\n")
    f.write(f"Simulation champion: {wn} ({sim_pct:.1f}% of {N_SIMULATIONS:,} brackets)\n")
    f.write("\n".join(bracket_output))
print("Saved: bracket_2025_rounds.txt")

print("\nBlind test on 2025 tournament")
actual = tourney[tourney.Season == PREDICT_YEAR]
correct, total, wrong = 0, 0, []

for _, row in actual.iterrows():
    t1, t2 = row["WTeamID"], row["LTeamID"]
    p = prob_cache.get((t1, t2), 0.5)
    pred = t1 if p >= 0.5 else t2
    n1 = teams[teams.TeamID == t1]["TeamName"].values
    n2 = teams[teams.TeamID == t2]["TeamName"].values
    n1 = n1[0] if len(n1) else str(t1)
    n2 = n2[0] if len(n2) else str(t2)
    correct += int(pred == t1)
    total   += 1
    if pred != t1:
        wrong.append(f"  WRONG: {n1} beat {n2} — we predicted {n2}")

accuracy = correct / total if total else 0
print(f"  Games correct: {correct}/{total} ({accuracy:.1%})")

with open("accuracy_2025.txt", "w") as f:
    f.write(f"2025 Blind Test Accuracy (v3)\n")
    f.write(f"Tuned on {BACKTEST_YEARS[0]}–{BACKTEST_YEARS[-1]}, tested on {PREDICT_YEAR}\n\n")
    f.write(f"Best hyperparams: {best_params}\n")
    f.write(f"Mean CV accuracy (2018-2024): {best['mean_acc']:.3f} ± {best['std_acc']:.3f}\n\n")
    f.write(f"2025 blind test: {correct}/{total} ({accuracy:.1%})\n\n")
    f.write(f"Year-by-year validation:\n")
    for yr, acc in zip(BACKTEST_YEARS, best["scores"]):
        f.write(f"  {yr}: {acc:.3f}\n")
    f.write(f"\nUpsets missed ({len(wrong)}):\n")
    f.write("\n".join(wrong))
print("Saved: accuracy_2025.txt")