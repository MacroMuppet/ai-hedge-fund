#!/usr/bin/env python3
"""
NBA in-game score differential builder.

Downloads the season's play-by-play (ESPN data, published daily by the
SportsDataverse project on GitHub) and builds, for every game and BOTH teams,
the running score differential through the game. Everything is concatenated
into analysis-ready tables.

Outputs (in --out, default data/nba/<season>/):
  differential_timeline.csv  one row per scoring change per team-game:
                             game_id, date, team, opp, home_away, period,
                             clock, seconds_elapsed, team_score, opp_score, diff
  team_games.csv             one row per team per game: final score, result,
                             largest lead / deficit, seconds leading/trailing,
                             diff at end of each quarter, first time up N
  team_summary.csv           one row per team: record, how often they led by N+,
                             promo hit rate (won OR led by N), split by opponent tier

Usage:
  pip install pandas pyarrow
  python scripts/nba_score_differential.py                 # 2026-27 season, N = 10
  python scripts/nba_score_differential.py --threshold 12
  python scripts/nba_score_differential.py --season 2026   # 2025-26 season

Season is the year the season ENDS (2027 = 2026-27).
"""

import argparse
import os
import sys
import urllib.request

import pandas as pd

PBP_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/"
           "download/espn_nba_pbp/play_by_play_{season}.parquet")

REG_SECONDS = 4 * 12 * 60


def download(season, raw_dir, refresh):
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"play_by_play_{season}.parquet")
    if refresh or not os.path.exists(path):
        url = PBP_URL.format(season=season)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def clock_to_seconds(clock):
    """'11:41' -> 701.0, '45.2' -> 45.2"""
    clock = str(clock)
    if ":" in clock:
        m, s = clock.split(":")
        return int(m) * 60 + float(s)
    return float(clock)


def seconds_elapsed(period, remaining):
    if period <= 4:
        return (period - 1) * 720 + (720 - remaining)
    return REG_SECONDS + (period - 5) * 300 + (300 - remaining)


def load_plays(path):
    cols = ["game_id", "game_date", "season_type", "game_play_number", "period_number",
            "clock_display_value", "home_score", "away_score", "home_team_abbrev",
            "away_team_abbrev", "type_text"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["season_type"] == 2].copy()  # regular season only
    df["period_number"] = df["period_number"].astype(int)
    df["remaining"] = df["clock_display_value"].map(clock_to_seconds)
    df["seconds_elapsed"] = [seconds_elapsed(p, r) for p, r in
                             zip(df["period_number"], df["remaining"])]
    df = df.sort_values(["game_id", "game_play_number"]).reset_index(drop=True)
    return df


def completed_games(df):
    """Keep games whose play-by-play reaches the end of regulation or later."""
    last = df.groupby("game_id").agg(
        max_period=("period_number", "max"),
        last_type=("type_text", "last"),
        last_remaining=("remaining", "last"))
    done = last[(last["max_period"] >= 4) &
                ((last["last_remaining"] == 0) | last["last_type"].str.contains("End", na=False))]
    dropped = len(last) - len(done)
    if dropped:
        print(f"Skipping {dropped} game(s) without a complete play-by-play.")
    return df[df["game_id"].isin(done.index)]


def build_timeline(df):
    """Two rows per scoring change (one from each team's perspective)."""
    scoring = df[["game_id", "game_date", "period_number", "clock_display_value",
                  "seconds_elapsed", "home_team_abbrev", "away_team_abbrev",
                  "home_score", "away_score"]].copy()
    # keep the opening row plus each row where the score changes
    prev = scoring.groupby("game_id")[["home_score", "away_score"]].shift()
    changed = (prev["home_score"].isna() | (scoring["home_score"] != prev["home_score"]) |
               (scoring["away_score"] != prev["away_score"]))
    scoring = scoring[changed]

    home = pd.DataFrame({
        "game_id": scoring["game_id"], "date": scoring["game_date"],
        "team": scoring["home_team_abbrev"], "opp": scoring["away_team_abbrev"],
        "home_away": "home", "period": scoring["period_number"],
        "clock": scoring["clock_display_value"], "seconds_elapsed": scoring["seconds_elapsed"],
        "team_score": scoring["home_score"], "opp_score": scoring["away_score"]})
    away = home.copy()
    away["team"], away["opp"] = scoring["away_team_abbrev"], scoring["home_team_abbrev"]
    away["home_away"] = "away"
    away["team_score"], away["opp_score"] = scoring["away_score"], scoring["home_score"]
    tl = pd.concat([home, away], ignore_index=True)
    tl["diff"] = tl["team_score"] - tl["opp_score"]
    return tl.sort_values(["date", "game_id", "team", "seconds_elapsed"]).reset_index(drop=True)


def summarize_team_game(g, threshold):
    g = g.sort_values("seconds_elapsed")
    t = g["seconds_elapsed"].to_numpy()
    d = g["diff"].to_numpy()
    end = t[-1]
    # time spent at each differential = until the next scoring change
    durations = list(t[1:] - t[:-1]) + [0]
    lead_secs = sum(dur for dur, x in zip(durations, d) if x > 0)
    trail_secs = sum(dur for dur, x in zip(durations, d) if x < 0)

    def diff_at(sec):
        before = g[g["seconds_elapsed"] <= sec]
        return int(before["diff"].iloc[-1]) if len(before) else 0

    up_n = g[g["diff"] >= threshold]
    down_n = g[g["diff"] <= -threshold]
    final = g.iloc[-1]
    return pd.Series({
        "date": final["date"], "opp": final["opp"], "home_away": final["home_away"],
        "team_score": int(final["team_score"]), "opp_score": int(final["opp_score"]),
        "final_diff": int(final["diff"]), "won": bool(final["diff"] > 0),
        "overtime": bool(end > REG_SECONDS),
        "largest_lead": int(max(d.max(), 0)), "largest_deficit": int(max(-d.min(), 0)),
        f"led_by_{threshold}": bool(len(up_n)),
        f"trailed_by_{threshold}": bool(len(down_n)),
        f"first_up_{threshold}_sec": float(up_n["seconds_elapsed"].iloc[0]) if len(up_n) else None,
        "pct_time_leading": round(lead_secs / end, 3) if end else 0,
        "pct_time_trailing": round(trail_secs / end, 3) if end else 0,
        "diff_end_q1": diff_at(720), "diff_end_q2": diff_at(1440),
        "diff_end_q3": diff_at(2160), "diff_end_reg": diff_at(REG_SECONDS),
    })


def build_team_games(tl, threshold):
    tg = (tl.groupby(["game_id", "team"], sort=False)
            .apply(summarize_team_game, threshold=threshold, include_groups=False)
            .reset_index())
    # opponent strength tier by current win % (top / mid / bot third)
    win_pct = tg.groupby("team")["won"].mean()
    ranks = win_pct.rank(ascending=False, method="first")
    n = len(win_pct)
    tier = ranks.map(lambda r: "top" if r <= n / 3 else ("mid" if r <= 2 * n / 3 else "bot"))
    tg["opp_win_pct"] = tg["opp"].map(win_pct).round(3)
    tg["opp_tier"] = tg["opp"].map(tier)
    return tg.sort_values(["team", "date"]).reset_index(drop=True)


def build_summary(tg, threshold):
    led, trailed = f"led_by_{threshold}", f"trailed_by_{threshold}"
    tg = tg.assign(promo_hit=tg["won"] | tg[led])
    s = tg.groupby("team").agg(
        gp=("won", "size"), wins=("won", "sum"),
        led_n=(led, "sum"), promo_hits=("promo_hit", "sum"),
        trailed_n=(trailed, "sum"),
        avg_largest_lead=("largest_lead", "mean"),
        avg_largest_deficit=("largest_deficit", "mean"),
        avg_pct_time_leading=("pct_time_leading", "mean"),
        avg_final_diff=("final_diff", "mean"))
    s["losses"] = s["gp"] - s["wins"]
    s["led_n_rate"] = s["led_n"] / s["gp"]
    s["promo_rate"] = s["promo_hits"] / s["gp"]
    s["blew_n_lead_and_lost"] = tg[tg[led] & ~tg["won"]].groupby("team").size()
    for t in ("top", "mid", "bot"):
        sub = tg[tg["opp_tier"] == t].groupby("team")
        s[f"led_n_vs_{t}"] = (sub[led].sum().astype(int).astype(str) + "/" +
                              sub.size().astype(str)).reindex(s.index).fillna("0/0")
    s["blew_n_lead_and_lost"] = s["blew_n_lead_and_lost"].fillna(0).astype(int)
    s = s.rename(columns=lambda c: c.replace("_n", f"_{threshold}"))
    s = s.round(3).sort_values(f"led_{threshold}_rate", ascending=False)
    return s.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", type=int, default=2027, help="season end year (2027 = 2026-27)")
    ap.add_argument("--threshold", type=int, default=10, help="lead size to track (default 10)")
    ap.add_argument("--out", default=None, help="output folder (default data/nba/<season>)")
    ap.add_argument("--refresh", action="store_true", help="re-download play-by-play")
    args = ap.parse_args()

    out = args.out or os.path.join("data", "nba", str(args.season))
    path = download(args.season, os.path.join(out, "raw"), args.refresh)

    plays = completed_games(load_plays(path))
    if plays.empty:
        sys.exit("No completed regular-season games in the data.")
    print(f"{plays['game_id'].nunique()} games, {plays['game_date'].min()} to "
          f"{plays['game_date'].max()}")

    tl = build_timeline(plays)
    tg = build_team_games(tl, args.threshold)
    summary = build_summary(tg, args.threshold)

    tl.to_csv(os.path.join(out, "differential_timeline.csv"), index=False)
    tg.to_csv(os.path.join(out, "team_games.csv"), index=False)
    summary.to_csv(os.path.join(out, "team_summary.csv"), index=False)

    t = args.threshold
    show = ["team", "gp", "wins", "losses", f"led_{t}", f"led_{t}_rate", "promo_rate",
            "avg_largest_lead", f"trailed_{t}", f"led_{t}_vs_top", f"led_{t}_vs_mid",
            f"led_{t}_vs_bot"]
    with pd.option_context("display.width", 200, "display.max_rows", 40):
        print(summary[show].to_string(index=False))
    print(f"\nWrote {len(tl):,} timeline rows, {len(tg)} team-games, "
          f"{len(summary)} teams to {out}/")


if __name__ == "__main__":
    main()
