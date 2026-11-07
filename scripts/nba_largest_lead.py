#!/usr/bin/env python3
"""
NBA largest in-game lead tracker.

For every completed regular-season game this season, pulls each team's
largest lead from NBA.com's public data and reports how often each team
has led by N+ points (default 10) — overall and split by opponent strength.

Built for "early win" promos that pay out when your team goes up 10.

Data sources (public JSON files from NBA.com, no API key needed):
  - Schedule:   cdn.nba.com/static/json/staticData/scheduleLeagueV2.json
  - Box score:  cdn.nba.com/static/json/liveData/boxscore/boxscore_<id>.json
                (team statistics -> "biggestLead")
  - Fallback:   cdn.nba.com/static/json/liveData/playbyplay/playbyplay_<id>.json
                (largest lead computed from the running score)

Usage examples:
  python scripts/nba_largest_lead.py                     # all teams, 10-pt threshold
  python scripts/nba_largest_lead.py --threshold 12
  python scripts/nba_largest_lead.py --teams NYK,HOU,POR # plus game-by-game logs
  python scripts/nba_largest_lead.py --csv leads.csv     # save per-game data

Results are cached in ./.nba_cache so re-runs only fetch new games.
Only the Python standard library is used.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

CACHE_DIR = ".nba_cache"


# ---------------------------------------------------------------- fetching

def fetch_json(url, retries=3, pause=1.0):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            last_err = e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def cached_game_result(gid):
    path = os.path.join(CACHE_DIR, f"{gid}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_game_result(gid, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{gid}.json"), "w") as f:
        json.dump(data, f)


def largest_leads_from_pbp(gid):
    """Return (home_largest_lead, away_largest_lead) from play-by-play."""
    data = fetch_json(PBP_URL.format(gid=gid))
    home_max = away_max = 0
    for action in data.get("game", {}).get("actions", []):
        try:
            h = int(action.get("scoreHome") or 0)
            a = int(action.get("scoreAway") or 0)
        except (TypeError, ValueError):
            continue
        home_max = max(home_max, h - a)
        away_max = max(away_max, a - h)
    return home_max, away_max


def get_game_leads(gid):
    """Return dict with largest lead and final score for both teams."""
    cached = cached_game_result(gid)
    if cached:
        return cached

    box = fetch_json(BOXSCORE_URL.format(gid=gid))["game"]
    home, away = box["homeTeam"], box["awayTeam"]
    h_lead = home.get("statistics", {}).get("biggestLead")
    a_lead = away.get("statistics", {}).get("biggestLead")
    source = "boxscore"
    if h_lead is None or a_lead is None:
        h_lead, a_lead = largest_leads_from_pbp(gid)
        source = "playbyplay"

    result = {
        "gameId": gid,
        "home": home["teamTricode"],
        "away": away["teamTricode"],
        "homeScore": int(home.get("score", 0)),
        "awayScore": int(away.get("score", 0)),
        "homeLargestLead": max(0, int(h_lead)),
        "awayLargestLead": max(0, int(a_lead)),
        "source": source,
    }
    save_game_result(gid, result)
    return result


def completed_regular_season_games(schedule):
    games = []
    for day in schedule["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            gid = g["gameId"]
            # 002 = regular season (001 preseason, 004 playoffs, 006 NBA Cup final)
            if not gid.startswith("002") or g.get("gameStatus") != 3:
                continue
            date = (g.get("gameDateEst") or day["gameDate"])[:10]
            games.append((date, gid))
    games.sort()
    return games


# ---------------------------------------------------------------- analysis

def build_team_logs(results, dates):
    logs = defaultdict(list)
    for r in results:
        date = dates[r["gameId"]]
        for side, opp in (("home", "away"), ("away", "home")):
            team, opp_team = r[side], r[opp]
            pts, opp_pts = r[f"{side}Score"], r[f"{opp}Score"]
            logs[team].append({
                "date": date,
                "gameId": r["gameId"],
                "team": team,
                "opp": opp_team,
                "home_away": "vs" if side == "home" else "@",
                "pts": pts,
                "opp_pts": opp_pts,
                "won": pts > opp_pts,
                "largest_lead": r[f"{side}LargestLead"],
                "opp_largest_lead": r[f"{opp}LargestLead"],
            })
    for team in logs:
        logs[team].sort(key=lambda g: g["date"])
    return logs


def opponent_tiers(logs):
    """Split teams into top/mid/bottom thirds by current win percentage."""
    pct = {}
    for team, games in logs.items():
        wins = sum(g["won"] for g in games)
        pct[team] = wins / len(games) if games else 0.0
    ranked = sorted(pct, key=lambda t: pct[t], reverse=True)
    n = len(ranked)
    tiers = {}
    for i, team in enumerate(ranked):
        tiers[team] = "top" if i < n / 3 else ("mid" if i < 2 * n / 3 else "bot")
    return tiers, pct


def rate(games, threshold):
    if not games:
        return "  -  "
    hit = sum(g["largest_lead"] >= threshold for g in games)
    return f"{hit}/{len(games)}"


def promo_cash(games, threshold):
    """Games where the promo would have paid: won OR led by threshold."""
    return sum(g["won"] or g["largest_lead"] >= threshold for g in games)


def print_summary(logs, tiers, pct, threshold):
    rows = []
    for team, games in logs.items():
        n = len(games)
        wins = sum(g["won"] for g in games)
        led = sum(g["largest_lead"] >= threshold for g in games)
        cash = promo_cash(games, threshold)
        avg_lead = sum(g["largest_lead"] for g in games) / n
        opp_led = sum(g["opp_largest_lead"] >= threshold for g in games)
        by_tier = {t: [g for g in games if tiers[g["opp"]] == t] for t in ("top", "mid", "bot")}
        rows.append((led / n, team, n, wins, led, cash, avg_lead, opp_led, by_tier))

    rows.sort(key=lambda r: (r[0], r[6]), reverse=True)
    print(f"\nLed by {threshold}+ at any point — completed regular-season games\n")
    print("Opponent tiers = top / mid / bottom third of the league by current win %.")
    print("'Promo' = games that would have paid (won OR led by threshold).")
    print("'Opp led' = games where the OPPONENT led by threshold (how often they get buried).\n")
    hdr = (f"{'Team':<5}{'GP':>4}{'W-L':>7}{'Led':>8}{'Led%':>7}{'Promo':>8}"
           f"{'AvgMax':>8}{'OppLed':>8}{'vsTop':>8}{'vsMid':>8}{'vsBot':>8}")
    print(hdr)
    print("-" * len(hdr))
    for share, team, n, wins, led, cash, avg_lead, opp_led, by_tier in rows:
        print(f"{team:<5}{n:>4}{f'{wins}-{n - wins}':>7}{f'{led}/{n}':>8}{share:>7.0%}"
              f"{f'{cash}/{n}':>8}{avg_lead:>8.1f}{f'{opp_led}/{n}':>8}"
              f"{rate(by_tier['top'], threshold):>8}{rate(by_tier['mid'], threshold):>8}"
              f"{rate(by_tier['bot'], threshold):>8}")


def print_team_log(team, games, tiers, threshold):
    print(f"\n{team} game log (largest lead; '*' = led by {threshold}+)\n")
    print(f"{'Date':<12}{'Opp':<9}{'Tier':<6}{'Result':<12}{'Max lead':>9}{'Opp max':>9}")
    for g in games:
        res = f"{'W' if g['won'] else 'L'} {g['pts']}-{g['opp_pts']}"
        star = "*" if g["largest_lead"] >= threshold else " "
        print(f"{g['date']:<12}{g['home_away'] + ' ' + g['opp']:<9}{tiers[g['opp']]:<6}"
              f"{res:<12}{g['largest_lead']:>8}{star}{g['opp_largest_lead']:>9}")


def write_csv(path, logs, tiers):
    fields = ["date", "gameId", "team", "home_away", "opp", "opp_tier", "pts",
              "opp_pts", "won", "largest_lead", "opp_largest_lead"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for team in sorted(logs):
            for g in logs[team]:
                w.writerow({**{k: g[k] for k in fields if k in g}, "opp_tier": tiers[g["opp"]]})
    print(f"\nWrote per-game data to {path}")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="NBA largest in-game lead tracker")
    p.add_argument("--threshold", type=int, default=10, help="lead size to count (default 10)")
    p.add_argument("--teams", default="", help="comma-separated tricodes for game logs, e.g. NYK,HOU")
    p.add_argument("--csv", default="", help="write per-game rows to this CSV file")
    p.add_argument("--delay", type=float, default=0.3, help="seconds between requests (default 0.3)")
    args = p.parse_args()

    print("Fetching schedule...")
    try:
        schedule = fetch_json(SCHEDULE_URL)
    except Exception as e:
        sys.exit(f"Could not download the NBA schedule ({e}). "
                 "Check your internet connection; some networks block cdn.nba.com.")

    season = schedule["leagueSchedule"].get("seasonYear", "?")
    games = completed_regular_season_games(schedule)
    dates = {gid: date for date, gid in games}
    print(f"Season {season}: {len(games)} completed regular-season games.")

    results = []
    for i, (date, gid) in enumerate(games, 1):
        cached = cached_game_result(gid) is not None
        try:
            results.append(get_game_leads(gid))
        except Exception as e:
            print(f"  skipped {gid} ({date}): {e}")
            continue
        if not cached:
            print(f"  [{i}/{len(games)}] {date} {results[-1]['away']} @ {results[-1]['home']}")
            time.sleep(args.delay)

    if not results:
        sys.exit("No game data retrieved.")

    logs = build_team_logs(results, dates)
    tiers, pct = opponent_tiers(logs)
    print_summary(logs, tiers, pct, args.threshold)

    for team in [t.strip().upper() for t in args.teams.split(",") if t.strip()]:
        if team in logs:
            print_team_log(team, logs[team], tiers, args.threshold)
        else:
            print(f"\nNo games found for '{team}'. Use tricodes like NYK, HOU, POR, PHI.")

    if args.csv:
        write_csv(args.csv, logs, tiers)

    print(f"\nGenerated {datetime.now():%Y-%m-%d %H:%M}.")


if __name__ == "__main__":
    main()
