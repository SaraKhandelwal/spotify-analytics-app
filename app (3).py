"""
Spotify User Behavior Analysis — Phase 2 Dashboard
Team 11 | Business Analytics Course

Research Questions:
  RQ1: How do exploration vs. repeat listening behaviors affect user engagement?
  RQ2: How do device usage patterns influence listening behavior?
  RQ3: Do content preferences (music vs. podcasts) shape engagement patterns?
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from pathlib import Path
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  GLOBAL SPOTIFY PALETTE
# ─────────────────────────────────────────────
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_DARK  = "#191414"
SPOTIFY_GRAY  = "#535353"
SPOTIFY_LIGHT = "#B3B3B3"
SPOTIFY_BRIGHT= "#1ED760"
PALETTE = [SPOTIFY_GREEN, SPOTIFY_DARK, SPOTIFY_GRAY, SPOTIFY_LIGHT, "#FFFFFF", SPOTIFY_BRIGHT]

plt.rcParams.update({
    "figure.dpi":        130,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F8F8",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
    "font.family":       "DejaVu Sans",
    "axes.titleweight":  "bold",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

SESSION_GAP_MIN = 30


# ─────────────────────────────────────────────
#  DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data
def locate_base_dir():
    """
    Find the directory that contains user_1/, user_2/ etc.
    Handles all common repo layouts:
      - Phase 2/user_1/  (original structure)
      - user_1/ sitting directly in the repo root
      - data/user_1/
    """
    def has_user_folders(path):
        """Return True if this directory contains at least one user_N subfolder."""
        try:
            return any(
                os.path.isdir(os.path.join(path, d))
                for d in os.listdir(path)
                if d.startswith("user_")
            )
        except Exception:
            return False

    app_dir = os.path.dirname(os.path.abspath(__file__))
    cwd     = os.getcwd()

    candidates = [
        # Named Phase 2 subfolder
        os.path.join(app_dir, "Phase 2"),
        os.path.join(cwd,     "Phase 2"),
        # Files uploaded directly to repo root (no subfolder)
        app_dir,
        cwd,
        # data/ subfolder variant
        os.path.join(app_dir, "data"),
        os.path.join(cwd,     "data"),
    ]

    for c in candidates:
        if has_user_folders(c):
            return c

    # Last resort: walk the tree looking for any dir with user_N children
    for root, dirs, _ in os.walk(app_dir):
        if has_user_folders(root):
            return root

    return None


def load_json_safe(filepath):
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(filepath, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    return None


@st.cache_data
def load_streaming(base_dir, kind="music"):
    frames = []
    for uf in sorted(os.listdir(base_dir)):
        up = os.path.join(base_dir, uf)
        if not os.path.isdir(up):
            continue
        files = sorted(
            f for f in os.listdir(up)
            if f.startswith(f"StreamingHistory_{kind}") and f.endswith(".json")
            and not f.startswith("._")
        )
        for fname in files:
            data = load_json_safe(os.path.join(up, fname))
            if data:
                df_u = pd.DataFrame(data)
                df_u.insert(0, "user_id", uf)
                frames.append(df_u)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def clean_music_df(df_raw):
    df = df_raw.copy()
    df = df[~(df["trackName"].isnull() & df["artistName"].isnull())]
    df["trackName"]  = df["trackName"].fillna("[Unknown Track]")
    df["artistName"] = df["artistName"].fillna("[Unknown Artist]")
    df["msPlayed"]   = df["msPlayed"].fillna(0)
    df["endTime"]    = pd.to_datetime(df["endTime"], format="%Y-%m-%d %H:%M", errors="coerce")
    df["hour"]       = df["endTime"].dt.hour
    df               = df.dropna(subset=["endTime"])
    df["startTime"]  = df["endTime"] - pd.to_timedelta(df["msPlayed"], unit="ms")
    df               = df[df["msPlayed"] > 0]
    df               = df.drop_duplicates(subset=["user_id", "endTime", "trackName", "artistName"])
    df               = df.sort_values(["user_id", "endTime"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
#  DEVICE CLASSIFICATION  (RQ2)
# ─────────────────────────────────────────────
@st.cache_data
def classify_device_usage(base_dir):
    device_map = {}
    for uf in sorted(os.listdir(base_dir)):
        up = Path(base_dir) / uf
        if not up.is_dir():
            continue
        sq = up / "SearchQueries.json"
        if not sq.exists() or sq.name.startswith("._"):
            device_map[uf] = "unknown"
            continue
        try:
            data = json.loads(sq.read_text(encoding="utf-8"))
        except Exception:
            device_map[uf] = "unknown"
            continue
        platforms = {
            (r.get("platform") or "").strip().upper()
            for r in data if isinstance(r, dict)
        }
        platforms.discard("")
        if not platforms:
            device_map[uf] = "unknown"
            continue
        has_mobile     = any(p.startswith(("IPHONE", "ANDROID", "IPAD")) for p in platforms)
        has_non_mobile = any(not p.startswith(("IPHONE", "ANDROID", "IPAD")) for p in platforms)
        if has_mobile and not has_non_mobile:
            device_map[uf] = "mobile_only"
        elif has_mobile and has_non_mobile:
            device_map[uf] = "multi_device_usage"
        else:
            device_map[uf] = "desktop_only"
    return device_map


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING  (shared sessions)
# ─────────────────────────────────────────────
@st.cache_data
def build_sessions(df):
    df = df.sort_values(["user_id", "endTime"]).copy()
    df["prev_end"] = df.groupby("user_id")["endTime"].shift(1)
    df["gap_min"]  = (df["endTime"] - df["prev_end"]).dt.total_seconds() / 60
    df["new_session"] = (df["prev_end"].isnull() | (df["gap_min"] > SESSION_GAP_MIN)).astype(int)
    df["session_id"]  = df.groupby("user_id")["new_session"].cumsum()
    return df


# ─────────────────────────────────────────────
#  RQ1 FEATURES
# ─────────────────────────────────────────────
@st.cache_data
def build_rq1_features(df):
    df = df.sort_values(["user_id", "artistName", "trackName", "endTime"]).copy()
    df["play_count_so_far"] = df.groupby(["user_id", "artistName", "trackName"]).cumcount()
    df["is_exploration"]    = (df["play_count_so_far"] == 0).astype(int)
    df["is_repeat"]         = (df["play_count_so_far"] > 0).astype(int)

    df = build_sessions(df)

    g_play = df.groupby("user_id").agg(
        total_plays       =("trackName",      "count"),
        exploration_plays =("is_exploration", "sum"),
        repeat_plays      =("is_repeat",      "sum"),
        unique_tracks     =("trackName",       pd.Series.nunique),
        unique_artists    =("artistName",      pd.Series.nunique),
        total_ms          =("msPlayed",        "sum"),
        avg_ms_played     =("msPlayed",        "mean"),
    ).reset_index()

    g_play["exploration_rate"]    = g_play["exploration_plays"] / g_play["total_plays"]
    g_play["repeat_rate"]         = g_play["repeat_plays"]      / g_play["total_plays"]
    g_play["total_listening_min"] = g_play["total_ms"] / 60_000

    session_stats = (
        df.groupby(["user_id", "session_id"])
          .agg(session_plays=("trackName", "count"), session_ms=("msPlayed", "sum"))
          .reset_index()
    )
    g_session = session_stats.groupby("user_id").agg(
        sessions           =("session_id",    "count"),
        avg_session_min    =("session_ms",    "mean"),
        avg_plays_per_sess =("session_plays", "mean"),
    ).reset_index()
    g_session["avg_session_min"] = g_session["avg_session_min"] / 60_000

    features = g_play.merge(g_session, on="user_id")
    features["track_completion_pct"] = (
        features["avg_ms_played"] / 210_000
    ).clip(upper=1.0) * 100

    # Segmentation
    median_exp = features["exploration_rate"].median()
    features["segment"] = np.where(
        features["exploration_rate"] >= median_exp, "Explorer", "Repeater"
    )

    # K-Means clustering
    CLUSTER_FEATURES = ["exploration_rate", "total_listening_min",
                        "avg_plays_per_sess", "track_completion_pct"]
    X = features[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sil_scores = {}
    for k in range(2, min(5, len(features))):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, lbl)
    best_k = max(sil_scores, key=sil_scores.get)

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    features["cluster"] = km_final.fit_predict(X_scaled)

    cluster_means = features.groupby("cluster")["exploration_rate"].mean().sort_values()
    label_map = {
        c: ("🔁 Deep Repeaters" if i == 0 else "🔍 Avid Explorers")
        for i, c in enumerate(cluster_means.index)
    }
    features["cluster_label"] = features["cluster"].map(label_map)

    return features, median_exp


# ─────────────────────────────────────────────
#  RQ2 FEATURES
# ─────────────────────────────────────────────
@st.cache_data
def build_rq2_features(base_dir, df, podcast_df):
    from pathlib import Path
    PHASE2_PATH = Path(base_dir)

    TIME_BUCKETS = {
        "morning":    list(range(6, 12)),
        "afternoon":  list(range(12, 18)),
        "night":      list(range(18, 24)),
        "late_night": list(range(0, 6)),
    }

    device_map = classify_device_usage(base_dir)

    # Build combined streaming (music + podcast) per user
    combined_frames = []
    for uf in sorted(os.listdir(base_dir)):
        up = os.path.join(base_dir, uf)
        if not os.path.isdir(up):
            continue
        for kind, ctype in [("music", "music"), ("podcast", "podcast"), ("audiobook", "audiobook")]:
            files = sorted(
                f for f in os.listdir(up)
                if f.startswith(f"StreamingHistory_{kind}") and f.endswith(".json")
                and not f.startswith("._")
            )
            for fname in files:
                data = load_json_safe(os.path.join(up, fname))
                if data:
                    tmp = pd.DataFrame(data)
                    tmp["user_folder"]  = uf
                    tmp["content_type"] = ctype
                    combined_frames.append(tmp)

    if not combined_frames:
        return pd.DataFrame(), pd.DataFrame()

    all_df = pd.concat(combined_frames, ignore_index=True)
    all_df["msPlayed"] = pd.to_numeric(all_df["msPlayed"], errors="coerce").fillna(0)
    all_df = all_df[all_df["msPlayed"] >= 0].copy()

    summary_rows = []
    for uf in sorted(all_df["user_folder"].unique()):
        evt_df        = all_df[all_df["user_folder"] == uf].copy()
        device_usage  = device_map.get(uf, "unknown")
        total_ms      = evt_df["msPlayed"].sum()
        total_hours   = total_ms / 3_600_000 if total_ms > 0 else 0

        content_ms  = evt_df.groupby("content_type", as_index=False)["msPlayed"].sum()
        content_pct = {
            r["content_type"]: (r["msPlayed"] / total_ms * 100 if total_ms > 0 else 0)
            for _, r in content_ms.iterrows()
        }

        cell_music_df    = evt_df[evt_df["content_type"] == "music"]
        unique_songs     = cell_music_df["trackName"].dropna().nunique()
        unique_artists   = cell_music_df["artistName"].dropna().nunique()
        songs_per_hour   = unique_songs   / total_hours if total_hours > 0 else 0
        artists_per_hour = unique_artists / total_hours if total_hours > 0 else 0

        dt = pd.to_datetime(evt_df["endTime"], errors="coerce")
        valid_hours = dt.dt.hour.dropna().astype(int)
        if len(valid_hours) > 0:
            morning_pct    = valid_hours.isin(TIME_BUCKETS["morning"]).mean()    * 100
            afternoon_pct  = valid_hours.isin(TIME_BUCKETS["afternoon"]).mean()  * 100
            night_pct      = valid_hours.isin(TIME_BUCKETS["night"]).mean()      * 100
            late_night_pct = valid_hours.isin(TIME_BUCKETS["late_night"]).mean() * 100
        else:
            morning_pct = afternoon_pct = night_pct = late_night_pct = 0.0

        summary_rows.append({
            "user_folder": uf, "device_usage": device_usage,
            "music_pct": content_pct.get("music", 0),
            "podcast_pct": content_pct.get("podcast", 0),
            "audiobook_pct": content_pct.get("audiobook", 0),
            "songs_per_hour": round(songs_per_hour, 2),
            "artists_per_hour": round(artists_per_hour, 2),
            "morning_pct_6_11":    round(morning_pct, 2),
            "afternoon_pct_12_17": round(afternoon_pct, 2),
            "night_pct_18_23":     round(night_pct, 2),
            "late_night_pct_0_5":  round(late_night_pct, 2),
        })

    rq2_metrics = pd.DataFrame(summary_rows).sort_values("user_folder").reset_index(drop=True)

    # Sessions
    session_rows = []
    for uf in sorted(os.listdir(base_dir)):
        up = Path(base_dir) / uf
        if not up.is_dir():
            continue
        device_usage = device_map.get(uf, "unknown")
        events = []
        for f in sorted(up.glob("StreamingHistory*.json")):
            if f.name.startswith("._"):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            for row in data:
                end_time  = pd.to_datetime(row.get("endTime"), errors="coerce")
                ms_played = pd.to_numeric(row.get("msPlayed", 0), errors="coerce")
                if pd.isna(end_time) or pd.isna(ms_played) or ms_played < 0:
                    continue
                events.append({"start": end_time - pd.to_timedelta(ms_played, unit="ms"), "end": end_time})

        if not events:
            session_rows.append({"user_folder": uf, "device_usage": device_usage,
                                  "session_count": 0, "avg_session_length_min": 0.0})
            continue

        events_df = pd.DataFrame(events).sort_values("start").reset_index(drop=True)
        gap    = events_df["start"] - events_df["end"].shift(1)
        is_new = gap.isna() | (gap > pd.Timedelta(minutes=SESSION_GAP_MIN))
        events_df["session_id"] = is_new.cumsum()

        sessions = events_df.groupby("session_id", as_index=False).agg(
            session_start=("start", "min"), session_end=("end", "max"))
        sessions["session_length_min"] = (
            sessions["session_end"] - sessions["session_start"]
        ).dt.total_seconds() / 60

        session_rows.append({
            "user_folder": uf, "device_usage": device_usage,
            "session_count": int(len(sessions)),
            "avg_session_length_min": float(sessions["session_length_min"].mean()),
        })

    rq2_sessions = pd.DataFrame(session_rows).sort_values("user_folder").reset_index(drop=True)
    rq2_sessions["avg_session_length_min"] = rq2_sessions["avg_session_length_min"].round(2)

    return rq2_metrics, rq2_sessions


# ─────────────────────────────────────────────
#  RQ3 FEATURES
# ─────────────────────────────────────────────
@st.cache_data
def build_rq3_features(df, podcast_df):
    rq3_total_music_min   = df.groupby("user_id")["msPlayed"].sum().div(60000).rename("total_music_min")
    rq3_total_podcast_min = podcast_df.groupby("user_id")["msPlayed"].sum().div(60000).rename("total_podcast_min")
    rq3_skip_rate         = (
        df.groupby("user_id")
          .apply(lambda g: (g["msPlayed"] < 30000).mean(), include_groups=False)
          .rename("skip_rate")
    )
    rq3_avg_min = df.groupby("user_id")["msPlayed"].mean().div(60000).rename("avg_min_per_stream")

    features = (
        pd.concat([rq3_total_music_min, rq3_total_podcast_min, rq3_skip_rate, rq3_avg_min], axis=1)
        .fillna(0)
        .reset_index()
    )
    rq3_total = features["total_music_min"] + features["total_podcast_min"]
    features["pct_podcast"]   = (features["total_podcast_min"] / rq3_total) * 100
    features["listener_type"] = features["pct_podcast"].apply(
        lambda x: "Podcast-Leaning" if x > 20 else "Music-Leaning"
    )
    return features


# ─────────────────────────────────────────────
#  ADVANCED CLUSTER SEGMENTATION  (new layer)
# ─────────────────────────────────────────────
@st.cache_data
def build_advanced_clusters(rq1_features, rq3_features, rq2_metrics, rq2_sessions):
    """
    Combine features from all three RQs into a unified user profile,
    then apply K-Means to produce business-labeled segments.
    """
    merged = rq1_features[
        ["user_id", "exploration_rate", "repeat_rate", "total_listening_min",
         "avg_session_min", "avg_plays_per_sess", "track_completion_pct", "sessions"]
    ].copy()

    rq3_sub = rq3_features[["user_id", "pct_podcast", "skip_rate", "avg_min_per_stream"]].copy()
    merged  = merged.merge(rq3_sub, on="user_id", how="left")

    # Map device label
    dev_map = {"mobile_only": 1, "multi_device_usage": 2, "desktop_only": 2, "unknown": 1}
    rq2_dev = rq2_metrics[["user_folder", "device_usage"]].rename(columns={"user_folder": "user_id"})
    rq2_dev["device_score"] = rq2_dev["device_usage"].map(dev_map)
    merged = merged.merge(rq2_dev[["user_id", "device_score"]], on="user_id", how="left").fillna(1)

    CLUSTER_FEATURES = [
        "exploration_rate", "total_listening_min", "avg_plays_per_sess",
        "track_completion_pct", "pct_podcast", "skip_rate"
    ]
    X = merged[CLUSTER_FEATURES].fillna(0)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sil_scores = {}
    for k in range(2, min(5, len(merged))):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, lbl)
    best_k = max(sil_scores, key=sil_scores.get)

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    merged["cluster"] = km_final.fit_predict(X_scaled)

    # Assign business labels based on cluster centroids
    centers = pd.DataFrame(
        scaler.inverse_transform(km_final.cluster_centers_), columns=CLUSTER_FEATURES
    )
    centers["cluster"] = range(best_k)

    # Rank clusters by exploration_rate and total listening to name them
    centers_sorted = centers.sort_values("exploration_rate")
    label_map = {}
    for i, (_, row) in enumerate(centers_sorted.iterrows()):
        c = int(row["cluster"])
        if row["exploration_rate"] < centers_sorted["exploration_rate"].median():
            if row["total_listening_min"] > centers_sorted["total_listening_min"].median():
                label_map[c] = ("🎵 Core Listeners", "High volume, familiar tracks — your most loyal users")
            else:
                label_map[c] = ("🔁 Comfort Zone Users", "Low exploration, casual listening — retention risk")
        else:
            if row["skip_rate"] < centers_sorted["skip_rate"].median():
                label_map[c] = ("🌟 Avid Explorers", "High discovery, high completion — Spotify's growth engine")
            else:
                label_map[c] = ("⚡ Skimming Discoverers", "Exploring broadly but skipping often — need better recs")

    merged["cluster_label"] = merged["cluster"].map(lambda c: label_map.get(c, (f"Cluster {c}", ""))[0])
    merged["cluster_desc"]  = merged["cluster"].map(lambda c: label_map.get(c, ("", "Unknown"))[1])

    return merged, centers, sil_scores, CLUSTER_FEATURES


# ─────────────────────────────────────────────
#  STREAMLIT APP LAYOUT
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Spotify User Behavior Analytics — Team 11",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #191414; }
        [data-testid="stSidebar"] * { color: #FFFFFF !important; }
        .metric-card {
            background: linear-gradient(135deg, #1DB954 0%, #158a3e 100%);
            border-radius: 12px; padding: 20px; color: white;
            text-align: center; margin: 6px 0;
        }
        .metric-card h2 { font-size: 2.2rem; margin: 0; }
        .metric-card p  { margin: 4px 0; font-size: 0.95rem; opacity: 0.9; }
        .insight-box {
            background: #f0faf4; border-left: 4px solid #1DB954;
            padding: 14px 18px; border-radius: 6px; margin: 12px 0;
        }
        .warning-box {
            background: #fff8f0; border-left: 4px solid #E88B00;
            padding: 14px 18px; border-radius: 6px; margin: 12px 0;
        }
        .cluster-card {
            background: #191414; color: white;
            border-radius: 10px; padding: 16px; margin: 8px 0;
        }
        h1, h2, h3 { color: #191414; }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            background: #f4f4f4; border-radius: 8px 8px 0 0; padding: 8px 20px;
        }
        .stTabs [aria-selected="true"] {
            background: #1DB954 !important; color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown("# 🎵 Spotify Analytics")
        st.markdown("**Team 11 | Business Analytics**")
        st.markdown("---")
        page = st.radio("Navigate to", [
            "📋 Overview",
            "📊 Data Summary",
            "🔍 RQ1 · Exploration vs Repeat",
            "📱 RQ2 · Device Usage Patterns",
            "🎙️ RQ3 · Content Preferences",
            "🧩 Cluster Segmentation",
            "💡 Decision-Maker Insights",
        ])
        st.markdown("---")
        st.markdown("**Data:** Spotify Personal Export  \n**Users:** 6 active users  \n**Period:** 2024–2025")

    # ── LOAD DATA ────────────────────────────
    base_dir = locate_base_dir()
    if base_dir is None:
        st.error("⚠️ **Data directory not found.** Please place the `Phase 2/` folder next to `app.py`.")
        st.stop()

    with st.spinner("Loading Spotify data…"):
        df_raw     = load_streaming(base_dir, "music")
        podcast_df = load_streaming(base_dir, "podcast")
        df         = clean_music_df(df_raw)

    # ══════════════════════════════════════════
    #  PAGE: OVERVIEW
    # ══════════════════════════════════════════
    if page == "📋 Overview":
        st.title("🎵 Spotify User Behavior Analysis")
        st.markdown("### Team 11 · Phase 2 · Business Analytics Course")
        st.markdown("---")

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("## Problem Statement")
            st.markdown("""
            Spotify generates massive volumes of implicit user data — every skip, replay, device
            switch, and late-night session tells a story. But raw streaming history alone doesn't
            reveal **why** users behave the way they do.

            This analysis transforms the personal Spotify data exports of **7 real users**
            into actionable behavioral profiles. By combining three focused research questions
            with an unsupervised clustering layer, we surface patterns that can directly inform
            Spotify's recommendation engine, retention strategy, and product roadmap.
            """)

            st.markdown("## Research Questions")
            st.markdown("""
            | # | Research Question | Business Goal |
            |---|-------------------|---------------|
            | RQ1 | How do exploration vs. repeat behaviors affect engagement? | Optimize recommendation engine |
            | RQ2 | How does device usage (mobile vs. multi-device) influence listening? | Personalize UX by context |
            | RQ3 | Do content preferences (music vs. podcasts) shape engagement? | Cross-format recommendations |
            """)

            st.markdown("## Analytical Approach")
            st.markdown("""
            1. **Data Loading & Cleaning** — Consolidated 7 user exports; removed zero-plays, duplicates, and nulls
            2. **Feature Engineering** — Sessionization (30-min gap rule), play-type labeling, device classification
            3. **Segmentation** — Threshold-based (Explorers/Repeaters, Mobile/Multi-Device, Music/Podcast)
            4. **K-Means Clustering** — Unsupervised behavioral segmentation with silhouette-optimized k
            5. **Business Insights** — Actionable recommendations for each user archetype
            """)

        with col2:
            n_users  = df["user_id"].nunique()
            n_plays  = len(df)
            n_hours  = round(df["msPlayed"].sum() / 3_600_000, 0)
            n_tracks = df["trackName"].nunique()
            date_rng = f"{df['endTime'].min().strftime('%b %Y')} – {df['endTime'].max().strftime('%b %Y')}"

            for label, val, icon in [
                ("Total Music Plays", f"{n_plays:,}", "🎶"),
                ("Active Users",      f"{n_users}",    "👥"),
                ("Total Listening Hours", f"{n_hours:,.0f}", "⏱️"),
                ("Unique Tracks", f"{n_tracks:,}", "🎵"),
            ]:
                st.markdown(f"""
                <div class="metric-card">
                    <p>{icon} {label}</p>
                    <h2>{val}</h2>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"**Date range:** {date_rng}")

    # ══════════════════════════════════════════
    #  PAGE: DATA SUMMARY
    # ══════════════════════════════════════════
    elif page == "📊 Data Summary":
        st.title("📊 Data Summary")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Per-User Music Plays")
            plays_by_user = df.groupby("user_id").size().reset_index(name="plays")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(plays_by_user["user_id"], plays_by_user["plays"],
                   color=SPOTIFY_GREEN, edgecolor="white", linewidth=1.2)
            for bar, val in zip(ax.patches, plays_by_user["plays"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f"{val:,}", ha="center", fontsize=9, fontweight="bold")
            ax.set_title("Total Music Plays per User")
            ax.set_xlabel("User"); ax.set_ylabel("Plays")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("### Listening Hours per User")
            hours_by_user = df.groupby("user_id")["msPlayed"].sum().div(3_600_000).reset_index()
            hours_by_user.columns = ["user_id", "hours"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(hours_by_user["user_id"], hours_by_user["hours"],
                    color=SPOTIFY_DARK, edgecolor="white", linewidth=1.2)
            for bar, val in zip(ax.patches, hours_by_user["hours"]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}h", va="center", fontsize=9, fontweight="bold")
            ax.set_title("Total Listening Hours per User")
            ax.set_xlabel("Hours"); ax.invert_yaxis()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("### Hourly Listening Distribution (All Users)")
        hourly = df.groupby("hour").size().reset_index(name="plays")
        fig, ax = plt.subplots(figsize=(12, 3.5))
        colors = [SPOTIFY_GREEN if h in range(18, 24) else SPOTIFY_GRAY for h in hourly["hour"]]
        ax.bar(hourly["hour"], hourly["plays"], color=colors, edgecolor="white")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45, fontsize=8)
        ax.set_title("Aggregate Plays by Hour of Day (green = evening peak)")
        ax.set_xlabel("Hour"); ax.set_ylabel("Total Plays")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("### Raw Data Preview")
        st.dataframe(
            df[["user_id", "endTime", "artistName", "trackName", "msPlayed", "hour"]]
            .head(50),
            use_container_width=True
        )

    # ══════════════════════════════════════════
    #  PAGE: RQ1 — EXPLORATION VS REPEAT
    # ══════════════════════════════════════════
    elif page == "🔍 RQ1 · Exploration vs Repeat":
        st.title("🔍 RQ1 · Exploration vs. Repeat Listening")
        st.markdown("""
        **Research Question:** How do exploration behaviors (discovering new songs) vs. repeat behaviors
        (replaying familiar songs) affect user engagement?

        **Business Goal:** Help Spotify optimize its recommendation engine by profiling users along
        the exploration–familiarity spectrum.
        """)
        st.markdown("---")

        rq1_features, median_exp = build_rq1_features(df)

        # KPI summary row
        exp_users = rq1_features[rq1_features["segment"] == "Explorer"]
        rep_users = rq1_features[rq1_features["segment"] == "Repeater"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Explorers", len(exp_users), f"avg {exp_users['exploration_rate'].mean()*100:.1f}% new tracks")
        c2.metric("Repeaters", len(rep_users), f"avg {rep_users['repeat_rate'].mean()*100:.1f}% repeat rate")
        c3.metric("Median Exploration Rate", f"{median_exp*100:.1f}%")
        c4.metric("Overall Unique Tracks", f"{rq1_features['unique_tracks'].sum():,}")

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["Exploration Rate", "Engagement Comparison", "Session Depth", "Correlation"])

        with tab1:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(
                rq1_features["user_id"],
                rq1_features["exploration_rate"] * 100,
                color=[SPOTIFY_GREEN if s == "Explorer" else SPOTIFY_GRAY
                       for s in rq1_features["segment"]],
                edgecolor="white", linewidth=1.2, zorder=3,
            )
            ax.axhline(median_exp * 100, color="#E22134", linestyle="--", linewidth=1.8,
                       label=f"Median ({median_exp*100:.1f}%)", zorder=4)
            for bar, val in zip(ax.patches, rq1_features["exploration_rate"] * 100):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_title("Exploration Rate by User\n(Green = Explorer, Gray = Repeater)", pad=12)
            ax.set_xlabel("User ID"); ax.set_ylabel("Exploration Rate (%)")
            ax.set_ylim(0, 110); ax.legend(frameon=False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("""
            <div class="insight-box">
            <b>Key Insight:</b> Most users are above the median exploration threshold, indicating
            the cohort skews toward discovery-oriented listening. Explorers open more sessions
            and accumulate more total listening time overall.
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            metrics = {
                "Total Listening\nTime (hrs)": rq1_features.groupby("segment")["total_listening_min"].mean() / 60,
                "Avg Session\nLength (min)":   rq1_features.groupby("segment")["avg_session_min"].mean(),
                "Plays per\nSession":          rq1_features.groupby("segment")["avg_plays_per_sess"].mean(),
                "Track\nCompletion (%)":       rq1_features.groupby("segment")["track_completion_pct"].mean(),
            }
            fig, axes = plt.subplots(1, 4, figsize=(14, 5))
            fig.suptitle("Explorer vs. Repeater — Engagement Metric Comparison",
                         fontsize=13, fontweight="bold", y=1.02)
            seg_colors = {"Explorer": SPOTIFY_GREEN, "Repeater": SPOTIFY_GRAY}
            for ax, (metric_label, values) in zip(axes, metrics.items()):
                segs  = values.index.tolist()
                bars  = ax.bar(segs, values.values,
                               color=[seg_colors[s] for s in segs],
                               edgecolor="white", linewidth=1.2, width=0.5, zorder=3)
                for bar, val in zip(bars, values.values):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + values.max() * 0.02,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
                ax.set_title(metric_label, fontsize=11)
                ax.set_ylim(0, values.max() * 1.35)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("""
                <div class="insight-box">
                <b>Explorers</b> drive higher total listening time, suggesting that discovery
                behaviour is positively correlated with platform engagement — not just noise.
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown("""
                <div class="insight-box">
                <b>Repeaters</b> show higher track completion rates — when users know a song,
                they listen all the way through, providing stronger positive signal for recommendations.
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            fig, ax = plt.subplots(figsize=(9, 5))
            uf_sorted   = rq1_features.sort_values("avg_plays_per_sess", ascending=False)
            bar_colors  = [SPOTIFY_GREEN if s == "Explorer" else SPOTIFY_GRAY
                           for s in uf_sorted["segment"]]
            bars = ax.barh(uf_sorted["user_id"], uf_sorted["avg_plays_per_sess"],
                           color=bar_colors, edgecolor="white", linewidth=1.2,
                           height=0.55, zorder=3)
            for bar, val in zip(bars, uf_sorted["avg_plays_per_sess"]):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}", va="center", ha="left", fontsize=9, fontweight="bold")
            mean_val = rq1_features["avg_plays_per_sess"].mean()
            ax.axvline(mean_val, color="#E22134", linestyle="--", linewidth=1.5,
                       label=f"Overall mean ({mean_val:.1f})")
            legend_elements = [
                Patch(facecolor=SPOTIFY_GREEN, edgecolor="white", label="Explorer"),
                Patch(facecolor=SPOTIFY_GRAY,  edgecolor="white", label="Repeater"),
                plt.Line2D([0], [0], color="#E22134", linestyle="--", linewidth=1.5, label="Overall mean"),
            ]
            ax.legend(handles=legend_elements, frameon=False)
            ax.set_title("Average Plays per Session (Session Depth)", pad=12)
            ax.set_xlabel("Avg Plays per Session"); ax.invert_yaxis()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with tab4:
            corr_cols   = ["exploration_rate", "repeat_rate", "total_listening_min",
                           "avg_session_min", "avg_plays_per_sess", "track_completion_pct",
                           "unique_tracks", "sessions"]
            corr_labels = ["Exploration Rate", "Repeat Rate", "Total Listening (min)",
                           "Avg Session (min)", "Plays/Session", "Track Completion %",
                           "Unique Tracks", "Sessions"]
            corr_matrix = rq1_features[corr_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                        cmap=sns.diverging_palette(10, 145, s=85, l=40, as_cmap=True),
                        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                        linecolor="white", cbar_kws={"shrink": 0.75}, ax=ax,
                        xticklabels=corr_labels, yticklabels=corr_labels)
            ax.set_title("Feature Correlation Matrix (Lower triangle — Pearson r)", pad=14)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("### Segment Feature Summary")
        seg_summary = rq1_features.groupby("segment")[
            ["exploration_rate", "repeat_rate", "total_listening_min",
             "avg_session_min", "avg_plays_per_sess", "track_completion_pct"]
        ].mean().round(3)
        st.dataframe(seg_summary, use_container_width=True)

    # ══════════════════════════════════════════
    #  PAGE: RQ2 — DEVICE USAGE
    # ══════════════════════════════════════════
    elif page == "📱 RQ2 · Device Usage Patterns":
        st.title("📱 RQ2 · Device Usage Patterns")
        st.markdown("""
        **Research Question:** How does device usage (mobile-only vs. multi-device) influence
        listening patterns, content variety, and session behavior?

        **Classification:** Platform identifiers from `SearchQueries.json` — iPhone/Android = Mobile;
        any non-mobile platform present = Multi-Device.
        """)
        st.markdown("---")

        with st.spinner("Building device features…"):
            rq2_metrics, rq2_sessions = build_rq2_features(base_dir, df, podcast_df)

        # Device distribution
        device_counts = rq2_metrics["device_usage"].value_counts()
        device_labels = {"mobile_only": "Mobile Only", "multi_device_usage": "Multi-Device",
                         "desktop_only": "Desktop Only", "unknown": "Unknown"}

        col1, col2, col3 = st.columns(3)
        mobile_n = device_counts.get("mobile_only", 0)
        multi_n  = device_counts.get("multi_device_usage", 0)
        col1.metric("📱 Mobile-Only Users", mobile_n)
        col2.metric("💻 Multi-Device Users", multi_n)
        col3.metric("Total Users Classified", mobile_n + multi_n)

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["Session Behavior", "Content Mix & Discovery", "Time-of-Day Patterns"])

        with tab1:
            chart_df = rq2_sessions[
                rq2_sessions["device_usage"].isin(["mobile_only", "multi_device_usage"])
            ].copy()
            chart_df["device_type"] = chart_df["device_usage"].map(
                {"mobile_only": "Mobile", "multi_device_usage": "Multi-Device"}
            )
            chart_summary = chart_df.groupby("device_type", as_index=False).agg(
                avg_session_length_min=("avg_session_length_min", "mean"),
                session_count=("session_count", "mean")
            )

            x     = range(len(chart_summary))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 5))
            bar1 = ax.bar([i - width/2 for i in x], chart_summary["avg_session_length_min"],
                          width=width, label="Avg Session Length (min)", color=SPOTIFY_GREEN)
            bar2 = ax.bar([i + width/2 for i in x], chart_summary["session_count"],
                          width=width, label="Session Count", color=SPOTIFY_DARK)
            ax.set_title("Session Behavior Comparison by Device Type")
            ax.set_xlabel("Device Type"); ax.set_ylabel("Value")
            ax.set_xticks(list(x)); ax.set_xticklabels(chart_summary["device_type"])
            ax.legend(); ax.bar_label(bar1, fmt="%.1f", padding=3); ax.bar_label(bar2, fmt="%.0f", padding=3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("""
            <div class="insight-box">
            <b>Finding:</b> Multi-device users tend to have more sessions, suggesting they switch
            contexts frequently (commute → desk → home), while mobile-only users concentrate
            listening into fewer, potentially shorter sessions.
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            chart_df2 = rq2_metrics[
                rq2_metrics["device_usage"].isin(["mobile_only", "multi_device_usage"])
            ].copy()
            chart_df2["device_type"] = chart_df2["device_usage"].map(
                {"mobile_only": "Mobile", "multi_device_usage": "Multi-Device"}
            )
            chart_summary2 = chart_df2.groupby("device_type", as_index=False).agg(
                music_pct=("music_pct", "mean"),
                podcast_pct=("podcast_pct", "mean"),
                songs_per_hour=("songs_per_hour", "mean")
            )

            x     = range(len(chart_summary2))
            width = 0.55
            fig, ax1 = plt.subplots(figsize=(9, 5))
            bar_music   = ax1.bar(x, chart_summary2["music_pct"], width=width,
                                  label="Music %", color=SPOTIFY_GREEN)
            bar_podcast = ax1.bar(x, chart_summary2["podcast_pct"], width=width,
                                  bottom=chart_summary2["music_pct"],
                                  label="Podcast %", color=SPOTIFY_GRAY)
            ax1.set_title("Music, Podcast Share and Songs/Hour by Device Type")
            ax1.set_xlabel("Device Type"); ax1.set_ylabel("Content Share (%)")
            ax1.set_xticks(list(x)); ax1.set_xticklabels(chart_summary2["device_type"])
            ax2 = ax1.twinx()
            ax2.plot(x, chart_summary2["songs_per_hour"], color=SPOTIFY_BRIGHT,
                     marker="o", linewidth=2, label="Songs per Hour")
            ax2.set_ylabel("Songs per Hour")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper right")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with tab3:
            time_cols = ["morning_pct_6_11", "afternoon_pct_12_17",
                         "night_pct_18_23", "late_night_pct_0_5"]
            time_labels = {
                "morning_pct_6_11":    "Morning (6-11)",
                "afternoon_pct_12_17": "Afternoon (12-17)",
                "night_pct_18_23":     "Night (18-23)",
                "late_night_pct_0_5":  "Late Night (0-5)",
            }
            plot_df = rq2_metrics[
                rq2_metrics["user_folder"] != "user_7"
            ][["user_folder"] + time_cols].copy().sort_values("user_folder")

            fig, ax = plt.subplots(figsize=(11, 6))
            bottom = [0.0] * len(plot_df)
            colors = [SPOTIFY_GREEN, SPOTIFY_GRAY, SPOTIFY_BRIGHT, SPOTIFY_LIGHT]
            for col, color in zip(time_cols, colors):
                bars = ax.bar(plot_df["user_folder"], plot_df[col], bottom=bottom,
                              label=time_labels[col], color=color)
                for bar, val, base in zip(bars, plot_df[col], bottom):
                    if val >= 5:
                        ax.text(bar.get_x() + bar.get_width()/2, base + val/2,
                                f"{val:.1f}%", ha="center", va="center",
                                color="white", fontsize=9, fontweight="bold")
                bottom = [b + v for b, v in zip(bottom, plot_df[col])]
            ax.set_title("Per-User Listening Time-of-Day Breakdown")
            ax.set_xlabel("User"); ax.set_ylabel("Listening Share (%)")
            ax.set_ylim(0, 100)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
            plt.tight_layout(rect=[0, 0, 0.85, 1]); st.pyplot(fig); plt.close()

            st.markdown("""
            <div class="insight-box">
            <b>Finding:</b> Time-of-day patterns are remarkably consistent across device types —
            both groups peak in evening hours (18–23). This suggests device type influences
            <i>how</i> people listen, but not <i>when</i>.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Device Classification Table")
        display_df = rq2_metrics.copy()
        display_df["device_usage"] = display_df["device_usage"].map(
            lambda x: device_labels.get(x, x)
        )
        st.dataframe(display_df, use_container_width=True)

    # ══════════════════════════════════════════
    #  PAGE: RQ3 — CONTENT PREFERENCES
    # ══════════════════════════════════════════
    elif page == "🎙️ RQ3 · Content Preferences":
        st.title("🎙️ RQ3 · Content Preferences vs. Engagement")
        st.markdown("""
        **Research Question:** Do users with different content preferences (music vs. podcasts)
        show different engagement patterns?

        **Business Goal:** Understand how content type shapes skip behavior, listening depth,
        and temporal patterns to improve cross-format recommendations.
        """)
        st.markdown("---")

        rq3_features = build_rq3_features(df, podcast_df)

        music_users   = rq3_features[rq3_features["listener_type"] == "Music-Leaning"]
        podcast_users = rq3_features[rq3_features["listener_type"] == "Podcast-Leaning"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Music-Leaning Users", len(music_users))
        c2.metric("Podcast-Leaning Users", len(podcast_users))
        c3.metric("Avg Skip Rate (Music)", f"{music_users['skip_rate'].mean()*100:.1f}%")
        c4.metric("Avg Skip Rate (Podcast)", f"{podcast_users['skip_rate'].mean()*100:.1f}%")

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["Skip Rate & Depth", "Temporal Patterns", "Full Feature Table"])

        with tab1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            x = rq3_features["pct_podcast"].values

            # Left: skip rate
            y1 = rq3_features["skip_rate"].values
            ax1.scatter(x, y1, color=SPOTIFY_GREEN, s=100, zorder=3)
            for _, row in rq3_features.iterrows():
                ax1.annotate(row["user_id"], (row["pct_podcast"], row["skip_rate"]),
                             textcoords="offset points", xytext=(6, 4),
                             fontsize=9, color=SPOTIFY_GRAY)
            m1, b1 = np.polyfit(x, y1, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax1.plot(x_line, m1 * x_line + b1, color=SPOTIFY_GREEN,
                     linewidth=1.5, linestyle="--", alpha=0.7)
            ax1.set_ylabel("Skip Rate"); ax1.set_title("Podcast % vs Skip Rate", pad=10)
            ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax1.grid(True)

            # Right: avg min per stream
            y2 = rq3_features["avg_min_per_stream"].values
            ax2.scatter(x, y2, color=SPOTIFY_GREEN, s=100, zorder=3)
            for _, row in rq3_features.iterrows():
                ax2.annotate(row["user_id"], (row["pct_podcast"], row["avg_min_per_stream"]),
                             textcoords="offset points", xytext=(6, 4),
                             fontsize=9, color=SPOTIFY_GRAY)
            m2, b2 = np.polyfit(x, y2, 1)
            ax2.plot(x_line, m2 * x_line + b2, color=SPOTIFY_GREEN,
                     linewidth=1.5, linestyle="--", alpha=0.7)
            ax2.set_ylabel("Avg Minutes per Stream")
            ax2.set_title("Podcast % vs Avg Min per Stream", pad=10); ax2.grid(True)

            fig.supxlabel("Podcast Listening Share (%)", y=0.02)
            fig.suptitle("Content Preference vs Engagement Metrics", fontsize=13, y=1.02)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("""
                <div class="insight-box">
                <b>Negative trend (skip rate):</b> As podcast share increases, skip rate falls.
                Podcast listeners are more intentional — they commit to content rather than skimming.
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown("""
                <div class="insight-box">
                <b>Positive trend (depth):</b> Higher podcast share correlates with longer
                average stream duration. Podcast episodes naturally extend engagement time.
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            df_temporal = df.merge(rq3_features[["user_id", "pct_podcast"]], on="user_id", how="left")
            df_temporal["listening_group"] = df_temporal["pct_podcast"].apply(
                lambda x: "Podcast-Leaning" if x > 20 else "Music-Leaning"
            )
            temporal = (
                df_temporal.groupby(["listening_group", "hour"])
                .size()
                .reset_index(name="avg_streams_per_hour")
            )
            user_counts = {"Music-Leaning": len(music_users), "Podcast-Leaning": max(len(podcast_users), 1)}
            temporal["avg_streams_per_hour"] = temporal.apply(
                lambda row: row["avg_streams_per_hour"] / user_counts[row["listening_group"]], axis=1
            )
            fig, ax = plt.subplots(figsize=(12, 5))
            for group, color in [("Music-Leaning", SPOTIFY_GRAY), ("Podcast-Leaning", SPOTIFY_GREEN)]:
                subset = temporal[temporal["listening_group"] == group]
                ax.plot(subset["hour"], subset["avg_streams_per_hour"],
                        color=color, linewidth=2.5, marker="o", markersize=4, label=group)
            ax.axvline(x=18, color="gray", linestyle="--", alpha=0.4, linewidth=1)
            ax.axvline(x=20, color="gray", linestyle="--", alpha=0.4, linewidth=1)
            hour_labels = {0:"12am",3:"3am",6:"6am",9:"9am",12:"12pm",15:"3pm",18:"6pm",21:"9pm",23:"11pm"}
            ax.set_xticks(list(hour_labels.keys()))
            ax.set_xticklabels(list(hour_labels.values()))
            ax.set_title("When Do Music vs Podcast Users Listen?", fontweight="bold", fontsize=13)
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Avg Streams per User")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("""
            <div class="insight-box">
            <b>Finding:</b> Both groups peak in evening hours (6–9 PM). Music-leaning users
            have sharp late-night activity (until ~3 AM), while podcast listeners show a
            flatter, more consistent curve throughout the day.
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.dataframe(
                rq3_features.round(3).sort_values("pct_podcast", ascending=False),
                use_container_width=True
            )
            r_skip  = rq3_features["pct_podcast"].corr(rq3_features["skip_rate"])
            r_depth = rq3_features["pct_podcast"].corr(rq3_features["avg_min_per_stream"])
            st.markdown(f"**Correlation — Podcast % vs Skip Rate:** `{r_skip:.3f}`  \n"
                        f"**Correlation — Podcast % vs Avg Min/Stream:** `{r_depth:.3f}`")

    # ══════════════════════════════════════════
    #  PAGE: CLUSTER SEGMENTATION
    # ══════════════════════════════════════════
    elif page == "🧩 Cluster Segmentation":
        st.title("🧩 Unified Cluster Segmentation")
        st.markdown("""
        An unsupervised K-Means clustering layer synthesizes features from all three research
        questions into a unified behavioral profile for each user. This reveals natural groupings
        beyond the binary segments used in individual RQs.

        **Features used:** Exploration rate, total listening time, plays per session,
        track completion %, podcast share, skip rate.
        """)
        st.markdown("---")

        with st.spinner("Building unified clusters…"):
            rq1_features, _ = build_rq1_features(df)
            rq3_features     = build_rq3_features(df, podcast_df)
            rq2_metrics, _   = build_rq2_features(base_dir, df, podcast_df)
            _, rq2_sessions  = build_rq2_features(base_dir, df, podcast_df)
            merged, centers, sil_scores, cluster_features = build_advanced_clusters(
                rq1_features, rq3_features, rq2_metrics, rq2_sessions
            )

        # Silhouette scores
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("### Optimal k (Silhouette)")
            best_k = max(sil_scores, key=sil_scores.get)
            for k, score in sil_scores.items():
                marker = " ← best" if k == best_k else ""
                st.markdown(f"**k = {k}:** {score:.3f}{marker}")
            st.markdown(f"""
            <div class="insight-box">
            Silhouette score of <b>{sil_scores[best_k]:.3f}</b> at k = {best_k} indicates
            well-separated clusters given the small sample size.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Cluster Assignments")
            for _, row in merged.iterrows():
                st.markdown(f"""
                <div class="cluster-card">
                <b>{row['user_id']}</b> → <span style="color:{SPOTIFY_GREEN}">{row['cluster_label']}</span><br>
                <small>{row['cluster_desc']}</small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Cluster Feature Profiles")
        cluster_profile = merged.groupby("cluster_label")[cluster_features].mean().round(3)
        st.dataframe(cluster_profile, use_container_width=True)

        # Radar / spider chart
        st.markdown("### Cluster Comparison — Key Metrics")
        fig, ax = plt.subplots(figsize=(12, 5))
        cluster_labels_uniq = merged["cluster_label"].unique()
        x_pos = np.arange(len(cluster_features))
        width = 0.8 / max(len(cluster_labels_uniq), 1)
        colors_avail = [SPOTIFY_GREEN, SPOTIFY_DARK, SPOTIFY_GRAY, SPOTIFY_BRIGHT]
        for i, label in enumerate(cluster_labels_uniq):
            sub  = cluster_profile.loc[label] if label in cluster_profile.index else None
            if sub is None:
                continue
            norm = (sub - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min() + 1e-9)
            ax.bar(x_pos + i * width, norm.values,
                   width=width * 0.9, label=label,
                   color=colors_avail[i % len(colors_avail)], edgecolor="white", alpha=0.85)
        ax.set_xticks(x_pos + width * (len(cluster_labels_uniq) - 1) / 2)
        ax.set_xticklabels(cluster_features, rotation=30, ha="right")
        ax.set_ylabel("Normalized Value (0–1)")
        ax.set_title("Cluster Profiles — Normalized Feature Comparison")
        ax.legend(loc="upper right", frameon=False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("### Full User Cluster Table")
        st.dataframe(
            merged[["user_id", "cluster_label", "cluster_desc",
                    "exploration_rate", "total_listening_min",
                    "skip_rate", "pct_podcast"]].round(3),
            use_container_width=True
        )

    # ══════════════════════════════════════════
    #  PAGE: DECISION-MAKER INSIGHTS
    # ══════════════════════════════════════════
    elif page == "💡 Decision-Maker Insights":
        st.title("💡 Decision-Maker Insights & Recommendations")
        st.markdown("""
        This page synthesizes the three research questions into six concrete,
        evidence-backed recommendations for Spotify's product and growth teams.
        """)
        st.markdown("---")

        rq1_features, median_exp = build_rq1_features(df)
        rq3_features             = build_rq3_features(df, podcast_df)
        rq2_metrics, rq2_sessions = build_rq2_features(base_dir, df, podcast_df)

        exp_users = rq1_features[rq1_features["segment"] == "Explorer"]
        rep_users = rq1_features[rq1_features["segment"] == "Repeater"]

        # Summary KPI table
        st.markdown("### At-a-Glance Findings")

        summary_data = {
            "Dimension":    ["Exploration Rate", "Session Length", "Skip Rate",
                             "Podcast Share", "Device Context", "Peak Listening"],
            "Explorer / Mobile / Music": [
                f"{exp_users['exploration_rate'].mean()*100:.1f}%",
                f"{exp_users['avg_session_min'].mean():.1f} min",
                f"{rq3_features[rq3_features['listener_type']=='Music-Leaning']['skip_rate'].mean()*100:.1f}%",
                "< 20% podcast",
                "Mobile — more sessions",
                "Evening + late night",
            ],
            "Repeater / Multi-Device / Podcast": [
                f"{rep_users['exploration_rate'].mean()*100:.1f}%",
                f"{rep_users['avg_session_min'].mean():.1f} min",
                f"{rq3_features[rq3_features['listener_type']=='Podcast-Leaning']['skip_rate'].mean()*100:.1f}%",
                "> 20% podcast",
                "Multi-Device — longer sessions",
                "Evening (consistent)",
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🎯 Recommendations")

        recs = [
            ("RQ1 · Recommender Engine", "🔁 Personalized Discovery Nudges for Repeaters",
             "Repeaters have high completion rates but low variety — retention risk if the platform feels stale.",
             "For users with repeat_rate > 70%, surface one 'curated new track' per session sharing musical DNA with their top-repeated songs (BPM, key, mood vector).",
             "Gradual increase in unique track count; improved long-term retention."),

            ("RQ1 · Session Design", "⚡ Format Playlists by Listening Style",
             "Explorers open many short sessions; Repeaters have fewer, deeper sessions.",
             "Explorers → 'Quick Discovery' playlists (10 tracks, 90s previews). Repeaters → Auto-Queue with familiar content to extend sessions.",
             "+15–20% session duration for Explorers; reduced session abandonment for Repeaters."),

            ("RQ2 · Mobile UX", "📱 Frictionless Re-Entry for Mobile Users",
             "Mobile-only users open more sessions but each session is shorter — they are interrupted listeners.",
             "Prioritize a 'Resume Listening' card as the first UI element on app open; reduce tap-to-play latency; push personalized notifications at typical session-start times.",
             "Higher session start rate; increased daily active listening minutes."),

            ("RQ2 · Cross-Device", "🔄 Seamless Device Handoff for Multi-Device Users",
             "Multi-device users have longer sessions and more diverse content consumption.",
             "Invest in seamless playback transfer (desktop → phone → speaker) and unified history across devices. Promote autoplay and curated album/artist flows.",
             "Increased total listening time per user; stronger premium subscription retention."),

            ("RQ3 · Late Night", "🌙 Late-Night Playlist Push for Music-Heavy Users",
             "Music-leaning users actively listen until ~3 AM — a window currently underserved by editorial playlists.",
             "Auto-generate contextual late-night playlists based on listening history: low tempo, high familiarity, matching mood vectors from the past 30 minutes.",
             "Increased streams in the 12 AM–3 AM window; reduced overnight churn."),

            ("RQ3 · Long-Form Cross-Sell", "🎙️ Long-Form Content Recommendations for Podcast Users",
             "Podcast-leaning users have lower skip rates and higher depth — they are willing to commit to long-form content.",
             "For users with pct_podcast > 20%, recommend albums, artist radios, and extended mixes alongside podcast feed, especially during 6–9 PM peak.",
             "Higher cross-format engagement; increased revenue-per-user from mixed consumption."),
        ]

        for rq_label, title, insight, action, impact in recs:
            with st.expander(f"**{title}** _(_{rq_label}_)_"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**📊 Insight**\n\n{insight}")
                with c2:
                    st.markdown(f"**⚙️ Action**\n\n{action}")
                with c3:
                    st.markdown(f"**📈 Expected Impact**\n\n{impact}")

        st.markdown("---")
        st.markdown("### Summary Matrix")
        st.markdown("""
        | Dimension | Explorer | Repeater |
        |-----------|----------|----------|
        | Exploration Rate | High (≥ median) | Low (< median) |
        | Total Listening Time | Higher | Lower |
        | Track Completion | Moderate | Higher |
        | Session Depth | Moderate | Higher |
        | Sessions per Period | Higher | Lower |
        | Churn Risk | Low (engaged) | Medium (routine) |
        | Recommendation Strategy | Novelty-first | Familiarity + gentle nudges |

        > **Core takeaway:** Exploration behaviour is a *positive* engagement signal.
        Spotify's recommendation engine should treat high-exploration users as an asset and
        design features that deepen — rather than suppress — discovery-oriented listening.
        """)


if __name__ == "__main__":
    main()
