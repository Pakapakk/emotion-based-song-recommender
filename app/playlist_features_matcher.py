import os
import re
import unicodedata
from typing import Dict, Tuple, Optional, List

import pandas as pd
from spotipy import Spotify

# We'll reuse the feature columns from songs_recommender
from songs_recommender import FEATURE_KEYS


def _normalize_text(s: str) -> str:
    """Normalize strings for fuzzy matching."""
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # remove feat/ft
    s = re.sub(r"\(feat\.?.*?\)|\[feat\.?.*?\]|feat\.?.*|ft\.?.*", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_feature_dataset(csv_path: str):
    """
    Load your CSV dataset and build lookup dicts.

    - Accepts your current schema (e.g. spotify_hit.csv) with columns like:
        track_id, artists, album_name, track_name, popularity, duration_ms, ...
    - Derives FEATURE_KEYS columns required by EmotionRecommender:
        track_popularity, artist_popularity, artist_followers,
        album_total_tracks, track_duration_min
    - Builds:
        df          : full dataframe with derived feature columns
        id_index    : dict track_id -> row index
        name_index  : dict (norm_track_name, norm_artist_name) -> row index
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # --- 1) Make sure we have track_name and artist_name columns ---

    # Your CSV uses 'artists' (string or list-like). We'll map it to 'artist_name'
    if "artist_name" not in df.columns:
        if "artists" in df.columns:
            df["artist_name"] = df["artists"]
        else:
            raise ValueError("CSV missing both 'artist_name' and 'artists' columns")

    if "track_name" not in df.columns:
        # some datasets use 'name' for track title
        if "name" in df.columns:
            df["track_name"] = df["name"]
        else:
            raise ValueError("CSV missing 'track_name' (or 'name') column")

    # Required base columns
    req_base = ["track_id", "track_name", "artist_name"]
    missing_base = [c for c in req_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"CSV missing required column(s): {missing_base}")

    # --- 2) Derive FEATURE_KEYS if they are missing ---

    # FEATURE_KEYS = [
    #     "track_popularity",
    #     "artist_popularity",
    #     "artist_followers",
    #     "album_total_tracks",
    #     "track_duration_min",
    # ]

    # track_popularity / artist_popularity from 'popularity'
    if "track_popularity" not in df.columns:
        if "popularity" not in df.columns:
            raise ValueError("CSV missing 'popularity' needed to derive track_popularity.")
        df["track_popularity"] = df["popularity"].astype(float)

    if "artist_popularity" not in df.columns:
        # reuse track popularity as a proxy
        df["artist_popularity"] = df["track_popularity"].astype(float)

    # artist_followers (we don't have this in CSV, so approximate with 0 or any heuristic)
    if "artist_followers" not in df.columns:
        df["artist_followers"] = 0.0

    # album_total_tracks = how many tracks share the same album_name
    if "album_total_tracks" not in df.columns:
        if "album_name" in df.columns:
            df["album_total_tracks"] = (
                df.groupby("album_name")["album_name"].transform("size").astype(float)
            )
        else:
            # fallback: assume single-track album
            df["album_total_tracks"] = 1.0

    # track_duration_min from duration_ms
    if "track_duration_min" not in df.columns:
        if "duration_ms" not in df.columns:
            raise ValueError("CSV missing 'duration_ms' needed to derive track_duration_min.")
        df["track_duration_min"] = df["duration_ms"].astype(float) / 60000.0

    # Ensure all FEATURE_KEYS are present now
    for c in FEATURE_KEYS:
        if c not in df.columns:
            raise ValueError(f"CSV still missing derived feature column: {c}")

    # --- 3) Add normalized columns for name-based matching ---

    df["track_id"] = df["track_id"].astype(str)
    df["norm_track_name"] = df["track_name"].astype(str).apply(_normalize_text)
    df["norm_artist_name"] = df["artist_name"].astype(str).apply(_normalize_text)

    # --- 4) Build indices ---

    # track_id -> index
    id_index: Dict[str, int] = {}
    for idx, tid in enumerate(df["track_id"]):
        if tid and tid not in id_index:
            id_index[tid] = idx

    # (norm_track_name, norm_artist_name) -> index
    name_index: Dict[Tuple[str, str], int] = {}
    for idx, row in df.iterrows():
        key = (row["norm_track_name"], row["norm_artist_name"])
        if key[0] and key[1] and key not in name_index:
            name_index[key] = idx

    return df, id_index, name_index


def _match_track_to_dataset(
    sp_track: Dict,
    df: pd.DataFrame,
    id_index: Dict[str, int],
    name_index: Dict[Tuple[str, str], int],
) -> Optional[pd.Series]:
    """
    Match a Spotify track to a row in the CSV dataset:
    1) by track_id (assuming CSV track_id is Spotify track ID)
    2) by normalized (track_name, artist_name)
    """
    tid = sp_track.get("id")
    title = sp_track.get("name", "")
    artists = sp_track.get("artists") or []
    main_artist = artists[0]["name"] if artists else ""

    # 1) ID match
    if tid:
        tid_str = str(tid)
        if tid_str in id_index:
            return df.iloc[id_index[tid_str]]

    # 2) name + artist
    key = (_normalize_text(title), _normalize_text(main_artist))
    if key in name_index:
        return df.iloc[name_index[key]]

    return None


def fetch_playlist_tracks(sp: Spotify, playlist_id: str) -> List[Dict]:
    """
    Fetch all tracks from a playlist.
    """
    out: List[Dict] = []
    offset = 0
    while True:
        page = sp.playlist_items(playlist_id, limit=100, offset=offset)
        items = page.get("items", [])
        if not items:
            break
        for it in items:
            tr = it.get("track")
            if tr:
                out.append(tr)
        offset += 100
    return out


def build_enriched_tracks_from_playlist(
    sp: Spotify,
    playlist_id: str,
    feature_df: pd.DataFrame,
    id_index: Dict[str, int],
    name_index: Dict[Tuple[str, str], int],
) -> List[Dict]:
    """
    - Fetch tracks from playlist
    - Match each track to CSV
    - Return list of enriched tracks, each with a 'features' dict for FEATURE_KEYS
    """
    playlist_tracks = fetch_playlist_tracks(sp, playlist_id)
    print(f"[Matcher] Fetched {len(playlist_tracks)} tracks from playlist.")

    enriched: List[Dict] = []
    matched = 0

    for tr in playlist_tracks:
        title = tr.get("name", "")
        artists = tr.get("artists") or []
        main_artist = artists[0]["name"] if artists else ""

        row = _match_track_to_dataset(tr, feature_df, id_index, name_index)
        if row is None:
            print(f"[NO MATCH] {title} — {main_artist}")
            continue

        try:
            feats = {k: float(row[k]) for k in FEATURE_KEYS}
        except Exception:
            print(f"[SKIP] {title} — {main_artist}: missing/invalid feature(s)")
            continue

        enriched.append(
            {
                "id": tr["id"],
                "name": tr["name"],
                "artists": tr.get("artists", []),
                "external_urls": tr.get("external_urls", {}),
                "uri": tr.get("uri"),
                "features": feats,
            }
        )
        matched += 1
        print(f"[MATCH] {title} — {main_artist}")

    print(f"[Matcher] {matched} / {len(playlist_tracks)} tracks matched with features.")
    return enriched
