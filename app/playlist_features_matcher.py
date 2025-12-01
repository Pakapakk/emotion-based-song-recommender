import os
import re
import unicodedata
from typing import Dict, Tuple, Optional, List

import pandas as pd
from spotipy import Spotify

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
    Load CSV dataset and build lookup dicts.

    Expected CSV schema (at minimum):
        track_id, artists, album_name, track_name, popularity, duration_ms,
        danceability, energy, valence, speechiness, acousticness,
        instrumentalness, liveness, loudness, tempo, ...

    Returns:
        df          : full dataframe
        id_index    : dict track_id -> row index
        name_index  : dict (norm_track_name, norm_artist_name) -> row index
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "artist_name" not in df.columns:
        if "artists" in df.columns:
            df["artist_name"] = (
                df["artists"]
                .astype(str)
                .str.split(";")
                .str[0]
                .str.strip()
            )
        else:
            raise ValueError("CSV missing both 'artist_name' and 'artists' columns")

    if "track_name" not in df.columns:
        if "name" in df.columns:
            df["track_name"] = df["name"]
        else:
            raise ValueError("CSV missing 'track_name' (or 'name') column")

    req_base = ["track_id", "track_name", "artist_name"]
    missing_base = [c for c in req_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"CSV missing required column(s): {missing_base}")

    missing_feat = [c for c in FEATURE_KEYS if c not in df.columns]
    if missing_feat:
        raise ValueError(f"CSV missing required feature columns: {missing_feat}")

    df["track_id"] = df["track_id"].astype(str)
    df["norm_track_name"] = df["track_name"].astype(str).apply(_normalize_text)
    df["norm_artist_name"] = df["artist_name"].astype(str).apply(_normalize_text)

    id_index: Dict[str, int] = {}
    for idx, tid in enumerate(df["track_id"]):
        if tid and tid not in id_index:
            id_index[tid] = idx

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

    if tid:
        tid_str = str(tid)
        if tid_str in id_index:
            return df.iloc[id_index[tid_str]]

    key = (_normalize_text(title), _normalize_text(main_artist))
    if key in name_index:
        return df.iloc[name_index[key]]

    return None


def fetch_playlist_tracks(sp: Spotify, playlist_id: str) -> List[Dict]:
    """
    Fetch all tracks from a playlist using Spotify API.
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
