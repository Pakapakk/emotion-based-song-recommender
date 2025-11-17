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
    Required columns: track_id, track_name, artist_name, plus FEATURE_KEYS.
    Returns: (df, id_index, name_index)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    req_cols = ["track_id", "track_name", "artist_name"]
    for c in req_cols + FEATURE_KEYS:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    df["track_id"] = df["track_id"].astype(str)
    df["norm_track_name"] = df["track_name"].astype(str).apply(_normalize_text)
    df["norm_artist_name"] = df["artist_name"].astype(str).apply(_normalize_text)

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
