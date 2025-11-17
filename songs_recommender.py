# songs_recommender.py
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv

load_dotenv()

# ------------------- config -------------------

EMO_MAP: Dict[str, Dict] = {
    "happiness": {"target_valence": 0.9, "target_energy": 0.8, "target_tempo": 125},
    "sadness":   {"target_valence": 0.2, "target_energy": 0.2, "target_tempo": 70},
    "anger":     {"target_valence": 0.2, "target_energy": 0.9, "target_tempo": 140},
    "surprise":  {"target_valence": 0.6, "target_energy": 0.6, "target_tempo": 115},
    "neutral":   {"target_valence": 0.5, "target_energy": 0.4, "target_tempo": 100},
    "disgust":   {"target_valence": 0.1, "target_energy": 0.7, "target_tempo": 130},
    "fear":      {"target_valence": 0.2, "target_energy": 0.5, "target_tempo": 85},
}

EMOTIONS = list(EMO_MAP.keys())

FEATURE_KEYS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
]


# ---------- Spotify clients ----------

def _get_spotify_client() -> spotipy.Spotify:
    """
    User-authenticated client (playlists, library, queue, search, etc.).
    """
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

    if client_id and client_secret and redirect_uri:
        scopes = [
            "user-read-private",
            "user-top-read",
            "playlist-read-private",
            "user-library-read",
            "user-modify-playback-state",
            "user-read-playback-state",
        ]
        return spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=" ".join(scopes),
                open_browser=False,
            )
        )

    if client_id and client_secret:
        # fallback: app-only, no user features
        return spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret,
            )
        )

    raise RuntimeError(
        "Spotify credentials not configured. "
        "Set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, "
        "and optionally SPOTIPY_REDIRECT_URI in your .env file."
    )


def _get_spotify_app_client() -> Optional[spotipy.Spotify]:
    """
    Client-credentials client for endpoints that don't need user scopes
    (e.g., audio_features). This avoids weird 403s for some user tokens.
    """
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

    if not (client_id and client_secret):
        return None

    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        )
    )


# ---------- Recommender ----------

class EmotionRecommender:
    def __init__(self) -> None:
        # user client (playlists, library, queue, search, etc.)
        self.sp = _get_spotify_client()
        # app client (audio_features)
        self.sp_app = _get_spotify_app_client()

        self.tracks: List[Dict] = []
        self.X: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_to_emotion: Dict[int, str] = {}

    # ---------- public API ----------

    def build_from_library(self, limit_tracks: int = 400) -> None:
        tracks = self._fetch_user_library(limit_tracks)
        self._build_model_from_tracks(tracks)
        print(f"[Library model] {len(self.tracks)} usable tracks.")

    def load_playlist(self, playlist_id: str) -> None:
        raw_tracks: List[Dict] = []
        seen = set()
        offset = 0

        while True:
            page = self.sp.playlist_items(playlist_id, limit=50, offset=offset)
            items = page.get("items", [])
            if not items:
                break

            for it in items:
                tr = it.get("track")
                if not tr:
                    continue
                tid = tr.get("id")
                if not tid:
                    continue
                if tid in seen:
                    continue
                seen.add(tid)
                raw_tracks.append(tr)

            offset += 50

        self._build_model_from_tracks(raw_tracks)
        print(f"[Playlist model] {len(self.tracks)} usable tracks.")

    def get_recommendations(
        self,
        emotion: str,
        k: int = 5,
        use_personal: bool = True,
    ) -> List[Dict]:
        emo = emotion if emotion in EMO_MAP else "neutral"

        if use_personal and self.X is not None and self.tracks:
            tracks = self._personalized_recs(emo, top_k=k)
            if tracks:
                return tracks

        return self._fallback_spotify_recs(emo, top_k=k)

    # ---------- building model ----------

    def _build_model_from_tracks(self, raw_tracks: List[Dict]) -> None:
        if not raw_tracks:
            self.tracks, self.X, self.scaler, self.kmeans, self.cluster_to_emotion = (
                [],
                None,
                None,
                None,
                {},
            )
            return

        ids = [t["id"] for t in raw_tracks if t.get("id")]
        feats = self._fetch_audio_features(ids)

        enriched = []
        for t in raw_tracks:
            tid = t.get("id")
            if tid in feats:
                enriched.append(
                    {
                        "id": tid,
                        "name": t["name"],
                        "artists": t.get("artists", []),
                        "external_urls": t.get("external_urls", {}),
                        "uri": t.get("uri"),
                        "features": feats[tid],
                    }
                )

        self.tracks = enriched
        Xs, scaler, _ = self._build_feature_matrix(enriched)
        self.X = Xs
        self.scaler = scaler

        if Xs.shape[0] >= len(EMOTIONS):
            km = KMeans(n_clusters=len(EMOTIONS), random_state=42, n_init="auto")
            km.fit(Xs)
            self.kmeans = km
            self.cluster_to_emotion = self._map_clusters_to_emotions(km, scaler)
        else:
            self.kmeans = None
            self.cluster_to_emotion = {}

    def _fetch_user_library(self, limit_tracks: int) -> List[Dict]:
        out: List[Dict] = []
        seen = set()

        # saved tracks
        try:
            offset = 0
            while offset < limit_tracks:
                page = self.sp.current_user_saved_tracks(limit=50, offset=offset)
                items = page.get("items", [])
                if not items:
                    break
                for it in items:
                    tr = it["track"]
                    if tr and tr.get("id") and tr["id"] not in seen:
                        out.append(tr)
                        seen.add(tr["id"])
                offset += 50
        except Exception:
            pass

        # a few playlists
        try:
            pls = self.sp.current_user_playlists(limit=5).get("items", [])
            for pl in pls:
                pl_id = pl["id"]
                offset = 0
                fetched = 0
                while offset < 200:
                    page = self.sp.playlist_items(pl_id, limit=50, offset=offset)
                    items = page.get("items", [])
                    if not items:
                        break
                    for it in items:
                        tr = it.get("track")
                        if not tr or not tr.get("id"):
                            continue
                        if tr["id"] not in seen:
                            out.append(tr)
                            seen.add(tr["id"])
                            fetched += 1
                    if fetched >= 200:
                        break
                    offset += 50
        except Exception:
            pass

        return out

    def _fetch_audio_features(self, track_ids: List[str]) -> Dict[str, Dict]:
        feats: Dict[str, Dict] = {}
        BATCH = 20  # smaller + slower on purpose

        printed_forbidden = False  # avoid spam

        # Prefer app client (client credentials); fall back to user client
        client = self.sp_app or self.sp

        for i in range(0, len(track_ids), BATCH):
            batch = track_ids[i: i + BATCH]
            try:
                af = client.audio_features(batch)
                for tid, f in zip(batch, af):
                    if f:
                        feats[tid] = f
                time.sleep(0.15)  # tiny pause between batches
            except SpotifyException as e:
                # 401 / 403 typically won't recover for further batches
                if e.http_status in (401, 403):
                    if not printed_forbidden:
                        print(
                            f"[audio_features WARNING] HTTP {e.http_status} "
                            "for this token / some tracks. "
                            "Skipping remaining features (fallback will still work)."
                        )
                        printed_forbidden = True
                    break
                else:
                    if not printed_forbidden:
                        print("[audio_features WARNING] SpotifyException:", e.http_status)
                        printed_forbidden = True
                    continue
            except Exception as ex:
                if not printed_forbidden:
                    print("[audio_features WARNING] non-Spotify error, some tracks skipped.", ex)
                    printed_forbidden = True
                continue

        return feats

    # ---------- feature matrix / targets / scoring ----------

    def _build_feature_matrix(
        self, tracks: List[Dict]
    ) -> Tuple[np.ndarray, StandardScaler, List[int]]:
        rows = []
        valid_idx = []
        for idx, tr in enumerate(tracks):
            f = tr.get("features") or {}
            if all(k in f and f[k] is not None for k in FEATURE_KEYS):
                row = [float(f[k]) for k in FEATURE_KEYS]
                rows.append(row)
                valid_idx.append(idx)

        if not rows:
            scaler = StandardScaler()
            return np.empty((0, len(FEATURE_KEYS)), dtype=np.float32), scaler, []

        X = np.array(rows, dtype=np.float32)
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        return Xs, scaler, valid_idx

    def _emotion_target_vector(self, emotion: str) -> np.ndarray:
        cfg = EMO_MAP.get(emotion, EMO_MAP["neutral"])
        base = {
            "danceability": 0.5,
            "energy": cfg["target_energy"],
            "valence": cfg["target_valence"],
            "tempo": float(cfg["target_tempo"]),
            "acousticness": 0.5,
            "instrumentalness": 0.0,
            "liveness": 0.2,
            "speechiness": 0.1,
        }
        return np.array([base[k] for k in FEATURE_KEYS], dtype=np.float32).reshape(
            1, -1
        )

    def _map_clusters_to_emotions(
        self, kmeans: KMeans, scaler: StandardScaler
    ) -> Dict[int, str]:
        cluster_to_emotion: Dict[int, str] = {}
        centers = kmeans.cluster_centers_

        tgt_scaled: Dict[str, np.ndarray] = {}
        for emo in EMOTIONS:
            vec = self._emotion_target_vector(emo)
            v_scaled = (vec - scaler.mean_) / scaler.scale_
            tgt_scaled[emo] = v_scaled.astype(np.float32)

        for ci, c in enumerate(centers):
            best_emo = None
            best_sim = -1.0
            for emo, v in tgt_scaled.items():
                sim = float(cosine_similarity(c.reshape(1, -1), v)[0, 0])
                if sim > best_sim:
                    best_sim = sim
                    best_emo = emo
            cluster_to_emotion[ci] = best_emo or "neutral"
        return cluster_to_emotion

    def _personalized_recs(self, emotion: str, top_k: int) -> List[Dict]:
        if self.X is None or not self.tracks or self.scaler is None:
            return []

        vec = self._emotion_target_vector(emotion)
        v_scaled = (vec - self.scaler.mean_) / self.scaler.scale_

        candidates = list(range(len(self.tracks)))
        if self.kmeans is not None and self.cluster_to_emotion:
            cid = None
            for c, emo in self.cluster_to_emotion.items():
                if emo == emotion:
                    cid = c
                    break
            if cid is not None:
                labels = self.kmeans.labels_
                cand = [i for i, lbl in enumerate(labels) if lbl == cid]
                if cand:
                    candidates = cand

        Xc = self.X[candidates]
        sims = cosine_similarity(Xc, v_scaled).reshape(-1)
        order = np.argsort(-sims)

        picks: List[Dict] = []
        for idx in order[:top_k]:
            ti = candidates[int(idx)]
            t = self.tracks[ti]
            picks.append(
                {
                    "id": t["id"],
                    "uri": t.get("uri") or f"spotify:track:{t['id']}",
                    "title": t["name"],
                    "artists": ", ".join(a["name"] for a in t.get("artists", [])),
                    "url": t.get("external_urls", {}).get("spotify", ""),
                }
            )
        return picks

    def _fallback_spotify_recs(self, emotion: str, top_k: int) -> List[Dict]:
        query_map = {
            "happiness": "happy upbeat pop",
            "sadness": "sad acoustic ballad",
            "anger": "aggressive rock rap",
            "surprise": "surprising experimental indie electronic",
            "neutral": "lofi chill beats",
            "disgust": "industrial metal",
            "fear": "dark ambient cinematic",
        }
        q = query_map.get(emotion, "chill pop")

        tracks: List[Dict] = []
        try:
            results = self.sp.search(q=q, type="track", limit=max(top_k, 10))
            for t in results.get("tracks", {}).get("items", [])[:top_k]:
                tracks.append(
                    {
                        "id": t["id"],
                        "uri": t["uri"],
                        "title": t["name"],
                        "artists": ", ".join(a["name"] for a in t["artists"]),
                        "url": t["external_urls"]["spotify"],
                    }
                )
        except Exception as e:
            print("[Spotify fallback search error]", e)

        return tracks

    # ---------- queue control ----------

    def queue_tracks(self, tracks: List[Dict]) -> int:
        """
        Queue the given tracks on the user's currently active Spotify device.
        Returns the number of tracks successfully queued.
        """
        if not tracks:
            return 0

        queued = 0

        for tr in tracks:
            uri = tr.get("uri")
            if not uri:
                tid = tr.get("id")
                if tid:
                    uri = f"spotify:track:{tid}"
                else:
                    continue

            try:
                self.sp.add_to_queue(uri=uri)
                queued += 1
                time.sleep(0.1)
            except SpotifyException as e:
                # Common: no active device or insufficient scope
                if e.http_status == 404 and "NO_ACTIVE_DEVICE" in str(e):
                    print(
                        "[queue WARNING] No active Spotify device found.\n"
                        "Open Spotify on your phone/PC, start playing a song,\n"
                        "then trigger a new recommendation."
                    )
                    break
                print(f"[queue WARNING] HTTP {e.http_status} while queueing {uri}: {e}")
                break
            except Exception as e:
                print(f"[queue WARNING] non-Spotify error for {uri}: {e}")
                break

        return queued
