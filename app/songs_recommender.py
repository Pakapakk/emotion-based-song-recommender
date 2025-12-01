import os
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

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
    "happy":     {"target_valence": 0.9, "target_energy": 0.8, "target_tempo": 125},
    "sad":       {"target_valence": 0.08, "target_energy": 0.1, "target_tempo": 65},
    "angry":     {"target_valence": 0.35, "target_energy": 0.98, "target_tempo": 160},
    "surprise":  {"target_valence": 0.6, "target_energy": 0.6, "target_tempo": 115},
    "neutral":   {"target_valence": 0.5, "target_energy": 0.4, "target_tempo": 100},
    "disgust":   {"target_valence": 0.1, "target_energy": 0.7, "target_tempo": 130},
    "fear":      {"target_valence": 0.2, "target_energy": 0.5, "target_tempo": 85},
    "contempt":  {"target_valence": 0.3, "target_energy": 0.5, "target_tempo": 95},
}

EMOTIONS = list(EMO_MAP.keys())

FEATURE_KEYS = [
    "danceability",
    "energy",
    "valence",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "loudness",
    "tempo",
    "duration_ms",
]

SIMILARITY_NOISE_SCALE = 0.05


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


class EmotionRecommender:
    """
    Recommendation + Spotify control.
    - Expects enriched tracks with 'features' dict containing FEATURE_KEYS
      (e.g., from your CSV with full audio features).
    """

    def __init__(self) -> None:
        self.sp = _get_spotify_client()

        self.tracks: List[Dict] = []  # enriched tracks with 'features'
        self.X: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_to_emotion: Dict[int, str] = {}

        self.recent_track_ids: deque[str] = deque(maxlen=50)

    # ---------- model building ----------

    def build_model_from_enriched(self, enriched_tracks: List[Dict]) -> None:
        """
        Take a list of tracks, each with a 'features' dict for FEATURE_KEYS,
        and build the internal model (X, scaler, kmeans, etc.).
        """
        if not enriched_tracks:
            print("[Recommender] No enriched tracks provided; personalization disabled.")
            self.tracks, self.X, self.scaler, self.kmeans, self.cluster_to_emotion = (
                [],
                None,
                None,
                None,
                {},
            )
            return

        self.tracks = enriched_tracks
        Xs, scaler, _ = self._build_feature_matrix(enriched_tracks)
        self.X = Xs
        self.scaler = scaler

        if Xs.shape[0] >= len(EMOTIONS):
            km = KMeans(n_clusters=len(EMOTIONS), random_state=42, n_init="auto")
            km.fit(Xs)
            self.kmeans = km
            self.cluster_to_emotion = self._map_clusters_to_emotions(km, scaler)
            print(f"[Recommender] Model built with {len(self.tracks)} tracks; KMeans ready.")
        else:
            self.kmeans = None
            self.cluster_to_emotion = {}
            print(
                f"[Recommender] {len(self.tracks)} tracks available, "
                "but not enough for clustering. Using similarity-only personalization."
            )

    # ---------- public API ----------

    def get_recommendations(
        self,
        emotion: str,
        k: int = 5,
        use_personal: bool = True,
    ) -> List[Dict]:
        """
        Mixed strategy:
        - Use personal model (playlist-based) if available.
        - Also use Spotify search fallback.
        - Combine, de-duplicate, avoid recent tracks, shuffle, then pick k.
        """
        emo = emotion if emotion in EMO_MAP else "neutral"

        base_k = max(k, 3)

        pool_factor = 5
        personal_top_k = base_k * pool_factor

        fallback_pool_factor = 8
        fallback_top_k = base_k * fallback_pool_factor

        personal_tracks: List[Dict] = []
        if use_personal and self.X is not None and self.tracks is not None:
            personal_tracks = self._personalized_recs(emo, top_k=personal_top_k)

        fallback_tracks = self._fallback_spotify_recs(emo, top_k=fallback_top_k)

        all_candidates: List[Dict] = []
        seen_ids = set()

        for tr_list in (personal_tracks, fallback_tracks):
            for tr in tr_list:
                tid = tr.get("id")
                if not tid or tid in seen_ids:
                    continue
                seen_ids.add(tid)
                all_candidates.append(tr)

        if not all_candidates:
            return []

        # filter out recently recommended songs
        fresh_candidates = [
            tr for tr in all_candidates
            if tr.get("id") not in self.recent_track_ids
        ]

        # if filtering removes too many, fall back to all_candidates
        if len(fresh_candidates) < k:
            fresh_candidates = all_candidates

        # shuffle for variety
        random.shuffle(fresh_candidates)

        picks = fresh_candidates[:k]

        # update history
        for tr in picks:
            tid = tr.get("id")
            if tid:
                self.recent_track_ids.append(tid)

        return picks

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
        """
        Map emotion into the same feature space as FEATURE_KEYS.

        Uses EMO_MAP's valence/energy/tempo as anchors and fills
        other features with interpretable heuristics.
        """
        cfg = EMO_MAP.get(emotion, EMO_MAP["neutral"])
        valence = cfg["target_valence"]
        energy = cfg["target_energy"]
        tempo = cfg["target_tempo"]

        danceability = (valence + energy) / 2.0
        speechiness = 0.15 + 0.3 * energy
        acousticness = 1.0 - energy
        instrumentalness = 0.3 if valence > 0.4 else 0.5
        liveness = 0.2 + 0.3 * energy

        loudness = -20.0 + 20.0 * energy
        duration_ms = (3.5 - 1.0 * (energy - 0.5)) * 60_000

        base = {
            "danceability":     danceability,
            "energy":           energy,
            "valence":          valence,
            "speechiness":      speechiness,
            "acousticness":     acousticness,
            "instrumentalness": instrumentalness,
            "liveness":         liveness,
            "loudness":         loudness,
            "tempo":            float(tempo),
            "duration_ms":      float(duration_ms),
        }

        return np.array([base[k] for k in FEATURE_KEYS], dtype=np.float32).reshape(1, -1)

    def _map_clusters_to_emotions(
        self, kmeans: KMeans, scaler: StandardScaler
    ) -> Dict[int, str]:
        """
        Assign each KMeans cluster to the closest emotion target vector
        using cosine similarity in the scaled feature space.
        """
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
        """
        top_k here is treated as the size of the *candidate pool* (top-N by similarity),
        then we randomize inside that pool. Adds a bit of noise to similarity so that
        repeated runs don't always pick exactly the same tracks.
        """
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

        if not candidates:
            return []

        Xc = self.X[candidates]
        sims = cosine_similarity(Xc, v_scaled).reshape(-1)

        noise = np.random.normal(loc=0.0, scale=SIMILARITY_NOISE_SCALE, size=sims.shape)
        sims_noisy = sims + noise

        order = np.argsort(-sims_noisy)

        pool_size = min(top_k, len(order))
        pool_indices = [candidates[int(i)] for i in order[:pool_size]]

        random.shuffle(pool_indices)

        picks: List[Dict] = []
        for ti in pool_indices:
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
            "happy": "happy upbeat pop",
            "sad": "sad acoustic ballad",
            "angry": "aggressive rock rap",
            "surprise": "surprising experimental indie electronic",
            "neutral": "lofi chill beats",
            "disgust": "industrial metal",
            "fear": "dark ambient cinematic",
            "contempt": "indie mellow alt",
        }
        q = query_map.get(emotion, "chill pop")

        tracks: List[Dict] = []
        try:
            limit = max(top_k, 10)
            offset = random.randint(0, 40)
            results = self.sp.search(q=q, type="track", limit=limit, offset=offset)

            for t in results.get("tracks", {}).get("items", []):
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

        random.shuffle(tracks)
        return tracks

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
