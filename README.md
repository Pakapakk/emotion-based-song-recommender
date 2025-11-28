# Emotion-based Songs Recommender

## Project Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Prerequisite for Spotipy API
- Get the credential from https://developer.spotify.com/dashboard
- Click `Create app`
- In the `API used` section, select:
  - Web API
  - Web Playback SDK
- `.env` format
```.env
SPOTIPY_CLIENT_ID=
SPOTIPY_CLIENT_SECRET=
SPOTIPY_REDIRECT_URI="http://127.0.0.1:8888/callback"
```

### First Run
```commandline
python3 main.py
```
- You will see the `URI`, open it
- Click `agree/accept` the terms
- Copy the `URI` of the page that you are redirected to (you might see the 404 page)
- Paste it in the terminal (it will ask for this URI).
- Press `enter`

---

## 🚀 Features

### 🎭 Emotion Detection
- **YOLOv8 Face Detection** → real-time face detection  
- **YOLOv8 Emotion Classifier** → classify 8 emotions (FER+)  
- Supports **multiple faces** per frame  
- Emotion aggregation via majority vote + tie-breaking  

### 🧠 Emotion Stabilization
- Emotion must remain consistent for **≥ 0.5 seconds**  
- Filters out microexpressions (40–500ms)  
- Produces a **stable emotion** to drive recommendations  

### 🎵 Spotify Integration
- Polls playback every second  
- Reads:
  - current track  
  - progress  
  - time left  
- Queues new tracks smoothly when:
  - **emotion changes**, or  
  - **current song has ≤ 20s remaining**  

### 🎶 Smart Recommendations
- **Personalized Playlist Clustering**:
  - Uses **KMeans Clustering (k=8)** to group your playlist tracks into distinct musical styles.
  - Maps each cluster to an emotion (e.g., "Happy", "Sad") using **Cosine Similarity** against "ideal" emotion vectors.
  - *Example*: Your "Angry" cluster might contain Heavy Metal, while another user's "Angry" cluster contains Aggressive Rap.
- **Feature Engineering**:
  - Analyzes 10 audio features including **Valence** (positivity), **Energy**, **Tempo**, and **Danceability**.
  - Standardizes features to ensure fair comparison.
- **Hybrid Recommendation Strategy**:
  - **Primary**: Picks tracks from *your* personalized emotion cluster.
  - **Fallback**: Searches Spotify for generic queries (e.g., "happy upbeat pop") if your playlist lacks a specific emotion.
  - **Filtering**: Deduplicates tracks and maintains a history buffer to avoid repeating songs.

---

### 🤖 Models

The project uses 2 models in total, located in the `models/` directory:

- **`model_face_detection.pt`**: YOLOv8-based model fine-tuned for robust face detection.
  - Trained on: [Face Detection Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset/data)
- **`model_emotion_classifier.pt`**: YOLOv8-based classifier trained on the FER+ dataset to recognize 8 emotions:
  - `0: angry`
  - `1: contempt`
  - `2: disgust`
  - `3: fear`
  - `4: happy`
  - `5: neutral`
  - `6: sad`
  - `7: surprise`
  - Trained on: [FERPlus](https://www.kaggle.com/datasets/arnabkumarroy02/ferplus)

---

### ✅ Detailed Program Flow

1. **Initialization & Data Loading**
   - Load `spotipy` client and `data/spotifydata.csv` (audio features dataset).
   - Initialize `EmotionDetector` (YOLOv8 + FER+) and `EmotionRecommender`.

2. **User Personalization**
   - **Select Playlist**: User picks a playlist to define their taste.
   - **Feature Extraction**: App fetches tracks and matches them to the CSV to get audio features.
   - **Clustering**: Trains KMeans model to map user's music taste to emotions.

3. **Real-time Execution Loop**
   - **Webcam Capture**: Grabs frames from the camera.
   - **Face Analysis**: Detects faces, classifies emotions, and aggregates to find the dominant "Scene Emotion".
   - **Stabilization**: Emotion must be consistent for **0.5s** to trigger a change.

4. **Smart Recommendation Engine**
   - **Triggers**: Emotion changes or current song ending (≤ 20s remaining).
   - **Selection**: Generates a list of tracks using the personalized model + fallback search.
   - **Queueing**: Automatically adds chosen tracks to the active Spotify device's queue.
   - **UI Overlay**: Displays the detected emotion, stable emotion, and recommendation status.

---

## 🔬 Deep Dive: KMeans & Emotion Mapping

This section explains how the **Emotion Based Song Recommender** personalizes music recommendations using Unsupervised Learning (KMeans) and Vector Space Models.

### 1. The Core Concept
The goal is to map a user's existing music taste (from a selected playlist) to specific emotions. Since Spotify tracks don't have "emotion" labels, we infer them by:
1.  **Clustering** the user's tracks into groups of similar songs.
2.  **Mapping** each group to an emotion based on its audio features.

### 2. Feature Engineering
We use 10 audio features provided by Spotify. To make them comparable, we standardize them (mean=0, variance=1) using `StandardScaler`.

| Feature | Description |
| :--- | :--- |
| **Valence** | Musical positiveness (0.0 = sad/depressed, 1.0 = happy/cheerful). |
| **Energy** | Intensity and activity (fast, loud, noisy). |
| **Tempo** | Speed in Beats Per Minute (BPM). |
| **Danceability** | Suitability for dancing. |
| **Acousticness** | Confidence the track is acoustic. |
| **Liveness** | Presence of an audience. |
| **Speechiness** | Presence of spoken words. |
| **Instrumentalness** | Likelihood of no vocals. |
| **Loudness** | Overall loudness in dB. |
| **Duration** | Length of the track. |

### 3. Emotion Target Vectors
We define "ideal" feature profiles for 8 emotions. Since we only intuitively know Valence, Energy, and Tempo for emotions, we use **heuristics** to derive the other 7 features.

**Example: Happy**
*   **Defined**: Valence=0.9, Energy=0.8, Tempo=125
*   **Derived**:
    *   `danceability` = (0.9 + 0.8) / 2 = **0.85** (Happy songs are danceable)
    *   `acousticness` = 1.0 - 0.8 = **0.2** (Happy songs are usually electric/produced)
    *   `loudness` = -20 + (20 * 0.8) = **-4.0 dB** (Loud)

**Example: Sad**
*   **Defined**: Valence=0.2, Energy=0.2, Tempo=70
*   **Derived**:
    *   `danceability` = (0.2 + 0.2) / 2 = **0.2** (Not danceable)
    *   `acousticness` = 1.0 - 0.2 = **0.8** (Likely acoustic/stripped back)

### 4. The Algorithm

#### Step A: Clustering (KMeans)
We take all tracks from the user's playlist and feed them into a **KMeans Clustering** algorithm.
*   **k=8**: We force the algorithm to find 8 distinct groups (clusters) of songs in the playlist.
*   **Result**: 8 "Centers" (centroids), each representing the *average* song of that cluster.

#### Step B: Mapping Clusters to Emotions
We now have 8 Cluster Centers and 8 Emotion Target Vectors. We need to match them.
For each **Cluster Center**:
1.  Calculate the **Cosine Similarity** between this Center and *every* Emotion Target Vector.
2.  Assign the Cluster to the Emotion with the **highest similarity**.

> **Note**: This is a "greedy" assignment. It's possible for multiple clusters to map to "Happy" if the user has many happy-sounding sub-genres (e.g., Pop-Happy and Rock-Happy), and no clusters to map to "Disgust" if the user lacks that style of music.

#### Step C: Recommendation
When the system detects an emotion (e.g., "Sad"):
1.  It identifies the **Cluster** mapped to "Sad".
2.  It selects all tracks belonging to that cluster.
3.  It calculates the similarity between those tracks and the **Ideal Sad Vector**.
4.  It returns the top `k` tracks that are most similar to the ideal "Sad" profile.