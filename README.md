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
SPOTIPY_REDIRECT_URI=
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
- YOLOv12n-face → real-time face detection  
- FER+ ONNX → classify 7 emotions  
- Supports **multiple faces** per frame  
- Emotion aggregation via majority vote + tie-breaking  

### 🧠 Emotion Stabilization
- Emotion must remain consistent for **≥ 0.65 seconds**  
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
- Mix of:
  - **personalized playlist-based recommendations**
  - **global Spotify fallback recommendations**
- Uses:
  - Derived track features (popularity, followers, duration, etc.)
  - Optional KMeans clustering
  - Cosine similarity to emotion target vectors
- Deduplication + fresh-song filtering  
- Maintains a history buffer to avoid repeats  

---

## 🏗 System Flow

Webcam → YOLO Face Detect → FER+ Emotion Classify → Aggregate Scene Emotion
- Stabilize (0.65s) → Check Spotify Playback
- Recommendation Logic:
  - emotion changed → recommend now 
  - same emotion → recommend when ≤20s left
- Mixed Personalized + Global Recs → Queue in Spotify