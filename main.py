# main.py
import time
import cv2

from emotion_classifier import EmotionDetector
from songs_recommender import EmotionRecommender
from playlist_features_matcher import load_feature_dataset, build_enriched_tracks_from_playlist


def choose_playlist(rec: EmotionRecommender):
    """
    CLI helper:
    - list user's playlists
    - let them pick one
    - RETURN playlist_id (we'll build the model separately)
    """
    try:
        playlists = rec.sp.current_user_playlists(limit=20)
    except Exception as e:
        print("[Playlist fetch error]", e)
        print("Will use generic (non-personal) recommendations.")
        return None

    items = playlists.get("items", [])
    if not items:
        print("No playlists found. Using generic recommendations.")
        return None

    print("\nYour playlists:")
    for i, pl in enumerate(items):
        name = pl.get("name", "Untitled")
        owner = pl.get("owner", {}).get("display_name", "unknown")
        print(f"{i:2d}) {name}   (owner: {owner})")

    choice = input("Select playlist index to use for personalization (or press Enter to skip): ")
    if not choice.strip():
        print("Skipping personalization (fallback only).")
        return None

    try:
        idx = int(choice)
        if idx < 0 or idx >= len(items):
            print("Invalid index. Skipping personalization.")
            return None
    except ValueError:
        print("Invalid input. Skipping personalization.")
        return None

    playlist_id = items[idx]["id"]
    print(f"Using playlist: {items[idx]['name']!r} for personalization...")
    return playlist_id


def main():
    # ---- 1) init modules ----
    detector = EmotionDetector(
        det_weights="models/yolov12n-face.pt",
        emotion_onnx="models/emotion-ferplus-8.onnx",
        classify_every=3,
        conf=0.5,
    )
    recommender = EmotionRecommender()

    # ---- 1.1) load feature dataset ----
    csv_path = "data/spotify_data clean.csv"
    try:
        feature_df, id_index, name_index = load_feature_dataset(csv_path)
        print(f"[Main] Feature dataset loaded from {csv_path}")
    except Exception as e:
        print(f"[Main] Could not load feature dataset ({e}).")
        feature_df = id_index = name_index = None

    # ---- 1.2) optional personalization (choose playlist + match features + build model) ----
    playlist_id = None
    if feature_df is not None:
        playlist_id = choose_playlist(recommender)
        if playlist_id:
            enriched_tracks = build_enriched_tracks_from_playlist(
                recommender.sp,
                playlist_id,
                feature_df,
                id_index,
                name_index,
            )
            recommender.build_model_from_enriched(enriched_tracks)
        else:
            print("[Main] No playlist selected; using fallback recommendations only.")
    else:
        print("[Main] No feature dataset; using fallback recommendations only.")

    # ---- 2) open webcam ----
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # Mac
    # cap = cv2.VideoCapture(0)  # Windows / Linux

    time.sleep(3)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Press 'q' to quit...")

    # tracking for recommendation timing
    last_reco_emotion = None
    last_reco_time = 0.0
    SAME_EMO_INTERVAL = 210  # 3 minutes 30 seconds

    current_tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not received")
            break

        # mirror for more natural interaction
        frame = cv2.flip(frame, 1)

        # ---- 3) run emotion detection ----
        emotion, box = detector.process_frame(frame)

        # ---- 4) draw face box + emotion ----
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + 180, y1), (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"{emotion}",
                (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # show first recommended track (if any) at bottom
        if current_tracks:
            now_playing = f"{current_tracks[0]['title']} - {current_tracks[0]['artists']}"
            cv2.putText(
                frame,
                now_playing[:60],
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # ---- 6) display FIRST ----
        cv2.imshow("Emotion-Based Song Recommender", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # ---- 5) AFTER display: trigger recommendation based on emotion + timing rule ----
        t_now = time.time()
        valid_emotion = emotion not in ("...", "unknown", None)
        should_recommend = False

        if valid_emotion:
            if last_reco_emotion is None:
                should_recommend = True
            elif emotion != last_reco_emotion:
                should_recommend = True
            elif (t_now - last_reco_time) >= SAME_EMO_INTERVAL:
                should_recommend = True

        if should_recommend:
            print(f"\n=== Recommendations for emotion: {emotion} ===")
            current_tracks = recommender.get_recommendations(
                emotion=emotion,
                k=2,
                use_personal=True,
            )

            if not current_tracks:
                print("No tracks returned.")
            else:
                for i, tr in enumerate(current_tracks, start=1):
                    print(f"{i}. {tr['title']} — {tr['artists']}")
                    print(f"   {tr['url']}")

                try:
                    queued = recommender.queue_tracks(current_tracks)
                    if queued > 0:
                        print(f"Queued {queued} track(s) to your active Spotify device.")
                    else:
                        print(
                            "Tracks were NOT queued. "
                            "Make sure you have an active Spotify device (Spotify app open & playing)."
                        )
                except Exception as e:
                    print("[Queue error]", e)

            last_reco_emotion = emotion
            last_reco_time = t_now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
