import time
import cv2

from emotion_classifier import EmotionDetector
from songs_recommender import EmotionRecommender


def choose_playlist(rec: EmotionRecommender):
    """
    CLI helper:
    - list user's playlists
    - let them pick one
    - build model from that playlist
    """
    try:
        playlists = rec.sp.current_user_playlists(limit=20)
    except Exception as e:
        print("[Playlist fetch error]", e)
        print("Will use generic (non-personal) recommendations.")
        return

    items = playlists.get("items", [])
    if not items:
        print("No playlists found. Using generic recommendations.")
        return

    print("\nYour playlists:")
    for i, pl in enumerate(items):
        name = pl.get("name", "Untitled")
        owner = pl.get("owner", {}).get("display_name", "unknown")
        print(f"{i:2d}) {name}   (owner: {owner})")

    choice = input("Select playlist index to use for personalization (or press Enter to skip): ")
    if not choice.strip():
        print("Skipping personalization (fallback only).")
        return

    try:
        idx = int(choice)
        if idx < 0 or idx >= len(items):
            print("Invalid index. Skipping personalization.")
            return
    except ValueError:
        print("Invalid input. Skipping personalization.")
        return

    playlist_id = items[idx]["id"]
    print(f"Building model from playlist: {items[idx]['name']!r}...")
    rec.load_playlist(playlist_id)


def main():
    # ---- 1) init modules ----
    detector = EmotionDetector(
        det_weights="models/yolov12n-face.pt",
        emotion_onnx="models/emotion-ferplus-8.onnx",
        classify_every=3,
        conf=0.5,
    )
    recommender = EmotionRecommender()

    # optional personalization (build from chosen playlist)
    choose_playlist(recommender)

    # ---- 2) open webcam ----
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # Mac
    # cap = cv2.VideoCapture(0)  # Windows / Linux

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Webcam ready. Waiting 3 seconds to stabilize…")
    time.sleep(3)

    print("Press 'q' to quit...")

    # tracking for recommendation timing
    last_reco_emotion = None
    last_reco_time = 0.0
    SAME_EMO_INTERVAL = 210  # 3 minutes 30 seconds

    current_tracks = []

    # emotion stabilization for recommendation logic
    logic_emotion = None          # emotion used for logic/recommendation
    logic_emotion_age = 0         # how many consecutive frames we've seen this emotion
    LOGIC_WARMUP_FRAMES = 5       # require N frames before using emotion for recs

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not received")
            break

        # mirror for more natural interaction
        frame = cv2.flip(frame, 1)

        # ---- 3) run emotion detection ----
        # emotion = "visual" label (shown immediately)
        emotion, box = detector.process_frame(frame)

        # ---- update logic_emotion (used for decisions) ----
        if emotion == logic_emotion:
            logic_emotion_age += 1
        else:
            logic_emotion = emotion
            logic_emotion_age = 0  # reset age when emotion changes

        # ---- 4) draw face box + emotion (always use the immediate emotion) ----
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + 180, y1), (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"{emotion}",  # label updates as soon as detector changes
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

        # ---- 5) trigger recommendation based on STABILIZED emotion + timing rule ----
        t_now = time.time()

        # logic_emotion must be valid AND stable for a few frames
        valid_emotion = (
            logic_emotion not in ("...", "unknown", None)
            and logic_emotion_age >= LOGIC_WARMUP_FRAMES
        )

        should_recommend = False

        if valid_emotion:
            if last_reco_emotion is None:
                # first time we get a stable valid emotion
                should_recommend = True
            elif logic_emotion != last_reco_emotion:
                # stable emotion changed -> recommend
                should_recommend = True
            elif (t_now - last_reco_time) >= SAME_EMO_INTERVAL:
                # same emotion, but enough time has passed
                should_recommend = True

        if should_recommend:
            print(f"\n=== Recommendations for emotion: {logic_emotion} ===")
            current_tracks = recommender.get_recommendations(
                emotion=logic_emotion,
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

            last_reco_emotion = logic_emotion
            last_reco_time = t_now

        # ---- 6) display ----
        cv2.imshow("Emotion-Based Song Recommender", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
