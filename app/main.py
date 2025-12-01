import time
import cv2
from collections import Counter

from emotion_classifier import EmotionDetector
from songs_recommender import EmotionRecommender
from playlist_features_matcher import load_feature_dataset, build_enriched_tracks_from_playlist
from ui_overlay import draw_face_box, draw_overlay


def choose_playlist(rec: EmotionRecommender):
    """
    - list user's playlists
    - let them pick one
    - RETURN playlist_id (build the model separately)
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
    # ---- init modules ----
    detector = EmotionDetector(
        det_weights="models/model_face_detection.pt",
        emotion_pt="models/model_emotion_classifier.pt",
        classify_every=3,
        conf=0.5,
    )
    recommender = EmotionRecommender()

    csv_path = "data/spotifydata.csv"
    try:
        feature_df, id_index, name_index = load_feature_dataset(csv_path)
        print(f"[Main] Loaded feature dataset from {csv_path} with {len(feature_df)} tracks.")
    except Exception as e:
        print(f"[Main] Could not load feature dataset from {csv_path}: {e}")
        feature_df, id_index, name_index = None, None, None

    playlist_id = choose_playlist(recommender)
    if playlist_id and feature_df is not None:
        enriched_tracks = build_enriched_tracks_from_playlist(
            recommender.sp,
            playlist_id,
            feature_df,
            id_index,
            name_index,
        )
        recommender.build_model_from_enriched(enriched_tracks)
    elif playlist_id and feature_df is None:
        print("[Main] Playlist selected but feature dataset not available; using fallback recommendations only.")
    else:
        print("[Main] No playlist selected; using fallback recommendations only.")

    # ---- open webcam ----
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # Mac
    # cap = cv2.VideoCapture(0)  # Windows / Linux

    time.sleep(3)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Press 'q' to quit...")

    window_name = "Emotion-Based Song Recommender"
    cv2.namedWindow(window_name)
    quit_clicked = [False]
    button_rect = [None]

    def handle_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            rect = button_rect[0]
            if rect:
                x1, y1, x2, y2 = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    quit_clicked[0] = True

    cv2.setMouseCallback(window_name, handle_mouse)

    last_scene_emotion = "..."
    stable_emotion = "..."
    candidate_emotion = None
    stable_since = None
    STABLE_SECONDS = 0.5  # stable emotion for {STABLE_SECONDS} sec

    '''
    Microexpressions: 
    These are involuntary, very brief facial muscle movements that 
    reveal a person's true emotion, often when they are trying to conceal it.
    They typically last between 0.04 and 0.5 seconds (40 to 500 milliseconds).
    '''

    last_reco_emotion = None
    current_tracks = []

    playback_check_interval = 1.0
    last_playback_check = 0.0
    current_track_id = None
    seconds_to_end = None
    last_queued_for_track_id = None

    while True:
        ret, frame = cap.read()
        # if not ret:
        #     print("Error: Frame not received")
        #     break

        frame = cv2.flip(frame, 1)

        # detect multiple faces + emotions
        emo_boxes = detector.process_frame_multi(frame)

        valid_face_emotions = []
        for emo, box in emo_boxes:
            if box is None:
                continue
            x1, y1, x2, y2 = box
            draw_face_box(frame, x1, y1, x2, y2, emo)

            if emo not in ("...", "unknown", None):
                valid_face_emotions.append(emo)

        # aggregate scene emotion with tie-breaking
        if valid_face_emotions:
            counts = Counter(valid_face_emotions)
            max_count = max(counts.values())
            candidates = [e for e, c in counts.items() if c == max_count]

            if len(candidates) == 1:
                agg_emotion = candidates[0]
            else:
                if last_scene_emotion in candidates:
                    agg_emotion = last_scene_emotion
                elif "neutral" in candidates:
                    agg_emotion = "neutral"
                else:
                    agg_emotion = sorted(candidates)[0]
        else:
            agg_emotion = "..."

        last_scene_emotion = agg_emotion

        # stable emotion
        t_now = time.time()
        if agg_emotion not in ("...", "unknown", None):
            if agg_emotion != candidate_emotion:
                candidate_emotion = agg_emotion
                stable_since = t_now
                # Reset stable emotion when candidate changes
                stable_emotion = "..."
            else:
                if stable_since is not None and (t_now - stable_since) >= STABLE_SECONDS:
                    stable_emotion = agg_emotion
        else:
            candidate_emotion = None
            stable_since = None
            # Reset stable emotion when no valid emotion detected
            stable_emotion = "..."

        # poll Spotify playback every 1s
        if (t_now - last_playback_check) >= playback_check_interval:
            last_playback_check = t_now
            try:
                playback = recommender.sp.current_playback()
            except Exception as e:
                print("[Playback error]", e)
                playback = None

            if playback and playback.get("is_playing"):
                item = playback.get("item") or {}
                current_track_id = item.get("id")
                duration_ms = item.get("duration_ms") or 0
                progress_ms = playback.get("progress_ms") or 0
                if duration_ms > 0:
                    seconds_to_end = max(0.0, (duration_ms - progress_ms) / 1000.0)
                else:
                    seconds_to_end = None
            else:
                current_track_id = None
                seconds_to_end = None

        overlay_state = draw_overlay(
            frame=frame,
            agg_emotion=agg_emotion,
            stable_emotion=stable_emotion,
            current_tracks=current_tracks,
        )
        button_rect[0] = overlay_state.get("quit_button")

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if quit_clicked[0]:
            break

        def do_recommend_and_queue(reason: str):
            nonlocal current_tracks, last_reco_emotion, last_queued_for_track_id
            print(f"\n=== Recommendations for emotion (stable): {stable_emotion} | reason: {reason} ===")
            current_tracks = recommender.get_recommendations(
                emotion=stable_emotion,
                k=1,
                use_personal=True,
            )
            if not current_tracks:
                print("No tracks returned.")
                return

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
                        "Make sure you have an active Spotify device "
                        "(Spotify app open & playing)."
                    )
            except Exception as e:
                print("[Queue error]", e)

            last_reco_emotion = stable_emotion
            last_queued_for_track_id = current_track_id

        # invalid stable emotion -> do nothing
        if stable_emotion in ("...", "unknown", None):
            continue

        # CASE 1: emotion changed -> recommend immediately
        if last_reco_emotion is None or stable_emotion != last_reco_emotion:
            do_recommend_and_queue("emotion changed")
            continue

        # CASE 2: emotion unchanged -> recommend when song is about to end (<= 20s)
        if (
                current_track_id is not None
                and seconds_to_end is not None
                and seconds_to_end <= 20.0
                and current_track_id != last_queued_for_track_id
        ):
            do_recommend_and_queue("song ending soon")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
