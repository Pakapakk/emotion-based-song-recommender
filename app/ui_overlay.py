import cv2


# Modern UI Theme Colors
COLORS = {
    "bg_dark": (20, 20, 30),

    # brown
    # "bg_panel": (30, 30, 45),
    # "bg_panel_light": (40, 40, 60),
    # "border": (100, 100, 120),

    # blue
    "bg_panel": (94, 39, 11),     
    "bg_panel_light": (138, 74, 33), 
    "border": (192, 129, 75),

    "accent_primary": (138, 43, 226),  # BlueViolet
    "accent_secondary": (30, 144, 255),  # DodgerBlue
    "accent_success": (50, 205, 50),  # LimeGreen
    "text_primary": (255, 255, 255),
    "text_secondary": (200, 200, 200),
    "text_muted": (150, 150, 150),
    
    "emotion_colors": {
        "happy": (255, 215, 0),  # Gold
        "sad": (70, 130, 180),  # SteelBlue
        "angry": (220, 20, 60),  # Crimson
        "surprise": (255, 165, 0),  # Orange
        "fear": (138, 43, 226),  # BlueViolet
        "disgust": (124, 252, 0),  # LawnGreen
        "neutral": (192, 192, 192),  # Silver
        "contempt": (255, 0, 0)
    },
}


def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10, alpha=0.8):
    """Draw a rounded rectangle with optional transparency."""
    x1, y1 = pt1
    x2, y2 = pt2

    overlay = img.copy()

    if thickness == -1:  # Filled
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    else:  # Outlined
        cv2.line(overlay, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(overlay, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(overlay, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(overlay, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(
            overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness
        )
        cv2.ellipse(
            overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness
        )
        cv2.ellipse(
            overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness
        )
        cv2.ellipse(
            overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness
        )

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        img[:] = overlay


def draw_text_with_bg(
    img, text, pos, font_scale, color, bg_color, thickness=2, padding=10, max_width=None
):
    """Draw text with a background panel; truncates if text overflows max_width."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if max_width is not None and text_width > max_width - (padding * 2):
        char_width = text_width / len(text)
        max_chars = int((max_width - (padding * 2)) / char_width) - 3
        if max_chars > 0:
            text = text[:max_chars] + "..."
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

    x, y = pos
    x1 = max(0, x - padding)
    y1 = max(0, y - text_height - padding)
    x2 = min(img.shape[1], x + text_width + padding)
    y2 = min(img.shape[0], y + baseline + padding)

    draw_rounded_rect(img, (x1, y1), (x2, y2), bg_color, thickness=-1, radius=8, alpha=0.95)

    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return (x2, y2)


def draw_info_panel(img, title, value, pos, width=280, value_color=None):
    """Draw a modern info panel with readable text."""
    x, y = pos
    panel_height = 90

    x = max(0, min(x, img.shape[1] - width))
    y = max(0, min(y, img.shape[0] - panel_height))

    draw_rounded_rect(
        img, (x, y), (x + width, y + panel_height), COLORS["bg_panel"], thickness=-1, radius=12, alpha=0.95
    )
    draw_rounded_rect(
        img, (x, y), (x + width, y + panel_height), COLORS["border"], thickness=2, radius=12, alpha=1.0
    )

    title_text = str(title)
    title_font_scale = 0.6
    title_thickness = 2
    (title_width, _), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_thickness)
    if title_width > width - 30:
        char_width = title_width / len(title_text)
        max_chars = int((width - 30) / char_width) - 3
        if max_chars > 0:
            title_text = title_text[:max_chars] + "..."

    title_x = x + 15
    title_y = y + 30
    cv2.putText(
        img,
        title_text,
        (title_x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_font_scale,
        (0, 0, 0),
        title_thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        title_text,
        (title_x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_font_scale,
        COLORS["text_muted"],
        title_thickness,
        cv2.LINE_AA,
    )

    value_text = str(value).upper()
    value_font_scale = 1.0
    value_thickness = 2
    (value_width, _), _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, value_font_scale, value_thickness)
    if value_width > width - 30:
        char_width = value_width / len(value_text)
        max_chars = int((width - 30) / char_width) - 3
        if max_chars > 0:
            value_text = value_text[:max_chars] + "..."
            (value_width, _), _ = cv2.getTextSize(
                value_text, cv2.FONT_HERSHEY_SIMPLEX, value_font_scale, value_thickness
            )

    value_color = value_color or COLORS["text_primary"]
    value_x = x + 15
    value_y = y + 65
    cv2.putText(
        img,
        value_text,
        (value_x, value_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        value_font_scale,
        (0, 0, 0),
        value_thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        value_text,
        (value_x, value_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        value_font_scale,
        value_color,
        value_thickness,
        cv2.LINE_AA,
    )

    return y + panel_height + 10


def get_emotion_color(emotion):
    """Return a color associated with a detected emotion."""
    if not emotion:
        return COLORS["text_primary"]
    return COLORS["emotion_colors"].get(emotion.lower(), COLORS["text_primary"])


def draw_face_box(img, x1, y1, x2, y2, emotion, confidence=1.0):
    """Draw modern face detection box with emotion label."""
    box_color = get_emotion_color(emotion)

    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
    cv2.rectangle(img, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), box_color, 1)

    label = emotion.upper() if emotion not in ("...", "unknown", None) else "DETECTING..."
    label_font_scale = 0.7
    label_thickness = 2
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness
    )

    label_bg_height = label_height + baseline + 12
    label_bg_width = label_width + 16
    label_y1 = max(0, y1 - label_bg_height)
    label_x2 = min(img.shape[1], x1 + label_bg_width)

    draw_rounded_rect(
        img, (x1, label_y1), (label_x2, y1), COLORS["bg_panel"], thickness=-1, radius=6, alpha=0.98
    )

    label_x = x1 + 8
    label_y = y1 - 8
    cv2.putText(
        img,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_font_scale,
        (0, 0, 0),
        label_thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_font_scale,
        box_color,
        label_thickness,
        cv2.LINE_AA,
    )


def draw_quit_button(img, pos, width=150, height=55, text="Quit"):
    """Draw a clickable quit button and return its bounding box."""
    x1, y1 = pos
    x2 = x1 + width
    y2 = y1 + height

    draw_rounded_rect(
        img, (x1, y1), (x2, y2), COLORS["accent_primary"], thickness=-1, radius=14, alpha=0.95
    )
    draw_rounded_rect(img, (x1, y1), (x2, y2), COLORS["border"], thickness=2, radius=14, alpha=1.0)

    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    text_x = x1 + (width - text_width) // 2
    text_y = y1 + (height + text_height) // 2

    cv2.putText(
        img,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        COLORS["text_primary"],
        thickness,
        cv2.LINE_AA,
    )

    return (x1, y1, x2, y2)


def draw_overlay(frame, agg_emotion, stable_emotion, current_tracks):
    """Render the full HUD overlay on the frame."""
    h, w = frame.shape[:2]

    title_text = "Emotion-Based Song Recommender"
    title_font_scale = 1.3
    title_thickness = 3
    (title_width, title_height), _ = cv2.getTextSize(
        title_text, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_thickness
    )
    title_x = max((w - title_width) // 2, 20)
    title_y = 70
    cv2.putText(
        frame,
        title_text,
        (title_x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_font_scale,
        (0, 0, 0),
        title_thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        title_text,
        (title_x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_font_scale,
        COLORS["text_primary"],
        title_thickness,
        cv2.LINE_AA,
    )

    quit_button_rect = draw_quit_button(frame, (20, 20))

    panel_x = w - 300
    panel_y = 20

    scene_color = (
        get_emotion_color(agg_emotion) if agg_emotion not in ("...", "unknown", None) else COLORS["text_muted"]
    )
    panel_y = draw_info_panel(
        frame,
        "Current Emotion",
        agg_emotion if agg_emotion not in ("...", "unknown", None) else "Detecting...",
        (panel_x, panel_y),
        value_color=scene_color,
    )

    stable_color = (
        get_emotion_color(stable_emotion) if stable_emotion not in ("...", "unknown", None) else COLORS["text_muted"]
    )
    # panel_y = draw_info_panel(
    #     frame,
    #     "Stable Emotion",
    #     stable_emotion if stable_emotion not in ("...", "unknown", None) else "Waiting...",
    #     (panel_x, panel_y),
    #     value_color=stable_color,
    # )

    if current_tracks:
        track = current_tracks[0]
        now_playing_text = f"{track['title']} - {track['artists']}"

        bottom_panel_y = h - 110
        panel_width = w - 30

        label_text = "Added to the queue"
        label_font_scale = 1.0
        label_thickness = 2
        label_x = 25
        label_y = bottom_panel_y + 30
        cv2.putText(
            frame,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_font_scale,
            (0, 0, 0),
            label_thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_font_scale,
            COLORS["text_muted"],
            label_thickness,
            cv2.LINE_AA,
        )

        track_font_scale = 0.9
        track_thickness = 2
        (track_width, _), _ = cv2.getTextSize(
            now_playing_text, cv2.FONT_HERSHEY_SIMPLEX, track_font_scale, track_thickness
        )
        max_track_width = panel_width - 50
        if track_width > max_track_width:
            char_width = track_width / len(now_playing_text)
            max_chars = int(max_track_width / char_width) - 3
            if max_chars > 0:
                now_playing_text = now_playing_text[:max_chars] + "..."

        track_x = 25
        track_y = bottom_panel_y + 65
        cv2.putText(
            frame,
            now_playing_text,
            (track_x, track_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            track_font_scale,
            (0, 0, 0),
            track_thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            now_playing_text,
            (track_x, track_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            track_font_scale,
            COLORS["text_primary"],
            track_thickness,
            cv2.LINE_AA,
        )

    return {"quit_button": quit_button_rect}

