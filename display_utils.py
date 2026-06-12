import cv2


def format_status_with_score(present_label, missing_label, match):
    """Return a human-readable status string with confidence when available."""
    if match:
        return f"{present_label} {match['score']:.2f}"
    return missing_label


def build_person_ppe_lines(base_label, helmet_match, vest_match, glove_match, goggle_match):
    """Return multi-line person label text for all PPE states."""
    return [
        base_label,
        format_status_with_score("Helmet", "No Helmet", helmet_match),
        format_status_with_score("Vest", "No Vest", vest_match),
        format_status_with_score("Gloves", "No Gloves", glove_match),
        format_status_with_score("Goggles", "No Goggles", goggle_match),
    ]


def draw_label_block(frame, lines, anchor, color, font_scale=0.65, thickness=2):
    """Draw a compact multi-line label block near a bounding box."""
    if not lines:
        return

    x, y = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 6
    line_gap = 6

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max(width for width, _ in text_sizes)
    line_height = max(height for _, height in text_sizes)
    block_height = padding * 2 + len(lines) * line_height + (len(lines) - 1) * line_gap

    top = max(0, y - block_height - 8)
    bottom = top + block_height
    right = x + max_width + padding * 2

    cv2.rectangle(frame, (x, top), (right, bottom), (20, 20, 20), -1)
    cv2.rectangle(frame, (x, top), (right, bottom), color, 2)

    baseline_y = top + padding + line_height
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x + padding, baseline_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        baseline_y += line_height + line_gap
