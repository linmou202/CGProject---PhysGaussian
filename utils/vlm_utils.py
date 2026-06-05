import json
import os
from pathlib import Path

import numpy as np


_BOX_COLORS = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "generate_bounded_image requires opencv-python. "
            "Install the project requirements before drawing bounded images."
        ) from exc
    return cv2


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "get_initial_params requires torch to return Young's modulus tensors. "
            "Install the project requirements before loading VLM parameters."
        ) from exc
    return torch


def _read_image(image):
    cv2 = _require_cv2()

    if isinstance(image, (str, os.PathLike)):
        image_path = str(image)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return img

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a file path or a numpy array")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    raise ValueError("image must have shape (H, W), (H, W, 3), or (H, W, 4)")


def _point_from_dict(box, names):
    for name in names:
        if name in box:
            return box[name]
    return None


def _parse_box(box, index):
    label = index

    if isinstance(box, dict):
        for key in ("label", "id", "object_id", "index"):
            if key in box:
                label = box[key]
                break

        left_bottom = _point_from_dict(
            box, ("left_bottom", "bottom_left", "lb", "min_point", "p0")
        )
        right_top = _point_from_dict(
            box, ("right_top", "top_right", "rt", "max_point", "p1")
        )
        if left_bottom is None or right_top is None:
            raise ValueError("box dict must contain left_bottom and right_top points")
    else:
        if len(box) == 2:
            left_bottom, right_top = box
        elif len(box) == 4:
            x0, y0, x1, y1 = box
            left_bottom, right_top = (x0, y0), (x1, y1)
        else:
            raise ValueError("box must be [left_bottom, right_top] or [x0, y0, x1, y1]")

    return np.asarray(left_bottom, dtype=float), np.asarray(right_top, dtype=float), label


def _to_image_box(left_bottom, right_top, height, width, coordinate_origin):
    x0, y0 = left_bottom[:2]
    x1, y1 = right_top[:2]

    if coordinate_origin == "bottom_left":
        y0 = height - 1 - y0
        y1 = height - 1 - y1
    elif coordinate_origin != "top_left":
        raise ValueError("coordinate_origin must be 'bottom_left' or 'top_left'")

    x_min = int(round(min(x0, x1)))
    x_max = int(round(max(x0, x1)))
    y_min = int(round(min(y0, y1)))
    y_max = int(round(max(y0, y1)))

    x_min = max(0, min(width - 1, x_min))
    x_max = max(0, min(width - 1, x_max))
    y_min = max(0, min(height - 1, y_min))
    y_max = max(0, min(height - 1, y_max))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("box has zero area after coordinate conversion")

    return x_min, y_min, x_max, y_max


def boxes_from_screen_points(screen_points, labels=None, padding=8):
    """Build 2D boxes from clustered screen-space points.

    Args:
        screen_points: A list of per-object tensors/arrays with shape (N, 2) or
            (N, >=2). The first two coordinates are used as image x/y values.
        labels: Optional labels for each object. Defaults to 0..N-1.
        padding: Extra pixels around each screen-space point extent.

    Returns:
        A list of boxes accepted by generate_bounded_image.
    """
    boxes = []
    labels = list(range(len(screen_points))) if labels is None else labels

    for index, points in enumerate(screen_points):
        if hasattr(points, "detach"):
            points = points.detach().cpu().numpy()
        points = np.asarray(points, dtype=float)

        if points.size == 0:
            continue
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("each screen_points item must have shape (N, 2) or (N, >=2)")

        xy = points[:, :2]
        finite_mask = np.isfinite(xy).all(axis=1)
        xy = xy[finite_mask]
        if xy.shape[0] == 0:
            continue

        lower = np.min(xy, axis=0) - padding
        upper = np.max(xy, axis=0) + padding
        boxes.append(
            {
                "label": labels[index],
                "left_bottom": lower.tolist(),
                "right_top": upper.tolist(),
            }
        )

    return boxes


def generate_bounded_image(
    image,
    boxes,
    output_path=None,
    coordinate_origin="bottom_left",
    thickness=None,
    font_scale=None,
):
    """Draw object AABB boxes and labels on an image for offline VLM use.

    Args:
        image: Input image path or numpy array.
        boxes: Object boxes. Each item can be
            {"label": 0, "left_bottom": [x0, y0], "right_top": [x1, y1]},
            [[x0, y0], [x1, y1]], or [x0, y0, x1, y1].
        output_path: Optional path to save the annotated image.
        coordinate_origin: "bottom_left" for Cartesian coordinates or
            "top_left" for image pixel coordinates.
        thickness: Optional rectangle thickness in pixels.
        font_scale: Optional label font scale.

    Returns:
        The annotated image as a BGR numpy array.
    """
    cv2 = _require_cv2()
    bounded_image = _read_image(image)
    height, width = bounded_image.shape[:2]

    if thickness is None:
        thickness = max(2, round(min(height, width) / 300))
    if font_scale is None:
        font_scale = max(0.5, min(height, width) / 700)

    for index, box in enumerate(boxes):
        left_bottom, right_top, label = _parse_box(box, index)
        x_min, y_min, x_max, y_max = _to_image_box(
            left_bottom, right_top, height, width, coordinate_origin
        )

        color = _BOX_COLORS[index % len(_BOX_COLORS)]
        cv2.rectangle(bounded_image, (x_min, y_min), (x_max, y_max), color, thickness)

        text = str(label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        pad = max(3, thickness * 2)

        text_x = x_min
        text_y = max(text_height + pad * 2, y_min)
        bg_x1 = min(width - 1, text_x + text_width + pad * 2)
        bg_y0 = max(0, text_y - text_height - pad * 2)
        bg_y1 = min(height - 1, text_y + baseline + pad)

        cv2.rectangle(bounded_image, (text_x, bg_y0), (bg_x1, bg_y1), color, -1)
        cv2.putText(
            bounded_image,
            text,
            (text_x + pad, text_y - pad),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), bounded_image):
            raise IOError(f"Could not write bounded image: {output_path}")

    return bounded_image


def call_vlm(input_file_name, output_file_name):
    """Offline VLM workflow placeholder.

    Manually provide generated_data/bounded_image/input_file_name to a VLM and
    save its answer under generated_data/vlm_data/output_file_name.
    """
    return None


def _resolve_vlm_data_path(file_name):
    file_path = Path(file_name)
    if file_path.exists():
        return file_path

    data_dir = Path("generated_data") / "vlm_data"
    if file_path.suffix:
        return data_dir / file_path
    return data_dir / f"{file_name}.json"


def get_initial_params(file_name="generated_data/vlm_data/vlm_params.json"):
    """Load offline VLM material predictions.

    The file can be either a JSON list, e.g. [10000, 20000], or a JSON object:
        {"E": [10000, 20000]}
        {"youngs_modulus": [10000, 20000], "filling_method": ["mcis", "legacy"]}

    Returns:
        A tuple of (E, FT), where E is a torch tensor and FT is a list or None.
    """
    torch = _require_torch()
    vlm_data_path = _resolve_vlm_data_path(file_name)
    with vlm_data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    filling_method = None
    if isinstance(data, list):
        youngs_modulus = data
    elif isinstance(data, dict):
        youngs_modulus = None
        for key in ("E", "youngs_modulus", "initial_params"):
            if key in data:
                youngs_modulus = data[key]
                break
        filling_method = data.get("filling_method", data.get("FT"))
    else:
        raise ValueError("VLM data must be a list or a JSON object")

    if not isinstance(youngs_modulus, list):
        raise TypeError("Young's modulus data must be a list")

    return torch.tensor(youngs_modulus, dtype=torch.float32), filling_method
