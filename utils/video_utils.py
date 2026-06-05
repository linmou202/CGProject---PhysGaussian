from pathlib import Path

import numpy as np


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "get_video_frames requires opencv-python. "
            "Install the project requirements before decoding videos."
        ) from exc
    return cv2


def _resolve_video_path(video_name):
    video_path = Path(video_name)
    if video_path.exists():
        return video_path

    candidates = []
    if video_path.suffix:
        candidates.extend(
            [
                Path("generated_data") / "video" / video_path,
                Path("generated_data") / "vlm_data" / video_path,
            ]
        )
    else:
        for suffix in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            candidates.extend(
                [
                    Path("generated_data") / "video" / f"{video_name}{suffix}",
                    Path("generated_data") / "vlm_data" / f"{video_name}{suffix}",
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    search_roots = [
        Path("generated_data") / "video",
        Path("generated_data") / "vlm_data",
    ]
    searched = [str(path) for path in candidates]
    searched.extend(str(root) for root in search_roots)
    raise FileNotFoundError(
        f"Could not find video '{video_name}'. Searched: {', '.join(searched)}"
    )


def _sample_frame_indices(total_frames, source_fps, target_fps):
    if total_frames <= 0:
        return None
    if target_fps is None or target_fps <= 0 or source_fps <= 0:
        return None

    duration = total_frames / source_fps
    target_count = max(1, int(np.floor(duration * target_fps)))
    indices = np.floor(np.arange(target_count) * source_fps / target_fps).astype(int)
    return np.clip(indices, 0, total_frames - 1)


def get_video_frames(
    video_name,
    output_width=None,
    output_height=None,
    frame_per_sec=None,
    rgb=True,
    as_tensor=False,
):
    """Decode a video into a frame sequence.

    Args:
        video_name: Full video path, or a file name under generated_data/video or
            generated_data/vlm_data.
        output_width: Optional resized frame width.
        output_height: Optional resized frame height.
        frame_per_sec: Optional target FPS. If omitted, all source frames are read.
        rgb: Return RGB frames when True, BGR frames when False.
        as_tensor: Return a torch tensor with shape (T, H, W, C) when True.

    Returns:
        A numpy array or torch tensor with shape (frame_num, height, width, 3).
    """
    cv2 = _require_cv2()
    video_path = _resolve_video_path(video_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_indices = _sample_frame_indices(total_frames, source_fps, frame_per_sec)
    sample_set = None if sample_indices is None else set(sample_indices.tolist())

    frames = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if sample_set is None or frame_index in sample_set:
            if output_width is not None or output_height is not None:
                height, width = frame.shape[:2]
                new_width = int(output_width) if output_width is not None else width
                new_height = int(output_height) if output_height is not None else height
                if new_width <= 0 or new_height <= 0:
                    raise ValueError("output_width and output_height must be positive")
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frame_index += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from video: {video_path}")

    frame_array = np.stack(frames, axis=0)
    if not as_tensor:
        return frame_array

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "as_tensor=True requires torch. Install the project requirements first."
        ) from exc
    return torch.from_numpy(frame_array)


def get_videos_frame(*args, **kwargs):
    """Backward-compatible alias for get_video_frames."""
    return get_video_frames(*args, **kwargs)
