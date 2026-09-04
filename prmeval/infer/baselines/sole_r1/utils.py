# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import base64
import io
import json
import logging
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image

from .constants import QUESTION_TEMPLATE, SYSTEM_PROMPT_TEMPLATE

matplotlib.use("Agg")  # Set non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
NUMPY_EXTENSIONS = {".npy", ".npz"}

FRONT_ALIASES = {"agentview", "front", "rgb", "external"}
WRIST_ALIASES = {"wristview", "wrist", "hand"}


def resize_with_padding(img: np.ndarray, size: int = 384) -> np.ndarray:
    """Aspect ratio preserving image resize with padding.

    Args:
        img: Input image.
        size: Desired output size (size x size).

    Returns:
        Resized image with padding.
    """
    h, w = img.shape[:2]
    # Determine scaling factor so max dimension == size
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Resize with preserved aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Create a black canvas 384x384
    output = np.zeros((size, size, 3), dtype=np.uint8)
    # Center the resized image on the canvas
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    output[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return output


def create_composite_frame(
    first_frame_wrist_view: np.ndarray,
    first_frame_external_view: np.ndarray,
    frame0_wrist_view: np.ndarray,
    frame0_external_view: np.ndarray,
    frame1_wrist_view: np.ndarray,
    frame1_external_view: np.ndarray,
    from_zero: bool = False,
    external_only: bool = False,
    size: int = 384,
    padding: int = 5,
) -> np.ndarray:
    """Create composite image frame for VLM input."""
    first_imgs = [first_frame_wrist_view, first_frame_external_view]
    imgs0 = [frame0_wrist_view, frame0_external_view]
    imgs2 = [frame1_wrist_view, frame1_external_view]

    for i in range(len(first_imgs)):
        first_imgs[i] = resize_with_padding(first_imgs[i], size)
        imgs0[i] = resize_with_padding(imgs0[i], size)
        imgs2[i] = resize_with_padding(imgs2[i], size)

    col_pad = np.zeros((size, padding, 3), dtype=np.uint8)
    if not from_zero:
        bottom_row = np.hstack([first_imgs[0], col_pad, imgs0[0], col_pad, imgs2[0]])
        top_row = np.hstack(
            [
                first_imgs[1],
                col_pad,
                imgs0[1],
                col_pad,
                imgs2[1],
            ]
        )
    else:
        bottom_row = np.hstack([first_imgs[0], col_pad, imgs2[0]])
        top_row = np.hstack([first_imgs[1], col_pad, imgs2[1]])
    # Create horizontal (row) padding between timesteps
    # row_pad = np.zeros((padding, size, 3), dtype=np.uint8)
    full_width = top_row.shape[1]
    row_pad = np.zeros((padding, full_width, 3), dtype=np.uint8)

    if external_only:
        return top_row
    # Now stack horizontally instead of vertically
    return np.vstack([top_row, row_pad, bottom_row])


def image_to_base64(img: np.ndarray, quality: int = 90) -> str:
    """Convert a numpy array image to base64 encoded string."""
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_images(images: list[list[np.ndarray]], quality: int = 90) -> list[list[str]]:
    """Encode a list of images to base64 strings."""
    encoded_images = []
    for ep_images in images:
        encoded_ep = []
        for img in ep_images:
            encoded_img = image_to_base64(img, quality=quality)
            encoded_ep.append(encoded_img)
        encoded_images.append(encoded_ep)
    return encoded_images


def decode_compressed_image(png_bytes: np.uint8) -> np.ndarray:
    """Decode PNG bytes into a numpy array image."""
    stream = io.BytesIO(png_bytes.tobytes())
    img = Image.open(stream)
    # Ensure that image data is fully loaded.
    img.load()
    return np.array(img)


def count_images(images: list[list]) -> tuple[int, list[int]]:
    """Count images and validate consistent episode counts and lengths."""
    num_episodes = len(images)
    episode_lengths = [len(epi) for epi in images]
    return num_episodes, episode_lengths


def count_and_validate_images(front_images: list[list], wrist_images: list[list]) -> tuple[int, list[int]]:
    """Count images and validate consistent episode counts and lengths."""
    assert len(front_images) == len(wrist_images), f"Length mismatch: {len(front_images)} vs {len(wrist_images)}"
    num_episodes = len(front_images)
    episode_lengths = []
    for epi_idx in range(num_episodes):
        front_imgs = front_images[epi_idx]
        wrist_imgs = wrist_images[epi_idx]
        assert len(front_imgs) == len(wrist_imgs), (
            f"Length mismatch in episode {epi_idx}: {len(front_imgs)} vs {len(wrist_imgs)}"
        )
        episode_lengths.append(len(front_imgs))
    return num_episodes, episode_lengths


def process_images(images: list[list]) -> list[list[np.ndarray]]:
    """Decode and validate images from bytes to numpy arrays.

    Args:
        images: List of episodes, each containing a list of image bytes.

    Returns:
        Processed images as numpy array in HWC uint8 format.
    """
    processed_images = []
    for ep_images in images:
        processed_ep = []
        for img_bytes in ep_images:
            img = decode_and_validate_image(img_bytes)
            processed_ep.append(img)
        processed_images.append(processed_ep)
    return processed_images


def decode_and_validate_image(image: np.ndarray | torch.Tensor | bytes) -> np.ndarray:
    """Decode and validate image input into a numpy array in HWC uint8 format.

    Case 1: image is bytes (compressed PNG/JPEG).
    Case 2: image is torch.Tensor (uint8 or float32).
    Case 3: image is uint8 np.ndarray of bytes (compressed PNG/JPEG).
    Case 4: image is np.ndarray in CHW or HWC format, uint8 or float32.
    """
    if isinstance(image, bytes):
        # Decode image bytes.
        image = decode_compressed_image(np.frombuffer(image, dtype=np.uint8))
    if isinstance(image, torch.Tensor):
        # Torch into numpy.
        image = image.detach().cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 1:
            # Decode image bytes.
            image = decode_compressed_image(image)
        if image.ndim == 3 and image.shape[0] in (1, 3):
            # CHW to HWC.
            image = np.moveaxis(image, 0, -1)
        if image.dtype != np.uint8:
            # E.g., float32 to uint8.
            if image.max() <= 1.0:
                # [0, 1] to [0, 255].
                # Note that this could fail with black images.
                # So better send uint8 images.
                image = (image * 255).clip(0, 255)
            image = image.astype(np.uint8)

    assert image.ndim == 3, f"Image must be HWC format, got shape {image.shape}."
    assert image.shape[2] in [
        1,
        3,
    ], f"Image must have 1 or 3 channels, got shape {image.shape}."
    assert image.dtype == np.uint8, f"Image must be uint8 type, got dtype {image.dtype}. This is an internal error."

    return image


def prepare_vlm_images(
    front_images: list[list[np.ndarray]],
    wrist_images: list[list[np.ndarray]],
    num_episodes: int,
    episode_lengths: list[int],
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    factor: int = 28,
    from_zero: bool = False,
    external_only: bool = False,
) -> list[list[Image.Image]]:
    """Prepare composite VLM images for each episode and timestep."""
    from qwen_vl_utils import smart_resize

    vlm_images = []
    for ep_idx in range(num_episodes):
        vlm_images_epi = []
        for t in range(1, episode_lengths[ep_idx]):
            composite_frame = create_composite_frame(
                first_frame_wrist_view=wrist_images[ep_idx][0],
                first_frame_external_view=front_images[ep_idx][0],
                frame0_wrist_view=wrist_images[ep_idx][t - 1],
                frame0_external_view=front_images[ep_idx][t - 1],
                frame1_wrist_view=wrist_images[ep_idx][t],
                frame1_external_view=front_images[ep_idx][t],
                from_zero=from_zero,
                external_only=external_only,
            )
            composite_frame = Image.fromarray(composite_frame)

            width, height = composite_frame.size
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            composite_frame = composite_frame.resize((resized_width, resized_height))

            vlm_images_epi.append(composite_frame)
        vlm_images.append(vlm_images_epi)
    return vlm_images


def make_conversation_image(
    question: str,
    system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE,
    question_template: str = QUESTION_TEMPLATE,
) -> list[dict]:
    """Create conversation format input for VLMs with image and text."""
    return [
        {
            "role": "system",
            "content": system_prompt_template.format(question=question),
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": question_template.format(question=question),
                },
            ],
        },
    ]


def assemble_output_batch(outputs: list, indices: list[int], video_count: int) -> list:
    """Assemble outputs into a batch aligned with input video indices.

    Some values can be None if the episode is shorter than others.
    """
    output_batch = [None] * video_count
    for out, idx in zip(outputs, indices, strict=False):
        output_batch[idx] = out
    return output_batch


def get_output_across_videos(
    video_count: int,
    text_input_list_batch: list[list],
    text_output_list_batch: list[list],
    answer_list_batch: list[list],
) -> tuple[list[list], list[list], list[list]]:
    """Organize outputs on a per-episode basis.

    This function handles cases where some episodes are shorter than others.
    This manifests as None values in the input lists for timesteps beyond the episode length.
    We double-check that once we see a None the episode is done and all subsequent values are also None.
    We also double-check that the three input lists are aligned in terms of None values.
    """
    text_output_list_list = []
    text_input_list_list = []
    answer_list_list = []
    for video_idx in range(video_count):
        text_inputs, text_outputs, answers = [], [], []
        seen_none = False
        for ti, to, an in zip(
            [x[video_idx] for x in text_input_list_batch],
            [x[video_idx] for x in text_output_list_batch],
            [x[video_idx] for x in answer_list_batch],
            strict=False,
        ):
            none_mask = (ti is None, to is None, an is None)
            assert all(none_mask) or not any(none_mask), f"Misaligned Nones at video {video_idx}: {none_mask}"

            is_none = ti is None
            if seen_none and not is_none:
                raise AssertionError(
                    f"Video {video_idx}: Non-None value appeared after None. "
                    f"Once None appears, all subsequent values must be None."
                )

            if is_none:
                seen_none = True
            else:
                # Only append if not None
                text_inputs.append(ti)
                text_outputs.append(to)
                answers.append(an)
        text_output_list_list.append(text_outputs)
        text_input_list_list.append(text_inputs)
        answer_list_list.append(answers)
    return (
        text_output_list_list,
        text_input_list_list,
        answer_list_list,
    )


def get_answer_from_completion(completion):
    """Extract task progress answer from model completion string."""
    answer = ""
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, completion, re.DOTALL)
    if not answer_match:
        answer_pattern = r"<answer>(.*?)</answer"
        answer_match = re.search(answer_pattern, completion, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        answer = answer.replace("%", "")
        try:
            answer_int = int(answer)
            if answer_int < -100:
                answer = "-100"
            elif answer_int > 100:
                answer = "100"
        except Exception:
            answer_int = 0

    return answer


def create_video_with_plot(
    output_video_path,
    frame_list,
    frame_desription_list,
    data_points,
    data_points_env=None,
    fps_=2,
    wrap_width=26,
    font_scale=0.5,
):
    """
    Creates a video combining robot frames, a text description panel, and a
    live-updating reward plot side by side.

    Args:
        output_video_path: Path to the output video file (.mp4 or .webm).
        frame_list: List of frames (PIL Images or numpy arrays) per step.
        frame_desription_list: List of text descriptions for each frame.
        data_points: List of predicted task progress values (0-100) per step.
        data_points_env: Optional list of environment reward values per step.
        fps_: Frames per second for the output video.
        wrap_width: Character width for text wrapping in the description panel.
        font_scale: Font scale for the description text overlay.
    """
    plt.rcParams.update({"font.size": 6})

    first_frame = frame_list[0]
    if isinstance(first_frame, Image.Image):
        first_frame = np.array(first_frame.convert("RGB"))
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)

    frame_width = first_frame.shape[1]
    frame_height = first_frame.shape[0]
    if frame_height < 384:
        first_frame = cv2.copyMakeBorder(
            first_frame,
            (384 - frame_height) // 2,
            (384 - frame_height) // 2,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
    if frame_width < 384:
        first_frame = cv2.copyMakeBorder(
            first_frame,
            0,
            0,
            (384 - frame_width) // 2,
            (384 - frame_width) // 2,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
    frame_width = first_frame.shape[1]
    frame_height = first_frame.shape[0]

    if frame_width == 2 * frame_height:
        output_width = int(2 * frame_width)
        denom_ = 2
    else:
        output_width = int(3 * frame_width)
        denom_ = 1

    output_height = frame_height

    if output_video_path.endswith(".webm"):
        fourcc = cv2.VideoWriter_fourcc("V", "P", "9", "0")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps_, (output_width, output_height))

    fig, ax = plt.subplots(dpi=200)
    canvas = FigureCanvas(fig)
    xdata, ydata, ydata2 = [], [], []
    ax.set_xlim(0, len(data_points) - 1)
    ax.set_title("")
    env_points = data_points_env if data_points_env is not None else []
    ax.set_ylim(
        min(data_points + env_points),
        max([100, *data_points, *env_points]),
    )
    ax.set_ylabel("Task progress (%)")
    ax.set_xlabel("Step number")

    (ln,) = ax.plot([], [], "r", label="Predicted task progress")
    ln.set_color("darkblue")
    ln.set_markerfacecolor("darkblue")
    ln.set_markersize(5)

    if data_points_env is not None:
        (ln2,) = ax.plot([], [], "b", label="Environment reward")
        ln2.set_color("darkred")
        ln2.set_markerfacecolor("darkred")
        ln2.set_markersize(5)
    else:
        ln2 = None

    ax.legend(fontsize="x-small")

    for frame_number in range(len(data_points)):
        frame = frame_list[frame_number]
        if isinstance(frame, Image.Image):
            frame = np.array(frame.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        if frame_height < 384:
            frame = cv2.copyMakeBorder(
                frame,
                (384 - frame_height) // 2,
                (384 - frame_height) // 2,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )
        if frame_width < 384:
            frame = cv2.copyMakeBorder(
                frame,
                0,
                0,
                (384 - frame_width) // 2,
                (384 - frame_width) // 2,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        fig.set_size_inches((frame_width // denom_) / fig.dpi, frame_height / fig.dpi)
        fig.tight_layout(pad=0.4)

        xdata.append(frame_number)
        ydata.append(data_points[frame_number])
        ln.set_data(xdata, ydata)
        if ln2 is not None:
            ydata2.append(data_points_env[frame_number])
            ln2.set_data(xdata, ydata2)
        ax.draw_artist(ax.patch)
        ax.draw_artist(ln)
        if ln2 is not None:
            ax.draw_artist(ln2)
        canvas.draw()
        plot_image = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
        plot_image = plot_image.reshape((*canvas.get_width_height()[::-1], 4))

        plot_image_resized = cv2.resize(
            plot_image,
            (frame_width // denom_, frame_height),
            interpolation=cv2.INTER_LINEAR,
        )
        plot_image_rgb = cv2.cvtColor(plot_image_resized, cv2.COLOR_RGBA2RGB)

        white_box_width = frame_width // denom_
        white_box = np.ones((frame_height, white_box_width, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)
        thickness = 1
        line_type = 2

        text = frame_desription_list[frame_number]
        language_description_split = text.split(" ")
        language_description_wrapped = ""
        line_i_len = 0
        for i in range(len(language_description_split)):
            if line_i_len < wrap_width:
                language_description_wrapped += language_description_split[i] + " "
                line_i_len += len(language_description_split[i])
            else:
                language_description_wrapped += "\n" + language_description_split[i] + " "
                line_i_len = len(language_description_split[i])
        text = language_description_wrapped
        text = text.replace("</think><answer>", "</think>\n<answer>")

        lines = text.split("\n")
        start_y = 20
        for i, line in enumerate(lines):
            y = start_y + (i * 20)
            cv2.putText(
                white_box,
                line,
                (10, y),
                font,
                font_scale,
                font_color,
                thickness,
                line_type,
            )

        combined_frame = np.hstack((frame, white_box, plot_image_rgb))
        out.write(combined_frame)

    out.release()


def _numeric_sort_key(p: Path) -> tuple:
    """Sort key: extract all digit groups from the stem, then fall back to name."""
    nums = re.findall(r"\d+", p.stem)
    return tuple(int(n) for n in nums) if nums else (p.name,)


def load_frames_from_video(path: Path, stride: int, start: int, end: int) -> list[np.ndarray]:
    """Load frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if end >= 0 and idx >= end:
            break
        if idx >= start and (idx - start) % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from video: {path}")
    return frames


def load_frames_from_directory(path: Path, stride: int, start: int, end: int) -> list[np.ndarray]:
    """Load images from a directory, sorted numerically by filename."""
    from PIL import Image

    candidates = sorted(
        [f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
        key=_numeric_sort_key,
    )
    if not candidates:
        raise ValueError(f"No images found in directory: {path}")

    frames = []
    for idx, img_path in enumerate(candidates):
        if end >= 0 and idx >= end:
            break
        if idx >= start and (idx - start) % stride == 0:
            img = Image.open(img_path).convert("RGB")
            frames.append(np.array(img))

    if not frames:
        raise ValueError(f"No frames selected from directory: {path}")
    return frames


def load_frames_from_numpy(path: Path, npz_key: str, stride: int, start: int, end: int) -> list[np.ndarray]:
    """Load frames from a .npy or .npz file. Expected shape: (T, H, W, C)."""
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    else:
        data = np.load(str(path))
        if npz_key not in data:
            available = list(data.keys())
            raise ValueError(f"Key '{npz_key}' not found in {path}. Available: {available}")
        arr = data[npz_key]

    if arr.ndim != 4:
        raise ValueError(f"Expected shape (T,H,W,C), got {arr.shape} from {path}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    indices = range(start, (len(arr) if end < 0 else min(end, len(arr))), stride)
    frames = [arr[i] for i in indices]
    if not frames:
        raise ValueError(f"No frames selected from numpy file: {path}")
    return frames


def estimate_source_fps(path: str) -> float | None:
    """
    Read the native frame rate from a video file via OpenCV metadata.
    Returns None for image directories, .npy/.npz, and .h5/.hdf5 inputs.
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return float(fps)
    return None


def compute_stride(source_fps: float, target_fps: float) -> int:
    """Return the frame stride needed to sample a source at approximately target_fps."""
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    return max(1, round(source_fps / target_fps))


def load_frames(
    path: str,
    stride: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    npz_key: str = "frames",
    hdf5_key: str = "frames",
) -> list[np.ndarray]:
    """
    Unified frame loader. Returns a list of RGB uint8 HWC numpy arrays.

    Supported sources:
        - Video files:     .mp4 .avi .mov .mkv .webm .m4v
        - Image directory: directory containing .jpg/.png/.bmp/.tiff/.webp files
        - NumPy archive:   .npy  (T,H,W,C)  or  .npz  (key=npz_key)
        - HDF5 file:       .h5 / .hdf5  (dataset key=hdf5_key)
    """
    p = Path(path)

    if p.is_dir():
        return load_frames_from_directory(p, stride, start_frame, end_frame)

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    ext = p.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return load_frames_from_video(p, stride, start_frame, end_frame)
    if ext in NUMPY_EXTENSIONS:
        return load_frames_from_numpy(p, npz_key, stride, start_frame, end_frame)

    raise ValueError(
        f"Unsupported file type '{ext}' for path: {path}\n"
        f"Supported: video {sorted(VIDEO_EXTENSIONS)}, "
        f"numpy {sorted(NUMPY_EXTENSIONS)}, "
        f"or a directory of images."
    )


def _find_subdir(parent: Path, aliases: set[str]) -> Path | None:
    """Return the first subdirectory of parent whose lower-case name is in aliases."""
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() in aliases:
            return child
    return None


def resolve_views(front_arg: str, wrist_arg: str | None) -> tuple[str, str | None]:
    """
    Resolve front and (optionally) wrist paths.

    If front_arg is a directory containing canonical multi-view subdirectories
    (e.g. agentview/ + wristview/), both are auto-detected and wrist_arg is
    ignored (with a warning if it was also provided).

    Returns (front_path, wrist_path_or_None).
    """
    front_path = Path(front_arg)
    if front_path.is_dir():
        front_sub = _find_subdir(front_path, FRONT_ALIASES)
        wrist_sub = _find_subdir(front_path, WRIST_ALIASES)

        if front_sub is not None and wrist_sub is not None:
            if wrist_arg is not None:
                logging.warning(
                    "Auto-detected multi-view subdirectories inside '%s'; ignoring --wrist %s",
                    front_arg,
                    wrist_arg,
                )
            logging.info("Auto-detected views: front='%s', wrist='%s'", front_sub, wrist_sub)
            return str(front_sub), str(wrist_sub)

    return front_arg, wrist_arg


def build_payload(
    tasks: list[str],
    front_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray] | None,
    from_zero: bool = False,
    temperature: float = 1.0,
) -> dict:
    """
    Build the request payload for the reward server.

    The server expects list[list[image]] (episodes x timesteps).
    We wrap the single episode in an outer list.
    """
    external_only = wrist_frames is None
    payload = {
        "tasks": tasks,
        "front_images": front_frames,
        "wrist_images": wrist_frames if wrist_frames is not None else front_frames,
        "temperature": temperature,
        "from_zero": from_zero,
        "external_only": external_only,
    }
    return payload


def print_results(valid_answers: list[int]) -> None:
    """Print a per-timestep progress table to stdout."""
    col_w = max(len(str(len(valid_answers))), 8)
    header = f"{'Timestep':>{col_w}}   {'Progress':>8}"
    print(header)
    print("-" * len(header))
    for t, v in enumerate(valid_answers):
        print(f"{t:>{col_w}}   {v:>7} %")
    print()
    final = valid_answers[-1]
    print(f"Final reward (last timestep): {final} %")


def save_visualization(
    video_output: str,
    front_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray] | None,
    valid_answers: list[int],
    text_outputs: list[str],
    fps: float,
) -> None:
    """
    Save a side-by-side visualization video:
      [agentview (+ optional wristview stacked vertically)] | [text output] | [reward plot]

    Frame alignment:
      valid_answers[0] == 0 by convention (first frame baseline).
      valid_answers[1..T] are predictions for input frames 0..T-1.
      text_outputs[0..T-1] are the model completions for frames 0..T-1.
      We skip index 0 of valid_answers, aligning predictions 1..T with frames 0..T-1.
    """
    T = len(front_frames)
    # Align predictions: drop the leading 0 baseline entry if present
    answers = list(valid_answers)
    if len(answers) == T + 1:
        answers = answers[1:]
    # Pad or truncate to match frame count
    n = min(T, len(answers))
    frame_list = []
    for i in range(n):
        f = front_frames[i]
        if wrist_frames is not None:
            w = wrist_frames[i]
            # Resize wrist to same height as front before stacking
            if f.shape[0] != w.shape[0]:
                scale = f.shape[0] / w.shape[0]
                w = cv2.resize(w, (int(w.shape[1] * scale), f.shape[0]), interpolation=cv2.INTER_AREA)
            f = np.hstack([f, w])
        frame_list.append(f)

    # Pad text_outputs if shorter
    texts = list(text_outputs) if text_outputs is not None else []
    texts = texts[:n]
    while len(texts) < n:
        texts.append("")

    data_points = answers[:n]

    logging.info("Saving visualization video to: %s", video_output)
    create_video_with_plot(
        output_video_path=video_output,
        frame_list=frame_list,
        frame_desription_list=texts,
        data_points=data_points,
        data_points_env=None,
        fps_=fps,
    )
    logging.info("Visualization video saved.")


def save_results(
    output_file: str,
    task: str,
    front_path: str,
    wrist_path: str | None,
    from_zero: bool,
    external_only: bool,
    data: dict,
) -> None:
    """Save inference results to a JSON file."""
    result = {
        "task": task,
        "front_path": front_path,
        "wrist_path": wrist_path,
        "from_zero": from_zero,
        "external_only": external_only,
        "valid_answers": data["valid_answers"][0].tolist(),
        "text_outputs": data["text_outputs"][0],
        "text_inputs": data["text_inputs"][0],
    }
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    logging.info("Results saved to %s", output_file)
