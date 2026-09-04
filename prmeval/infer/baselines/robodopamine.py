#!/usr/bin/env python3
# ruff: noqa: E501
"""Robo-Dopamine (GRM) baseline for progress prediction.

Reference: https://github.com/FlagOpen/Robo-Dopamine
Models:
  - GRM-3B: https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-3B
  - GRM-8B: https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview
Supports single-view (same frames for all three camera inputs) and optional goal image.
When no goal/reference is provided, a blank placeholder image is used per upstream recommendation.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..base import Infer

# Known model IDs for config / docs
ROBODOPAMINE_GRM_3B = "tanhuajie2001/Robo-Dopamine-GRM-3B"
ROBODOPAMINE_GRM_8B = "tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview"

SYSTEM_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): <image>
- REFERENCE END — Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = “just beginning”; REFERENCE END = “fully completed.”
   - Visually estimate how far BEFORE and AFTER are along this START→END continuum.
2) Direction:
   - AFTER better than BEFORE → positive score.
   - AFTER worse than BEFORE → negative score.
   - Essentially the same → 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged → 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_pngs_sorted(dir_path: Path) -> list[Path]:
    """Return lexicographically sorted .png files (case-insensitive) under dir_path."""
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".png"])


def get_frame_count(path: Path) -> tuple[str, int]:
    """
    Detects if path is a video file or a directory of images.
    Returns: (source_type, frame_count). source_type is 'dir' or 'video'.
    """
    if path.is_dir():
        files = list_pngs_sorted(path)
        if not files:
            raise RuntimeError(f"No PNG frames found in directory: {path}")
        return "dir", len(files)
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n <= 0:
            raise RuntimeError(f"Invalid frame count from video: {path}")
        return "video", n


def make_sample_indices_by_interval(num_frames: int, interval: int) -> list[int]:
    """
    Generate indices based on a fixed frame interval.
    Always includes the first frame (0) and ensures the very last frame (num_frames-1) is included.
    """
    if num_frames < 1:
        return []

    # Generate base steps: 0, interval, 2*interval, ...
    indices = list(range(0, num_frames, interval))

    # Ensure the absolute last frame is included to cover the full episode
    last_idx = num_frames - 1
    if not indices or indices[-1] != last_idx:
        indices.append(last_idx)

    return indices


def save_frames(src_path: Path, out_dir: Path, indices: list[int], src_type: str) -> None:
    """
    Extracts and saves specific frames from a video or copies them from a directory.
    Output format: frame_{:06d}.png
    """
    ensure_dir(out_dir)

    if src_type == "video":
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src_path}")

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                ok, frame = cap.read()  # Retry once
            if not ok or frame is None:
                print(f"[WARN] Failed to read frame {idx} from {src_path}")
                continue

            out_path = out_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cap.release()

    elif src_type == "dir":
        files = list_pngs_sorted(src_path)
        n = len(files)
        for idx in indices:
            if not (0 <= idx < n):
                continue
            src = files[idx]
            dst = out_dir / f"frame_{idx:06d}.png"
            shutil.copyfile(src, dst)


def build_samples_json(
    run_root: Path, task: str, indices: list[int], ref_end_path: str, mode: str = "incremental"
) -> list[dict[str, Any]]:
    """
    Constructs the list of samples for Transformers inference.

    Args:
        mode: "incremental", "forward", or "backward"
        ref_end_path: Absolute path to the Goal Image.
    """
    timestamp_name = run_root.name
    cache_root = run_root / ".cache"
    items = []

    if len(indices) < 2:
        return items

    # For Forward mode, we need the Start Frame (Index 0)
    # For Incremental, we use Index k
    # For Backward, we use Goal Image

    for k in range(len(indices) - 1):
        af = indices[k + 1]  # The "After" frame (Current Step)

        # Define "Before" images based on mode
        bf_images = []
        bf_id_str = ""

        if mode == "incremental":
            bf = indices[k]
            bf_id_str = f"bf_{bf:06d}"
            bf_images = [
                str(cache_root / "cam_high" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "forward":
            # Compare Start(0) -> Current(af)
            bf = indices[0]
            bf_id_str = f"start_{bf:06d}"
            bf_images = [
                str(cache_root / "cam_high" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "backward":
            # Compare Goal -> Current(af)
            # Use ref_end_path for all 3 views if wrist goals aren't explicit
            bf_id_str = "goal"
            bf_images = [ref_end_path, ref_end_path, ref_end_path]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Define "After" images (Always the current step)
        af_images = [
            str(cache_root / "cam_high" / f"frame_{af:06d}.png"),
            str(cache_root / "cam_left_wrist" / f"frame_{af:06d}.png"),
            str(cache_root / "cam_right_wrist" / f"frame_{af:06d}.png"),
        ]

        items.append(
            {
                "id": f"step-{timestamp_name}-{k:04d}-{bf_id_str}-af_{af:06d}",
                "task": task,
                "image": [
                    str(cache_root / "cam_high" / f"frame_{0:06d}.png"),  # 1. Ref Start
                    ref_end_path,  # 2. Ref End
                    bf_images[0],  # 3. Before High
                    bf_images[1],  # 4. Before Left
                    bf_images[2],  # 5. Before Right
                    af_images[0],  # 6. After High
                    af_images[1],  # 7. After Left
                    af_images[2],  # 8. After Right
                ],
            }
        )
    return items


class GRMInference:
    def __init__(self, model_path: str, max_image_num=8, min_pixels=12544, max_pixels=76800):
        print(f"Loading model from {model_path} ...")

        prompt_image_count = SYSTEM_PROMPT.count("<image>")
        if max_image_num != prompt_image_count:
            raise ValueError(f"RoboDopamine prompt requires exactly {prompt_image_count} images")

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("RoboDopamine requires torch and transformers with multimodal model support") from exc

        self._torch = torch
        self.max_image_num = max_image_num
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        # Decoder-only generation must be left-padded in a mixed-length batch; otherwise the
        # shorter prompts start generation from padding tokens and can silently change scores.
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = max_pixels
            self.processor.image_processor.min_pixels = min_pixels
        self.processor.tokenizer.padding_side = "left"

    @staticmethod
    def _messages(task: str) -> list[dict[str, Any]]:
        prompt_parts = SYSTEM_PROMPT.format(task=task).split("<image>")
        content: list[dict[str, str]] = []
        for index, text in enumerate(prompt_parts):
            content.append({"type": "text", "text": text})
            if index < len(prompt_parts) - 1:
                content.append({"type": "image"})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _load_images(paths: list[str]) -> list[Image.Image]:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return images

    def inference_batch(self, batch_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not batch_data:
            return []

        prompts = []
        images = []
        for item in batch_data:
            item_images = item["image"]
            if len(item_images) != self.max_image_num:
                raise ValueError(f"Expected {self.max_image_num} images per sample, got {len(item_images)}")
            messages = self._messages(str(item["task"]))
            prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)
            # The multimodal processor consumes images for the whole text batch in prompt order.
            # Keeping this flat preserves the one-to-one image-token alignment across samples.
            images.extend(self._load_images(item_images))

        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Match the former vLLM sampling policy. max_new_tokens is deliberate: visual/text prompt
        # tokens must not reduce the response allowance.
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                top_k=50,
                max_new_tokens=1024,
            )
        generated_ids = [sample_ids[input_length:] for sample_ids in output_ids]
        output_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results = []
        for orig_item, output_text in zip(batch_data, output_texts, strict=True):
            res_item = orig_item.copy()
            res_item["pred"] = output_text
            results.append(res_item)
        return results

    def run_pipeline(
        self,
        cam_high_path: str,
        cam_left_path: str,
        cam_right_path: str,
        out_root: str,
        task: str,
        frame_interval: int = 10,
        batch_size: int = 1,
        goal_image: str | None = None,
        eval_mode: str = "incremental",
        visualize: bool = False,
    ) -> str:
        """
        Main entry point.
        Args:
            eval_mode: 'incremental', 'forward', or 'backward'
        """
        valid_modes = ["incremental", "forward", "backward"]
        if eval_mode not in valid_modes:
            raise ValueError(f"Invalid eval_mode '{eval_mode}'. Must be one of {valid_modes}")

        out_root = Path(out_root)
        ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        run_root = out_root / ts
        cache_root = run_root / ".cache"

        cam_dirs = {
            "cam_high": cache_root / "cam_high",
            "cam_left_wrist": cache_root / "cam_left_wrist",
            "cam_right_wrist": cache_root / "cam_right_wrist",
        }
        for d in cam_dirs.values():
            ensure_dir(d)

        paths = [Path(cam_high_path), Path(cam_left_path), Path(cam_right_path)]
        types_counts = [get_frame_count(p) for p in paths]

        counts = [tc[1] for tc in types_counts]
        if len(set(counts)) != 1:
            raise ValueError(f"Frame count mismatch among cameras: {counts}")
        total_frames = counts[0]

        indices = make_sample_indices_by_interval(total_frames, frame_interval)
        print(f"Frames: {total_frames}, Int: {frame_interval}, Mode: {eval_mode}, Indices: {len(indices)}")

        for p, key, (stype, _) in zip(paths, cam_dirs.keys(), types_counts, strict=True):
            save_frames(p, cam_dirs[key], indices, stype)

        # Handle Goal Image / Ref End
        if goal_image is not None and os.path.exists(goal_image):
            ref_end_path = cache_root / "ref_end.png"
            shutil.copy(goal_image, ref_end_path)
            ref_end_path_str = str(ref_end_path)
            print(f"Using Goal Image: {goal_image}")
        else:
            ref_end_path_str = str(cam_dirs["cam_high"] / f"frame_{total_frames - 1:06d}.png")
            print("No Goal Image provided. Using last frame.")

        # Build Samples based on Mode
        samples = build_samples_json(run_root, task, indices, ref_end_path_str, mode=eval_mode)
        json_path = run_root / "sample.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

        print(f"Running inference on {len(samples)} samples...")
        results = []

        for i in tqdm(range(0, len(samples), batch_size)):
            batch = samples[i : i + batch_size]
            results.extend(self.inference_batch(batch))

        # --- Post-Processing Logic based on Mode ---
        prev_prog = 0.0

        for idx, item in enumerate(results):
            raw = item.get("pred", "")
            try:
                val_str = raw.split("<score>")[-1].split("</score>")[0].replace("%", "").strip()
                raw_score = max(-100.0, min(100.0, float(val_str))) / 100.0
            except Exception:
                print(f"[ERR] Failed to parse score for item {idx}: {raw}")
                raw_score = 0.0

            curr_progress = 0.0
            hop = 0.0

            if eval_mode == "incremental":
                # Original Logic: raw_score is incremental change
                if idx == 0:
                    curr_progress = raw_score
                else:
                    if raw_score >= 0:
                        curr_progress = prev_prog + (1 - prev_prog) * raw_score
                    else:
                        curr_progress = prev_prog + prev_prog * raw_score
                hop = raw_score  # In incremental, Model Output IS the Hop signal

            elif eval_mode == "forward":
                # Forward: Model Output IS Progress (Start -> Current)
                curr_progress = raw_score
                # Hop is the change in progress from previous step
                hop = curr_progress - prev_prog

            elif eval_mode == "backward":
                # Backward: Compare Goal -> Current
                # raw_score is likely negative (Current is worse than Goal)
                # Formula: Progress = 1 + ModelOutput
                curr_progress = 1.0 + raw_score
                # Hop is the change in progress
                hop = curr_progress - prev_prog

            # Clamp progress to reasonable bounds [0, 1] for safety?
            # Not strictly enforcing clamping to allow debugging, but typically progress is 0-1.
            # curr_progress = max(0.0, min(1.0, curr_progress))

            item["hop"] = hop
            item["progress"] = curr_progress

            prev_prog = curr_progress

        # Retain the historical artifact name so existing callers of run_pipeline remain compatible.
        result_path = run_root / "pred_vllm.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Done. Results saved to {result_path}")

        # if visualize:
        #     print("Generating visualization video...")
        #     plot_video_reward(run_root)

        return str(run_root)


@register_infer("robodopamine")
class RoboDopamine(Infer):
    capabilities: ClassVar[set[str]] = {"progress"}
    """Robo-Dopamine GRM baseline. Uses single-view frames for all three camera inputs.
    Supports single-view without goal image (blank placeholder used for REFERENCE END).
    """

    def __init__(self, config: InferConfig):
        super().__init__(config)
        if not config.model_path:
            raise ValueError("robodopamine requires infer.model_path")
        options = config.options
        self._initialize(
            model_path=config.model_path,
            frame_interval=int(options.get("frame_interval", 1)),
            batch_size=int(options.get("micro_batch_size", 1)),
            eval_mode=str(options.get("eval_mode", "incremental")),
        )

    def _initialize(
        self,
        model_path: str = ROBODOPAMINE_GRM_3B,
        frame_interval: int = 1,
        batch_size: int = 1,
        eval_mode: str = "incremental",
    ):
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self._grm = GRMInference(model_path=model_path, max_image_num=8)

    def _make_blank_goal_image(self, out_path: Path, height: int = 224, width: int = 224) -> None:
        """Write a neutral gray placeholder image for 'no goal' single-view setting."""
        blank = np.full((height, width, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(blank, cv2.COLOR_RGB2BGR))

    def _goal_image_path(
        self, tmpdir: Path, frames_dir: Path, num_frames: int, reference_video_path: str | None
    ) -> str | None:
        """Resolve goal image path: reference video last frame, or blank placeholder when none."""
        if reference_video_path and os.path.exists(reference_video_path):
            cap = cv2.VideoCapture(reference_video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    goal_path = tmpdir / "goal_from_reference.png"
                    cv2.imwrite(str(goal_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                    return str(goal_path)
        # Single-view without goal: use blank placeholder per upstream recommendation
        blank_path = tmpdir / "blank_goal.png"
        self._make_blank_goal_image(blank_path)
        return str(blank_path)

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray:
        if frames_array is None or frames_array.size == 0:
            return np.array([], dtype=np.float64)

        num_frames = frames_array.shape[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            frames_dir = tmpdir_path / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i in range(num_frames):
                frame = frames_array[i]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(frames_dir / f"frame_{i:06d}.png"),
                    frame_bgr,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
                )

            out_root = tmpdir_path / "out"
            out_root.mkdir(parents=True, exist_ok=True)
            # goal_image = self._goal_image_path(tmpdir_path, frames_dir, num_frames, reference_video_path)
            # run_pipeline: single-view = same dir for all cams; no-goal = blank placeholder
            run_root = self._grm.run_pipeline(
                cam_high_path=str(frames_dir),
                cam_left_path=str(frames_dir),
                cam_right_path=str(frames_dir),
                out_root=str(out_root),
                task=task_description,
                frame_interval=self.frame_interval,
                batch_size=self.batch_size,
                # set to None use last frame
                goal_image=None,
                eval_mode=self.eval_mode,
                visualize=False,
            )

            pred_path = Path(run_root) / "pred_vllm.json"
            with open(pred_path, encoding="utf-8") as f:
                results = json.load(f)

        progress_list = [0.0]
        for item in results:
            p = item.get("progress", 0.0)
            if isinstance(p, str) and p == "Error":
                p = progress_list[-1] if progress_list else 0.0
            progress_list.append(float(p))

        progress_arr = np.clip(np.array(progress_list, dtype=np.float64), 0.0, 1.0)
        if len(progress_arr) < num_frames:
            progress_arr = np.pad(
                progress_arr,
                (0, num_frames - len(progress_arr)),
                mode="edge",
            )
        elif len(progress_arr) > num_frames:
            progress_arr = progress_arr[:num_frames]

        return progress_arr

    def predict(self, samples: list[EvaluationSample]) -> list[Prediction]:
        result = []
        for sample in samples:
            if not isinstance(sample, ProgressSample):
                raise TypeError(f"{self.config.name} only supports progress samples")
            reference_path = sample.trajectory.metadata.get("reference_video_path")
            values = np.asarray(
                self.compute_progress(
                    np.asarray(sample.trajectory.frames),
                    sample.trajectory.task,
                    str(reference_path) if reference_path else None,
                ),
                dtype=float,
            ).reshape(-1)
            expected = len(sample.trajectory.frames)
            if len(values) != expected:
                raise ValueError(f"Progress length mismatch: expected {expected}, got {len(values)}")
            if not np.isfinite(values).all():
                raise ValueError("Progress values must be finite")
            if ((values < 0) | (values > 1)).any():
                raise ValueError("Progress values must be in [0, 1]")
            result.append(
                ProgressPrediction(
                    sample_id=sample.sample_id,
                    progress=values.tolist(),
                    model=self.config.model_id or self.config.model_path or self.config.name,
                    model_version=self.config.model_version,
                )
            )
        return result
