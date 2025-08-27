#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indoor-first panorama segmentation for VR assets.

Pipeline:
- (Optionally) ask GPT to propose candidate asset labels from a whitelisted vocabulary.
- Use GroundingDINO for text-conditioned object detection.
- Use SAM to segment final masks for kept boxes.
- Build a window/door "exclusion mask" and drop candidates that significantly overlap it.
  (This aims to avoid outdoor objects while keeping indoor items near windows.)
- Only output/save labels that actually produced at least one final mask.
"""

import os, cv2, json, base64, argparse, requests
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import torch

# ==============================
# Optional dependency guards
# ==============================
try:
    # SAM for segmentation
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except Exception as e:
    print("⚠️ segment_anything import failed:", repr(e))
    SAM_AVAILABLE = False

TRANSFORMERS_AVAILABLE = True
try:
    # Use the specific GroundingDINO class to avoid Auto* → generation → accelerate chain
    from transformers import GroundingDinoForObjectDetection
except Exception as e:
    print("⚠️ transformers import failed:", repr(e))
    TRANSFORMERS_AVAILABLE = False


# ==============================
# Label control (vocab/synonyms/blacklist)
# ==============================
ALLOWED_LABELS = {
    # seating
    "sofa", "couch", "armchair", "chair", "stool", "bench",
    # tables
    "table", "coffee table", "side table",
    # decor / appliances
    "plant", "potted plant", "lamp", "floor lamp",
    "cabinet", "shelf", "bookshelf", "tv", "television",
}
BLACKLIST_LABELS = {
    # background-like categories we don't segment as assets
    "door", "window", "wall", "floor", "ceiling", "sky", "balcony", "frame", "stairs",
    # too amorphous for stable interaction
    "shadow", "light", "reflection", "curtain"
}
SYNONYMS = {
    "couch": "sofa",
    "television": "tv",
    "potted plant": "plant",
    "curtains": "curtain",
}
DEFAULT_FALLBACK = ["sofa", "chair", "table", "plant", "lamp", "bench"]

# Default thresholds (can be overridden via CLI)
MIN_PROMPTS = 1
MAX_PROMPTS = 3
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
MIN_AREA_RATIO = 0.01      # keep boxes covering at least 1% of the image
MAX_AREA_RATIO = 0.30      # and at most 30% of the image

# Priority order to ensure at least one object remains
PREFERRED_ORDER = [
    "sofa", "couch", "armchair", "chair", "bench", "stool",
    "table", "coffee table", "side table",
    "plant", "lamp", "floor lamp", "tv", "cabinet", "bookshelf"
]

# Context exclusions (windows/doors) → mask-first policy
CONTEXT_EXCLUDE_LABELS = ["window", "door"]
EXCLUSION_BOX_TH = 0.25
EXCLUSION_TEXT_TH = 0.25
EXCLUSION_PAD_RATIO = 0.01     # padding for fallback box-only exclusions
EXCLUSION_CENTER_RULE = True   # additional conservative rule if mask is unavailable

# Mask-based indoor/outdoor split (preferred)
EXCLUSION_USE_MASK = True
EXCLUSION_MASK_DILATE_PX = 12      # dilation (px) applied to union of window/door masks
EXCLUSION_OVERLAP_DROP = 0.40      # drop if (exclusion-mask coverage within candidate box) ≥ this

# Cross-label NMS to avoid duplicates across semantically-close labels
CROSS_LABEL_NMS_IOU = 0.60


# ==============================
# OpenAI Vision (GPT-4o family)
# ==============================
def get_asset_prompts_from_gpt(image_path: str, api_key: str, override_labels: Optional[List[str]] = None) -> List[str]:
    """Return asset-like nouns from a constrained vocabulary via GPT-4o, or use overrides."""
    if override_labels:
        print(f"🔖 Using overridden labels: {override_labels}")
        return [s.strip().lower() for s in override_labels if s.strip()]
    print("🧠 Contacting GPT-4o to analyze the panorama and identify assets...")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ ERROR: OpenAI API key is not provided or is a placeholder.")
        return []

    def encode_image_to_base64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    allowed_str = ", ".join(sorted(ALLOWED_LABELS))
    banned_str = ", ".join(sorted(BLACKLIST_LABELS))
    b64 = encode_image_to_base64(image_path)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt_text = (
        "Analyze this panoramic interior image. Select up to 5 distinct, opaque, movable, "
        "well-bounded objects suitable for interaction in VR. "
        f"Choose ONLY from this vocabulary: {allowed_str}. "
        f"DO NOT include any of: {banned_str}. "
        "Return ONLY a comma-separated list of simple, lowercase, singular nouns (from the vocabulary)."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ],
        }],
        "max_tokens": 100,
    }

    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        prompts = [p.strip().lower() for p in content.split(",") if p.strip()]
        print(f"✅ GPT raw labels: {prompts}")
        return prompts
    except Exception as e:
        err_body = ""
        if hasattr(e, "response") and getattr(e, "response") is not None:
            try:
                err_body = e.response.text
            except Exception:
                pass
        print(f"❌ ERROR: GPT request failed: {e} {(' :: ' + err_body) if err_body else ''}")
        return []


# ==============================
# Label post-processing
# ==============================
def normalize_and_filter_labels(labels: List[str]) -> List[str]:
    """Normalize synonyms, apply whitelist/blacklist, and deduplicate."""
    out: List[str] = []
    for lab in labels:
        lab = SYNONYMS.get(lab.strip().lower(), lab.strip().lower())
        if lab in BLACKLIST_LABELS:
            continue
        if (lab in ALLOWED_LABELS) or (lab in SYNONYMS.values()):
            if lab not in out:
                out.append(lab)
    return out


# ==============================
# Processor loader + post-process compatibility
# ==============================
def _load_grounding_processor(model_id: str):
    """Robust processor loader across Transformers versions."""
    e1 = e2 = e3 = None
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(model_id)
    except Exception as ex:
        e1 = ex
    try:
        from transformers import GroundingDinoProcessor
        return GroundingDinoProcessor.from_pretrained(model_id)
    except Exception as ex:
        e2 = ex
    try:
        from transformers.models.auto.processing_auto import AutoProcessor as SubAuto
        return SubAuto.from_pretrained(model_id)
    except Exception as ex:
        e3 = ex
    raise RuntimeError(
        f"Failed to load processor for {model_id}:\n"
        f"- top-level AutoProcessor error: {e1}\n"
        f"- GroundingDinoProcessor error: {e2}\n"
        f"- submodule AutoProcessor error: {e3}"
    )


def _post_process_compat(processor, outputs, input_ids, target_sizes,
                         box_threshold: float, text_threshold: float):
    """Call the appropriate post-processing API across Transformers versions."""
    try:
        res = processor.post_process_grounded_object_detection(
            outputs=outputs, input_ids=input_ids,
            threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=target_sizes
        ); return res[0]
    except TypeError:
        pass
    try:
        res = processor.post_process_grounded_object_detection(
            outputs=outputs, input_ids=input_ids,
            box_threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=target_sizes
        ); return res[0]
    except TypeError:
        pass
    res = processor.post_process_grounded_object_detection(
        outputs=outputs, input_ids=input_ids,
        score_threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=target_sizes
    ); return res[0]


# ==============================
# Geometry helpers
# ==============================
def iou_xyxy(a: List[float], b: List[float]) -> float:
    """IoU for [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    union = aw * ah + bw * bh - inter + 1e-9
    return inter / union

def center_in_box(box: List[float], region: List[float]) -> bool:
    """Return True if the center of `box` lies inside `region`."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    rx1, ry1, rx2, ry2 = region
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

def expand_box(box: List[float], W: int, H: int, pad: float) -> List[float]:
    """Expand the box by a fraction of min(W,H)."""
    x1, y1, x2, y2 = box
    dx = pad * min(W, H)
    dy = dx
    return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(W - 1.0, x2 + dx), min(H - 1.0, y2 + dy)]


# ==============================
# GroundingDINO wrapper (HF)
# ==============================
class HFGroundedDINO:
    """Thin wrapper around GroundingDinoForObjectDetection + processor."""
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base", device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is not installed.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Initializing GroundingDINO on device: {self.device}")
        self.processor = _load_grounding_processor(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(self.device)
        print("✅ GroundingDINO initialized.")

    @torch.inference_mode()
    def detect(self, image: Image.Image, labels: List[str],
               box_threshold: float, text_threshold: float) -> List[Dict[str, Any]]:
        """Run text-conditioned detection for a list of labels."""
        if len(labels) == 0:
            return []
        text = ". ".join([f"a {l}" for l in labels]) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = [image.size[::-1]]
        result = _post_process_compat(self.processor, outputs, inputs.input_ids, target_sizes, box_threshold, text_threshold)
        label_list = result.get("text_labels", result.get("labels", []))
        dets: List[Dict[str, Any]] = []
        for box, score, lab in zip(result["boxes"], result["scores"], label_list):
            x_min, y_min, x_max, y_max = [float(x) for x in box.tolist()]
            lab = lab if isinstance(lab, str) else str(lab)
            dets.append({"box": [x_min, y_min, x_max, y_max], "score": float(score), "label": lab})
        return dets


# ==============================
# GroundedSAM Pipeline
# ==============================
class GroundedSAMPipeline:
    """GroundingDINO detection → SAM segmentation → save masks & JSON."""
    def __init__(self, sam_checkpoint: str, output_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        self.sam_predictor: Optional[SamPredictor] = None
        self.grounding: Optional[HFGroundedDINO] = None

        # Prepare output folders
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)

        # Initialize SAM and DINO
        if SAM_AVAILABLE and os.path.exists(sam_checkpoint):
            self._init_sam(sam_checkpoint)
        if TRANSFORMERS_AVAILABLE:
            self._init_grounding_detector()
        else:
            print("❌ transformers not available; detection will be skipped.")

    def _init_sam(self, checkpoint_path: str):
        """Load SAM (ViT-H) checkpoint and create predictor."""
        try:
            print("🔧 Initializing SAM...")
            sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
            print(f"✅ SAM initialized (device: {self.device})")
        except Exception as e:
            print(f"❌ Failed to initialize SAM: {e}")

    def _init_grounding_detector(self):
        """Create GroundingDINO detector + processor."""
        try:
            self.grounding = HFGroundedDINO(model_id="IDEA-Research/grounding-dino-base", device=self.device)
        except Exception as e:
            print(f"❌ Failed to initialize GroundingDINO: {e}")
            self.grounding = None

    def segment_with_sam(self, image_rgb: np.ndarray, boxes_xyxy: List[List[float]]) -> List[np.ndarray]:
        """Segment each box with SAM; returns a list of 0/1 masks."""
        if self.sam_predictor is None or len(boxes_xyxy) == 0:
            return []
        self.sam_predictor.set_image(image_rgb)
        masks_out: List[np.ndarray] = []
        for box in boxes_xyxy:
            box_np = np.array(box, dtype=np.float32)
            masks, scores, _ = self.sam_predictor.predict(box=box_np, multimask_output=True)
            best_mask = masks[np.argmax(scores)]
            masks_out.append(best_mask.astype(np.uint8))
        return masks_out


# ==============================
# Exclusion (window/door) helpers
# ==============================
def exclusion_overlap_ratio(box: List[float], excl_mask: np.ndarray) -> float:
    """Return the fraction (0..1) of the candidate box covered by the exclusion mask."""
    x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(excl_mask.shape[1]-1, x2); y2 = min(excl_mask.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = excl_mask[y1:y2, x1:x2]
    if sub.size == 0:
        return 0.0
    return float(sub.mean())  # mean equals coverage ratio for binary mask


def overlaps_exclusions_fallback(label: str, box: List[float], exclusions: List[List[float]]) -> bool:
    """Fallback rule when we only have exclusion boxes (no mask): IoU and optional center rule."""
    if not exclusions:
        return False
    for ex in exclusions:
        if iou_xyxy(box, ex) >= 0.20:
            return True
        if EXCLUSION_CENTER_RULE and center_in_box(box, ex):
            return True
    return False


# ==============================
# Candidate filtering (mask-aware)
# ==============================
def _valid_by_area(box: List[float], img_area: float,
                   min_ratio: float, max_ratio: float) -> bool:
    """Check if box area falls within a given [min_ratio, max_ratio] of image area."""
    x1, y1, x2, y2 = box
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    ratio = area / img_area if img_area > 0 else 0.0
    return (min_ratio <= ratio <= max_ratio)

def filter_candidates_with_detector(
    detector: Optional[HFGroundedDINO],
    image: Image.Image,
    candidates: List[str],
    exclusions: Optional[List[List[float]]] = None,
    exclusion_mask: Optional[np.ndarray] = None,
    min_area_ratio: float = MIN_AREA_RATIO,
    max_area_ratio: float = MAX_AREA_RATIO,
    top_k: int = MAX_PROMPTS,
    box_th: float = BOX_THRESHOLD,
    text_th: float = TEXT_THRESHOLD,
    overlap_drop: float = EXCLUSION_OVERLAP_DROP,
) -> List[str]:
    """
    Keep labels that produce at least one valid detection after:
    - exclusion mask coverage check (preferred),
    - or fallback box-based exclusion (IoU/center),
    - and area constraints.
    """
    if detector is None or len(candidates) == 0:
        return candidates[:top_k]

    W, H = image.size
    img_area = float(W * H)
    ranked: List[Tuple[str, float]] = []

    for label in candidates:
        dets = detector.detect(image, [label], box_threshold=box_th, text_threshold=text_th)
        if len(dets) == 0:
            continue
        best = 0.0
        valid_any = False
        for d in dets:
            box = d["box"]
            # 1) Exclusion mask (preferred)
            if exclusion_mask is not None:
                ov = exclusion_overlap_ratio(box, exclusion_mask)
                if ov >= overlap_drop:
                    continue
            # 2) Fallback to box-only exclusion if mask is unavailable
            elif exclusions and overlaps_exclusions_fallback(label, box, exclusions):
                continue
            # 3) Area constraint
            if not _valid_by_area(box, img_area, min_area_ratio, max_area_ratio):
                continue

            valid_any = True
            best = max(best, d["score"])

        if valid_any:
            ranked.append((label, best))

    ranked.sort(key=lambda t: t[1], reverse=True)
    return [lab for lab, _ in ranked[:top_k]]


def select_labels_for_segmentation(
    detector: Optional[HFGroundedDINO],
    image: Image.Image,
    gpt_raw: List[str],
    exclusions: Optional[List[List[float]]] = None,
    exclusion_mask: Optional[np.ndarray] = None,
    box_th: float = BOX_THRESHOLD,
    text_th: float = TEXT_THRESHOLD,
    min_area_ratio: float = MIN_AREA_RATIO,
    max_area_ratio: float = MAX_AREA_RATIO,
    overlap_drop: float = EXCLUSION_OVERLAP_DROP,
) -> List[str]:
    """
    Label selection with a ≥1 guarantee (indoor-first):
    1) GPT → normalize → filter, respecting exclusion mask (or fallback boxes).
    2) If empty, try prioritized defaults.
    3) If still empty, relax thresholds once.
    4) If still empty, fallback to ["sofa"].
    """
    # Step 1
    labels = normalize_and_filter_labels(gpt_raw)
    labels = filter_candidates_with_detector(
        detector, image, labels,
        exclusions=exclusions, exclusion_mask=exclusion_mask,
        top_k=MAX_PROMPTS, box_th=box_th, text_th=text_th,
        min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio,
        overlap_drop=overlap_drop,
    )
    if labels:
        return labels

    # Step 2
    priors = normalize_and_filter_labels(PREFERRED_ORDER)
    labels = filter_candidates_with_detector(
        detector, image, priors,
        exclusions=exclusions, exclusion_mask=exclusion_mask,
        top_k=1, box_th=box_th, text_th=text_th,
        min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio,
        overlap_drop=overlap_drop,
    )
    if labels:
        return labels

    # Step 3 (relax)
    labels = filter_candidates_with_detector(
        detector, image, priors,
        exclusions=exclusions, exclusion_mask=exclusion_mask,
        top_k=1,
        box_th=max(0.15, box_th - 0.15),
        text_th=max(0.15, text_th - 0.10),
        min_area_ratio=max(0.005, min_area_ratio / 2.0),
        max_area_ratio=min(0.50, max_area_ratio * 1.5),
        overlap_drop=min(0.60, max(0.30, overlap_drop + 0.10)),
    )
    if labels:
        return labels

    # Step 4
    return ["sofa"]


# ==============================
# Entrypoint
# ==============================
def main(args):
    print("🚀 Starting Panorama Segmentation Pipeline")
    print("=" * 60)

    # Load image
    try:
        panorama_pil = Image.open(args.panorama_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ Panorama image not found at: {args.panorama_path}")
        return
    image_rgb = np.array(panorama_pil)

    # Initialize SAM + GroundingDINO
    pipeline = GroundedSAMPipeline(args.sam_checkpoint, args.output_dir)

    # 1) Detect window/door boxes first
    W, H = panorama_pil.size
    exclusions_boxes: List[List[float]] = []
    exclusion_mask: Optional[np.ndarray] = None

    if pipeline.grounding is not None:
        dets_ex = pipeline.grounding.detect(
            panorama_pil, CONTEXT_EXCLUDE_LABELS,
            box_threshold=args.exclusion_box_th, text_threshold=args.exclusion_text_th
        )
        exclusions_boxes = [expand_box(d["box"], W, H, args.exclusion_pad_ratio) for d in dets_ex]

    # 2) Build a union exclusion mask with SAM if possible
    if args.exclusion_use_mask and SAM_AVAILABLE and len(exclusions_boxes) > 0:
        masks = pipeline.segment_with_sam(image_rgb, exclusions_boxes)
        if len(masks) > 0:
            exclusion_mask = np.zeros((H, W), dtype=np.uint8)
            for m in masks:
                exclusion_mask = np.maximum(exclusion_mask, (m > 0).astype(np.uint8))
            if args.exclusion_mask_dilate_px > 0:
                k = int(args.exclusion_mask_dilate_px)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, k), max(1, k)))
                exclusion_mask = cv2.dilate(exclusion_mask, kernel, iterations=1)
            print("🚧 Exclusion mask ready (windows/doors union).")
    elif len(exclusions_boxes) > 0:
        print(f"🚧 Exclusion zones (boxes only): {len(exclusions_boxes)}")

    # Labels: override or GPT
    override_labels = [s for s in (args.labels.split(",") if args.labels else [])]
    raw_labels = get_asset_prompts_from_gpt(args.panorama_path, args.openai_api_key, override_labels or None)

    # Label selection with ≥1 guarantee (respecting exclusion)
    final_labels = select_labels_for_segmentation(
        pipeline.grounding, panorama_pil, raw_labels,
        exclusions=exclusions_boxes, exclusion_mask=exclusion_mask,
        box_th=args.box_threshold, text_th=args.text_threshold,
        min_area_ratio=args.min_area_ratio, max_area_ratio=args.max_area_ratio,
        overlap_drop=args.exclusion_overlap_drop
    )
    print(f"🎯 Final labels to try: {final_labels}")

    # Detection → cross-label NMS → SAM segmentation
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    kept_boxes: List[List[float]] = []
    img_area = float(W * H)

    for prompt in final_labels:
        dets = pipeline.grounding.detect(
            panorama_pil, [prompt],
            box_threshold=args.box_threshold, text_threshold=args.text_threshold
        ) if pipeline.grounding else []

        good: List[Dict[str, Any]] = []
        for d in dets:
            box = d["box"]

            # Exclude outdoor/undesired via mask
            if exclusion_mask is not None:
                if exclusion_overlap_ratio(box, exclusion_mask) >= args.exclusion_overlap_drop:
                    continue
            # Fallback to simple box-based exclusion
            elif exclusions_boxes and overlaps_exclusions_fallback(prompt, box, exclusions_boxes):
                continue

            # Area constraints
            area_ratio = ((box[2]-box[0])*(box[3]-box[1]))/img_area
            if not (args.min_area_ratio <= area_ratio <= args.max_area_ratio):
                continue

            # Cross-label NMS
            if any(iou_xyxy(box, kb) >= CROSS_LABEL_NMS_IOU for kb in kept_boxes):
                continue

            good.append(d)
            kept_boxes.append(box)

        if len(good) == 0:
            continue

        print(f"🧩 {prompt}: kept={len(good)}")

        # Segment with SAM
        boxes = [d["box"] for d in good]
        masks = pipeline.segment_with_sam(image_rgb, boxes)

        merged = []
        for d, m in zip(good, masks):
            merged.append({
                "label": prompt,
                "score": d["score"],
                "box": d["box"],
                "mask": m
            })
        all_results[prompt] = merged

    # Summary
    segmented_counts = {k: len(v) for k, v in all_results.items() if len(v) > 0}
    if segmented_counts:
        pretty = ", ".join([f"{k} x{c}" for k, c in segmented_counts.items()])
        print(f"🏁 Segmented assets: {pretty}")
    else:
        print("⚠️ No assets segmented.")

    # Visualization & JSON export
    image_bgr = cv2.imread(args.panorama_path)
    if image_bgr is None:
        print("❌ Failed to load image for visualization.")
        return

    vis_image = image_bgr.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 64, 0), (0, 128, 255), (128, 0, 128)
    ]
    summary = {"image": os.path.basename(args.panorama_path), "prompts": {}}
    color_idx = 0

    for prompt, dets in all_results.items():
        if len(dets) == 0:
            continue

        color = colors[color_idx % len(colors)]
        color_idx += 1
        H2, W2 = vis_image.shape[:2]
        combined_mask = np.zeros((H2, W2), dtype=np.uint8)
        json_items = []

        for d in dets:
            mask = d["mask"].astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask)
            item = {
                "label": d.get("label", prompt),
                "score": float(d.get("score", 0.0)),
                "box": d.get("box", None),
                "mask_saved_as": None
            }
            json_items.append(item)

            overlay = np.zeros_like(vis_image)
            overlay[mask > 0] = color
            vis_image = cv2.addWeighted(vis_image, 1.0, overlay, 0.45, 0)

        prompt_fname = f"{prompt.replace(' ', '_')}.png"
        mask_path = os.path.join(args.output_dir, "masks", prompt_fname)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, combined_mask * 255)
        for it in json_items:
            it["mask_saved_as"] = os.path.relpath(mask_path, args.output_dir)

        summary["prompts"][prompt] = json_items

    base = os.path.splitext(os.path.basename(args.panorama_path))[0]
    vis_path = os.path.join(args.output_dir, "visualizations", f"{base}_visualization.png")
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    cv2.imwrite(vis_path, vis_image)
    summary["visualization"] = os.path.relpath(vis_path, args.output_dir)

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Visualization saved to: {vis_path}")
    print(f"✅ JSON summary saved to: {json_path}")
    print("=" * 60)
    print("🎉 Pipeline Finished Successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indoor-first panorama segmentation (success-only).")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM ViT-H checkpoint.")
    parser.add_argument("--panorama_path", type=str, required=True, help="Path to the input equirect panorama.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write masks/visualization/JSON.")
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key for GPT step.")

    # Optional overrides
    parser.add_argument("--labels", type=str, default=None, help="Comma-separated labels to force (skip GPT).")

    # Thresholds & ratios
    parser.add_argument("--box_threshold", type=float, default=BOX_THRESHOLD, help="GroundingDINO box threshold.")
    parser.add_argument("--text_threshold", type=float, default=TEXT_THRESHOLD, help="GroundingDINO text threshold.")
    parser.add_argument("--min_area_ratio", type=float, default=MIN_AREA_RATIO, help="Min area ratio for a kept box.")
    parser.add_argument("--max_area_ratio", type=float, default=MAX_AREA_RATIO, help="Max area ratio for a kept box.")

    # Exclusion controls
    parser.add_argument("--exclusion_use_mask", type=lambda x: str(x).lower() not in ["0","false","no"],
                        default=EXCLUSION_USE_MASK, help="Use SAM mask for exclusions instead of box-only fallback.")
    parser.add_argument("--exclusion_mask_dilate_px", type=int, default=EXCLUSION_MASK_DILATE_PX,
                        help="Dilation (px) applied to the union exclusion mask.")
    parser.add_argument("--exclusion_overlap_drop", type=float, default=EXCLUSION_OVERLAP_DROP,
                        help="Drop a candidate if exclusion mask covers this fraction of its box.")
    parser.add_argument("--exclusion_box_th", type=float, default=EXCLUSION_BOX_TH,
                        help="Box threshold for window/door detection.")
    parser.add_argument("--exclusion_text_th", type=float, default=EXCLUSION_TEXT_TH,
                        help="Text threshold for window/door detection.")
    parser.add_argument("--exclusion_pad_ratio", type=float, default=EXCLUSION_PAD_RATIO,
                        help="Padding ratio applied to exclusion boxes in box-only fallback.")

    args = parser.parse_args()
    main(args)
