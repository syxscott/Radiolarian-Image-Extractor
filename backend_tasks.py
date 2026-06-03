# -*- coding: utf-8 -*-
"""
@file: backend_tasks.py
@description: Core scientific processing logic. Contains classes for:
              1. PdfConverter: Multiprocess PDF rendering.
              2. ObjectDetector: YOLOv8/11 inference logic.
              3. ImageClassifier: ResNet/CNN filtering logic.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import os
import re
import shutil
import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from PIL import Image

# Third-party scientific libraries
import fitz  # PyMuPDF
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from ultralytics import YOLO

from config import Config
from backend_signals import WorkerSignals


# ---------------------------------------------------------
# Classifier Architecture Factory
# ---------------------------------------------------------
def _detect_classifier_architecture(state_dict: dict) -> str:
    """
    Inspect a classifier state_dict and return the architecture key
    that matches its structure.  Used to recover gracefully from
    mis-named weight files (e.g. a resnet18 checkpoint saved as
    "resnet50_xxx.pth").

    Detection covers the full set of architectures we build in
    `_build_classifier_model`:
        - ResNet18 / ResNet50 / WideResNet50_2 / ResNeXt50_32x4d
        - DenseNet121
        - VGG16
        - MobileNet V2 / V3 (large/small)
        - EfficientNet B0 / B2
        - ShuffleNet V2 x1.0
        - SqueezeNet 1.1

    Returns one of the arch_keys accepted by `_build_classifier_model`,
    or 'unknown' if the structure does not match any supported model.
    """
    keys = list(state_dict.keys())
    keyset = set(keys)

    # ---------- SqueezeNet 1.1: classifier.1 is a 1x1 Conv2d ----------
    if 'features.0.weight' in keyset and 'classifier.1.weight' in keyset:
        cls_w = state_dict.get('classifier.1.weight')
        if cls_w is not None and cls_w.dim() == 4:
            # Conv weight shape = (out_ch, in_ch, kH, kW); in_ch == 512 for sq1.1
            if int(cls_w.shape[1]) == 512:
                return 'squeezenet1_1'

    # ---------- DenseNet121 ----------
    if any('features.denseblock' in k for k in keys):
        return 'densenet121'

    # ---------- VGG16 ----------
    if 'classifier.6.weight' in keyset and any(k.startswith('features.28.') for k in keys):
        return 'vgg16'

    # ---------- MobileNet V3 (large/small) ----------
    # V3 has classifier.0 (Linear) + classifier.3 (Linear) sandwich.
    if ('classifier.0.weight' in keyset and 'classifier.3.weight' in keyset
            and 'features.0.0.weight' in keyset):
        cls_w = state_dict.get('classifier.3.weight')
        if cls_w is not None and cls_w.dim() == 2:
            in_features = int(cls_w.shape[1])
            if in_features == 1280:
                return 'mobilenet_v3_large'
            if in_features == 1024:
                return 'mobilenet_v3_small'
            # Unknown V3 variant — fall back to large
            return 'mobilenet_v3_large'

    # ---------- MobileNet V2 vs EfficientNet B0/B2 ----------
    # Both share `features.0.0.weight` + `classifier.1.weight` (Linear).
    # Distinguishing trait: MobileNetV2 has 19 inverted-residual blocks
    # (features.0..features.18), while EfficientNet B0/B2 only have 9
    # (features.0..features.8).  Check the MobileNet-specific high block
    # FIRST so we don't mis-route a MobileNet V2 weight to EfficientNet.
    if 'features.0.0.weight' in keyset and 'classifier.1.weight' in keyset:
        cls_w = state_dict.get('classifier.1.weight')
        if cls_w is not None and cls_w.dim() == 2:
            in_features = int(cls_w.shape[1])
            has_block_18 = any(k.startswith('features.18.') for k in keys)
            has_block_8 = any(k.startswith('features.8.') for k in keys)

            if has_block_18 and in_features == 1280:
                # MobileNet V2 (features.0..18, head=1280)
                return 'mobilenet_v2'

            # EfficientNet (features.0..8, head=1280 for B0, 1408 for B2)
            if has_block_8 and not has_block_18:
                if in_features == 1280:
                    return 'efficientnet_b0'
                if in_features == 1408:
                    return 'efficientnet_b2'
                # Unknown EfficientNet variant — default to B0
                return 'efficientnet_b0'

    # ---------- ShuffleNet V2 x1.0 ----------
    if 'conv1.0.weight' in keyset and 'fc.weight' in keyset:
        # ShuffleNet wraps conv1 in a Sequential, ResNet does not.
        return 'shufflenet_v2_x1_0'

    # ---------- ResNet family (BasicBlock / Bottleneck) ----------
    fc_weight = state_dict.get('fc.weight')
    if fc_weight is not None and fc_weight.dim() >= 2:
        fc_in = int(fc_weight.shape[1])
        # Bottleneck blocks have a `conv3`; BasicBlock blocks do not.
        has_conv3 = any('.conv3.weight' in k for k in keys)

        if not has_conv3:
            # BasicBlock family.  We can't reliably tell resnet18 from
            # resnet34 just by state_dict structure, so default to the
            # more common one — resnet18.
            return 'resnet18'

        if fc_in == 2048:
            # All three of resnet50 / wide_resnet50_2 / resnext50_32x4d
            # have fc.in_features == 2048.  Distinguish by the first
            # bottleneck's conv2 shape:
            #   resnet50           : (64, 64,  3, 3)
            #   wide_resnet50_2    : (128, 128, 3, 3)   — wide channels
            #   resnext50_32x4d    : (128, 4,   3, 3)   — grouped conv
            conv2 = state_dict.get('layer1.0.conv2.weight')
            if conv2 is not None and conv2.dim() == 4:
                out_ch, in_ch = int(conv2.shape[0]), int(conv2.shape[1])
                if out_ch == 128 and in_ch == 4:
                    return 'resnext50_32x4d'
                if out_ch == 128 and in_ch == 128:
                    return 'wide_resnet50_2'
            return 'resnet50'
        if fc_in == 1024:
            return 'resnext50_32x4d'

    return 'unknown'


def _build_classifier_model(arch_key: str, num_classes: int) -> nn.Module:
    """
    Construct a torchvision classifier with the given architecture and
    replace its final classification layer to produce `num_classes` outputs.

    The architecture key must be one of the values defined in
    `Config.CLASSIFIER_ARCH_KEYWORDS`.

    All head-replacement operations use `[-1]` indexing for future-proofing
    against torchvision layout changes.  (For ResNet/ResNeXt/WideResNet/
    ShuffleNet, the head is `model.fc`; for everything else it is the
    last child of `model.classifier`.)
    """
    if arch_key == 'resnet18':
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch_key == 'resnet50':
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch_key == 'wide_resnet50_2':
        m = models.wide_resnet50_2(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch_key == 'resnext50_32x4d':
        m = models.resnext50_32x4d(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch_key == 'shufflenet_v2_x1_0':
        m = models.shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch_key == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if arch_key == 'efficientnet_b2':
        m = models.efficientnet_b2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if arch_key == 'mobilenet_v2':
        m = models.mobilenet_v2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if arch_key == 'mobilenet_v3_large':
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if arch_key == 'mobilenet_v3_small':
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if arch_key == 'vgg16':
        m = models.vgg16(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if arch_key == 'densenet121':
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    if arch_key == 'squeezenet1_1':
        m = models.squeezenet1_1(weights=None)
        # NOTE: SqueezeNet's `classifier` is Sequential(Dropout, Conv2d, ReLU, AdaptiveAvgPool2d).
        # The classification layer is the Conv2d at index 1, NOT the last child
        # (which is the avg-pool).  Hence we explicitly use [1] here even
        # though the rest of this factory prefers [-1] for future-proofing.
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        m.num_classes = num_classes
        return m
    raise ValueError(f"Unsupported classifier architecture: '{arch_key}'")


class TaskRunner:
    """Base class for all background tasks."""

    def __init__(self, signals: WorkerSignals):
        self.signals = signals
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True


class PdfConverter(TaskRunner):
    """Handles high-performance PDF to Image conversion."""

    def __init__(self, signals: WorkerSignals, dpi: int = 300):
        super().__init__(signals)
        # `dpi` is the *real* target DPI (e.g. 300 means 300 DPI).  The
        # internal fitz zoom factor is `dpi / 72`.
        self.dpi = int(dpi)
        self.zoom = self.dpi / 72.0

    @staticmethod
    def _convert_page_worker(pdf_path, pages_to_convert, zoom, output_folder):
        """Static worker method for ProcessPoolExecutor context.

        `zoom` is the fitz resolution multiplier (target_dpi / 72).

        Returns:
            (count, errors)
                count  : int — number of pages successfully rendered+saved
                errors : list[str] — one entry per failed page (preserves
                         ALL errors instead of just the last one).
        """
        count = 0
        errors = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_info in pages_to_convert:
                    try:
                        page = doc.load_page(page_info['page_num'])
                        # Set resolution matrix
                        mat = fitz.Matrix(zoom, zoom)
                        # Render to pixmap (RGB)
                        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
                        pix.save(page_info['img_path'])
                        count += 1
                    except Exception as e:
                        errors.append(f"Page {page_info['page_num'] + 1}: {e}")
        except Exception as e:
            errors.append(f"File error: {e}")
        return count, errors

    def run(self):
        self.signals.log(f"▶️ Starting PDF Conversion Engine (DPI: {self.dpi})")

        # 1. Scan Phase
        all_pdfs = [os.path.join(Config.UPLOAD_FOLDER, f) for f in sorted(os.listdir(Config.UPLOAD_FOLDER))
                    if f.lower().endswith('.pdf')]

        if not all_pdfs:
            self.signals.log("⚠️ No PDF files found in Source folder.", "warn")
            self.signals.progress("conversion", 0)
            return

        tasks_by_pdf = {}
        total_pages_found = 0

        self.signals.log("🔍 Scanning PDFs for incremental update...", "info")
        for pdf_path in all_pdfs:
            if self.stop_requested: break
            try:
                doc = fitz.open(pdf_path)
                num_pages = len(doc)
                doc.close()

                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                needed_pages = []

                for p_num in range(num_pages):
                    img_name = f"{base_name}_{p_num + 1}.jpg"
                    img_path = os.path.join(Config.PAGES_FOLDER, img_name)
                    if not os.path.exists(img_path):
                        needed_pages.append({'page_num': p_num, 'img_path': img_path})

                if needed_pages:
                    tasks_by_pdf[pdf_path] = needed_pages
                    total_pages_found += len(needed_pages)
            except Exception as e:
                self.signals.log(f"❌ Scan failed for {os.path.basename(pdf_path)}: {e}", "error")

        if total_pages_found == 0:
            self.signals.log("✅ All pages are up to date.", "success")
            self.signals.progress("conversion", 100)
            return

        # 2. Execution Phase
        self.signals.log(f"🚀 Processing {total_pages_found} pages using {Config.PDF_MAX_WORKERS} cores...", "info")
        # Progress is tracked in *pages*, not in files, so the bar
        # moves smoothly even when PDFs have very different page counts.
        pages_done = 0
        total_pages = total_pages_found

        with ProcessPoolExecutor(max_workers=Config.PDF_MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._convert_page_worker, pdf, pages, self.zoom, Config.PAGES_FOLDER): pdf
                for pdf, pages in tasks_by_pdf.items()
            }

            for future in as_completed(futures):
                if self.stop_requested:
                    # NOTE: cancel_futures only cancels PENDING tasks.  Any
                    # PDF currently being rendered in a child process will
                    # still run to completion — the user should be aware
                    # there may be a short delay before the UI is fully
                    # idle.
                    self.signals.log(
                        "🛑 Stop requested — cancelling pending PDFs "
                        "(active conversions will finish first)...", "warn")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                pdf_name = os.path.basename(futures[future])
                try:
                    # Dynamic timeout based on workload
                    timeout = 120 + len(tasks_by_pdf[futures[future]]) * 15
                    converted, errors = future.result(timeout=timeout)

                    # Always count successful pages — the progress bar must
                    # not freeze just because some pages of a PDF failed.
                    pages_done += converted
                    progress = int((pages_done / total_pages) * 100) if total_pages > 0 else 100
                    self.signals.progress("conversion", progress)

                    if errors:
                        # Surface ALL errors (was previously only the last one).
                        preview = '; '.join(errors[:3])
                        more = '' if len(errors) <= 3 else f' (+{len(errors) - 3} more)'
                        self.signals.log(
                            f"⚠️ {pdf_name}: {len(errors)} error(s): {preview}{more}",
                            "warn")
                    if converted:
                        self.signals.log(
                            f"✅ Converted {pdf_name} ({converted} pages)", "info")

                except TimeoutError:
                    self.signals.log(f"❌ Timeout: {pdf_name} took too long.", "error")
                except Exception as e:
                    self.signals.log(f"❌ Critical error {pdf_name}: {e}", "error")

        if not self.stop_requested:
            self.signals.log("🎉 PDF Conversion Pipeline Completed.", "success")
            self.signals.progress("conversion", 100)
        else:
            self.signals.log("🛑 Operation terminated by user.", "warn")


class ObjectDetector(TaskRunner):
    """Runs YOLO inference to extract fossil regions of interest."""

    def __init__(self, signals: WorkerSignals, model_name: str, model_path: str,
                 confidence: float):
        super().__init__(signals)
        self.model_name = model_name        # Display name (e.g. "yolo11x")
        self.model_path = model_path        # Absolute path, supplied by the UI
        self.conf = confidence

    def run(self):
        if not self.model_path or not os.path.exists(self.model_path):
            self.signals.log(f"❌ Model file missing: {self.model_path}", "error")
            return

        self.signals.log(f"▶️ Initializing YOLO Detector ({self.model_name}) on {Config.DEVICE}...")

        try:
            model = YOLO(self.model_path)
            self.signals.log("🧠 Model loaded successfully.", "info")
        except Exception as e:
            self.signals.log(f"❌ Failed to load model: {e}", "error")
            return

        # Prepare Logging / Resume logic
        det_log_path = os.path.join(Config.LOGS_FOLDER, "detection_log.csv")
        scanned_log_path = os.path.join(Config.LOGS_FOLDER, "scanned_pages_log.txt")

        processed_files = set()
        if os.path.exists(scanned_log_path):
            with open(scanned_log_path, 'r', encoding='utf-8') as f:
                processed_files = set(line.strip() for line in f)

        # ID Counter recovery — take the max across both the CSV log
        # AND the existing files on disk.  Consulting only the CSV would
        # let a deleted/corrupt log silently reset the counter to 1 and
        # OVERWRITE existing rad_0000001.jpg in 03_Candidate_Images/
        # and 03a_Rejected_Images/.
        rad_counter = 1

        # (a) from CSV log
        if os.path.exists(det_log_path) and os.path.getsize(det_log_path) > 0:
            try:
                df = pd.read_csv(det_log_path)
                if 'temp_image_id' in df.columns:
                    ids = (df['temp_image_id'].dropna().astype(str)
                           .str.extract(r'rad_(\d+)', expand=False)
                           .dropna().astype(int))
                    if not ids.empty:
                        rad_counter = max(rad_counter, int(ids.max()) + 1)
            except Exception as e:
                self.signals.log(f"⚠️ Could not parse {det_log_path}: {e}", "warn")

        # (b) from existing files on disk (kept + rejected)
        id_re = re.compile(r'rad_(\d+)')
        for folder in (Config.CROPPED_FOLDER, Config.REJECTED_FOLDER):
            try:
                for fname in os.listdir(folder):
                    m = id_re.match(fname)
                    if m:
                        rad_counter = max(rad_counter, int(m.group(1)) + 1)
            except OSError:
                pass

        # Identify pending work
        all_imgs = sorted([f for f in os.listdir(Config.PAGES_FOLDER) if f.lower().endswith(('.jpg', '.png'))])
        to_process = [f for f in all_imgs if f not in processed_files]

        if not to_process:
            self.signals.log("✅ Detection is up to date.", "success")
            self.signals.progress("detection", 100)
            return

        self.signals.log(f"Processing {len(to_process)} new page images "
                         f"(starting ID: rad_{rad_counter:07d})...", "info")

        # File handles — use context managers so we never leak descriptors
        # when an exception breaks out of the loop.  Also pre-compute
        # whether the CSV needs a header BEFORE opening in append mode.
        needs_header = (not os.path.exists(det_log_path)
                        or os.path.getsize(det_log_path) == 0)

        # Batch Processing
        batch_size = Config.DETECTION_BATCH_SIZE
        processed_count = 0

        with open(det_log_path, 'a', newline='', encoding='utf-8') as f_det, \
                open(scanned_log_path, 'a', encoding='utf-8') as f_scan:
            writer = csv.writer(f_det)
            if needs_header:
                writer.writerow(['temp_image_id', 'source_paper', 'source_page',
                                 'bbox', 'confidence'])

            for i in range(0, len(to_process), batch_size):
                if self.stop_requested: break

                batch_files = to_process[i: i + batch_size]
                batch_paths = [os.path.join(Config.PAGES_FOLDER, f) for f in batch_files]

                try:
                    results = model(batch_paths, verbose=False, device=Config.DEVICE, stream=False)

                    # Iterate results
                    img_cache = {}  # Lazy load images for cropping

                    for j, res in enumerate(results):
                        filename = batch_files[j]
                        src_path = batch_paths[j]

                        # Metadata extraction
                        parts = os.path.splitext(filename)[0].rsplit('_', 1)
                        s_paper = parts[0]
                        s_page = parts[1] if len(parts) > 1 else '?'

                        # Track whether this page was processed cleanly.
                        # We only mark it as scanned in the log AFTER all
                        # crops for the page succeed — otherwise a mid-page
                        # crash would permanently skip the page on resume.
                        page_ok = True

                        # Process boxes
                        for box in res.boxes:
                            conf = float(box.conf[0])
                            if conf < self.conf:
                                continue

                            coords = [int(x) for x in box.xyxy[0].tolist()]

                            # Lazy-open the source page image
                            if src_path not in img_cache:
                                try:
                                    img_cache[src_path] = Image.open(src_path)
                                except Exception as e:
                                    self.signals.log(
                                        f"⚠️ Cannot open page image '{filename}': {e}",
                                        "warn")
                                    page_ok = False
                                    break  # skip the rest of this page's boxes

                            # Crop — only increment counter on full success
                            crop_id = f"rad_{rad_counter:07d}"
                            try:
                                crop_img = img_cache[src_path].crop(coords)
                                crop_img.save(os.path.join(
                                    Config.CROPPED_FOLDER, f"{crop_id}.jpg"))
                                writer.writerow([crop_id, s_paper, s_page,
                                                 str(coords), f"{conf:.4f}"])
                                rad_counter += 1
                            except Exception as e:
                                self.signals.log(
                                    f"⚠️ Crop/CSV error on {filename}: {e}", "warn")
                                page_ok = False
                                # Don't increment counter — the slot is unused.

                        # Mark page as scanned ONLY if everything succeeded.
                        # On partial failure we'll re-attempt on next run.
                        if page_ok:
                            f_scan.write(f"{filename}\n")

                    # Close cached images to release file handles
                    for img in img_cache.values():
                        try:
                            img.close()
                        except Exception:
                            pass

                except Exception as e:
                    self.signals.log(f"❌ Batch Inference Error: {e}", "error")

                processed_count += len(batch_files)
                self.signals.progress("detection", int((processed_count / len(to_process)) * 100))

                # Flush buffers every batch so a crash doesn't lose progress
                f_det.flush()
                f_scan.flush()

        if not self.stop_requested:
            self.signals.log("🎉 Detection Pipeline Completed.", "success")
        else:
            self.signals.log("🛑 Detection Aborted.", "warn")


class ImageClassifier(TaskRunner):
    """Runs a torchvision classifier to filter out noise from extracted images."""

    def __init__(self, signals: WorkerSignals, model_name: str, model_path: str,
                 arch_key: str):
        super().__init__(signals)
        self.model_name = model_name        # Display name (e.g. "best_radiolarian_classifier_resnet50_torchvision")
        self.model_path = model_path        # Absolute path to the .pth file
        self.arch_key = arch_key            # e.g. "resnet50", "mobilenet_v2", ...

        # Standard ImageNet normalization (works for all torchvision
        # classifiers we support)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def run(self):
        if not self.model_path or not os.path.exists(self.model_path):
            self.signals.log(f"❌ Model file missing: {self.model_path}", "error")
            return

        self.signals.log(
            f"▶️ Initializing Classifier ({self.model_name} / arch={self.arch_key})..."
        )

        try:
            num_classes = len(Config.CLASSIFIER_CLASS_NAMES)
            # Load Weights.
            # PyTorch 2.6+ defaults `weights_only=True`, which refuses
            # to unpickle files that use `persistent_load` (a common
            # case for weight files saved by older PyTorch versions or
            # third-party training scripts).  We try the safe mode
            # first, and fall back to the legacy full-pickle mode with
            # an explicit warning so the user knows they should trust
            # the source of the .pth file.
            try:
                state_dict = torch.load(
                    self.model_path, map_location=Config.DEVICE, weights_only=True
                )
                self.signals.log("🔒 Loaded weights in safe mode (weights_only=True).", "info")
            except Exception as safe_err:
                self.signals.log(
                    f"⚠️ Safe-mode load failed ({type(safe_err).__name__}: {safe_err}). "
                    f"Falling back to legacy pickle. Only do this with TRUSTED weight files!",
                    "warn",
                )
                state_dict = torch.load(
                    self.model_path, map_location=Config.DEVICE, weights_only=False
                )

            # Build the right architecture based on the filename's
            # arch_key, but be robust to mis-named weight files
            # (e.g. a resnet18 checkpoint saved as "resnet50_xxx.pth"):
            # if the load fails with a structural mismatch, inspect
            # the state_dict to detect the real architecture and
            # retry with the correct one.
            model = _build_classifier_model(self.arch_key, num_classes)
            actual_arch = self.arch_key
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as load_err:
                detected = _detect_classifier_architecture(state_dict)
                if detected != 'unknown' and detected != self.arch_key:
                    self.signals.log(
                        f"⚠️ Filename says arch='{self.arch_key}' but the weight "
                        f"file's structure matches '{detected}'. Auto-correcting. "
                        f"(Consider renaming the file so the architecture keyword "
                        f"in its name matches the actual model.)",
                        "warn",
                    )
                    try:
                        model = _build_classifier_model(detected, num_classes)
                        model.load_state_dict(state_dict)
                        actual_arch = detected
                    except RuntimeError as retry_err:
                        # The detected arch also doesn't load — give a
                        # clearer error than the cryptic state_dict mismatch.
                        raise RuntimeError(
                            f"Auto-detected arch='{detected}' also failed to "
                            f"load the weights: {retry_err}"
                        ) from retry_err
                else:
                    raise  # re-raise the original mismatch error

            model.to(Config.DEVICE)
            model.eval()
            self.signals.log(f"🧠 Classifier loaded (arch={actual_arch}).", "info")
        except Exception as e:
            self.signals.log(f"❌ Classifier Load Error: {e}", "error")
            return

        # Sort for deterministic ordering — `os.listdir` order is
        # filesystem-dependent, which made progress reporting and stop/resume
        # behaviour non-reproducible across runs.
        imgs = sorted(f for f in os.listdir(Config.CROPPED_FOLDER)
                      if f.lower().endswith('.jpg'))
        total = len(imgs)

        if total == 0:
            self.signals.log("⚠️ No candidate images to filter.", "warn")
            self.signals.progress("classification", 100)
            return

        self.signals.log(f"Filtering {total} candidates...", "info")

        batch_size = Config.CLASSIFICATION_BATCH_SIZE
        rejected_count = 0
        processed_count = 0
        rejected_ids = []

        with torch.no_grad():
            for i in range(0, total, batch_size):
                if self.stop_requested: break

                batch_files = imgs[i: i + batch_size]
                batch_tensors = []
                valid_files = []
                skipped_in_batch = []

                # Preprocessing
                for fname in batch_files:
                    try:
                        p = os.path.join(Config.CROPPED_FOLDER, fname)
                        img = Image.open(p).convert('RGB')
                        batch_tensors.append(self.transform(img))
                        valid_files.append(fname)
                    except Exception as e:
                        skipped_in_batch.append(fname)
                        continue

                if not batch_tensors:
                    # Entire batch was unreadable — log a warning so the
                    # user can investigate (was previously a silent skip).
                    self.signals.log(
                        f"⚠️ Skipped {len(skipped_in_batch)} unreadable image(s) "
                        f"in batch starting at index {i}: {skipped_in_batch[:3]}",
                        "warn",
                    )
                    continue

                if skipped_in_batch:
                    self.signals.log(
                        f"⚠️ Skipped {len(skipped_in_batch)} unreadable image(s) "
                        f"in batch starting at index {i}.",
                        "warn",
                    )

                # Inference
                input_tensor = torch.stack(batch_tensors).to(Config.DEVICE)
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)

                # Post-processing
                for idx, class_idx in enumerate(preds):
                    cls_name = Config.CLASSIFIER_CLASS_NAMES[class_idx.item()]
                    fname = valid_files[idx]

                    if cls_name != Config.POSITIVE_CLASS_NAME:
                        # Reject: Move to rejected folder
                        src = os.path.join(Config.CROPPED_FOLDER, fname)
                        dst = os.path.join(Config.REJECTED_FOLDER, fname)
                        try:
                            # On Windows, shutil.move raises if the
                            # destination already exists (unlike POSIX which
                            # silently overwrites).  Remove a stale target
                            # first so re-runs after a partial classification
                            # don't bail out and leave images stuck in the
                            # candidate folder.
                            if os.path.exists(dst):
                                try:
                                    os.remove(dst)
                                except OSError as e:
                                    self.signals.log(
                                        f"⚠️ Cannot overwrite existing "
                                        f"'{fname}' in rejected folder: {e}",
                                        "warn",
                                    )
                                    continue
                            shutil.move(src, dst)
                            rejected_count += 1
                            rejected_ids.append(os.path.splitext(fname)[0])
                        except OSError as e:
                            # Surface to the GUI so the user sees data
                            # integrity issues instead of getting
                            # misleading "Removed X artifacts" messages.
                            self.signals.log(
                                f"⚠️ Failed to move '{fname}' to rejected folder: {e}",
                                "warn",
                            )

                # Count only files we actually processed (not the
                # skipped/invalid ones) so the progress bar is accurate.
                processed_count += len(valid_files)
                self.signals.progress("classification", int((processed_count / total) * 100))

        # Cleanup Logs
        if rejected_ids:
            self._clean_csv_log(rejected_ids)

        if not self.stop_requested:
            self.signals.log(f"🎉 Classification Complete. Removed {rejected_count} artifacts.", "success")
        else:
            self.signals.log("🛑 Classification Stopped.", "warn")

    def _clean_csv_log(self, rejected_ids):
        log_path = os.path.join(Config.LOGS_FOLDER, "detection_log.csv")
        if os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                if 'temp_image_id' not in df.columns:
                    return
                clean_df = df[~df['temp_image_id'].isin(rejected_ids)]
                clean_df.to_csv(log_path, index=False)
                self.signals.log("📄 Updated metadata log.", "info")
            except Exception as e:
                self.signals.log(f"⚠️ Log update failed: {e}", "warn")