# -*- coding: utf-8 -*-
"""
@file: ui_panels.py
@description: Implementation of the specific content tabs (Processing & Gallery).
@author: Yaxuan Shen
@date: 2025-10-01
"""

import os
import sys
import math
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

from config import Config
from ui_widgets import (
    ToolTip, ScrollableFrame, Card, StepCard, StatusBadge,
    SectionHeader, HSeparator, VSeparator,
)
from ui_platform import emoji_or_text
from ui_theme import Colors, Spacing
from utils import sanitize_filename, open_path
import shutil


# Optional drag-and-drop.  tkinterdnd2 is an optional dependency
# (pip install tkinterdnd2).  If it's not installed we degrade silently
# to button-only import.
try:
    from tkinterdnd2 import DND_FILES   # noqa: F401
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False


# Cross-platform PDF filter.  On Linux GTK, filetype patterns are
# CASE SENSITIVE, so a single "*.pdf" hides any file named "*.PDF"
# (which is common for files exported from Mendeley/EndNote/etc.).
# Listing all common case variants makes the dialog show every PDF.
PDF_FILETYPES = [
    ("PDF Files", ("*.pdf", "*.PDF", "*.Pdf")),
    ("All Files", "*.*"),
]


class ProcessingPanel:
    """The main dashboard for running the data pipeline."""

    def __init__(self, parent, controller):
        self.controller = controller  # Reference to main app to start threads
        self.fonts = getattr(controller, 'fonts', None)

        # The Pipeline tab is itself a scrollable area with a soft grey
        # background so the white step cards "float" against it.  The
        # ScrollableFrame uses a Canvas internally, so we colour the
        # canvas to match the page background.
        self.frame = ScrollableFrame(parent)
        try:
            self.frame.canvas.config(bg=Colors.BG_PAGE)
        except tk.TclError:
            pass
        self.content = self.frame.scrollable_frame
        # Apply page-style background to the content frame
        try:
            ttk.Style().configure('Pipeline.TFrame', background=Colors.BG_PAGE)
            self.content.configure(style='Pipeline.TFrame')
        except tk.TclError:
            pass

        # Caches of the most recent directory scan results.  These are
        # consulted by the click handlers so we do NOT hit the disk
        # every time the user clicks a button.
        self._yolo_cache = {}    # display_name -> absolute_path
        self._cls_cache = {}     # display_name -> (absolute_path, arch_key)

        # Remember the user's last-used import directory so consecutive
        # imports don't re-start from $HOME every time.
        self._last_pdf_dir = os.path.expanduser('~')

        # Step cards registry — controller calls set_step_status(key, state)
        self._step_cards = {}    # 'convert' / 'detect' / 'classify' -> StepCard

        self._build_step1_data()
        self._build_step2_conversion()
        self._build_step3_detection()
        self._build_step4_classification()
        self._build_logs()

        # Register the panel as a drop target so users can drop files
        # or folders anywhere on it.  No-op if tkinterdnd2 isn't installed.
        self._setup_drag_and_drop()

    def set_step_status(self, key, state):
        """Update the status badge of a step card.

        Args:
            key:   'convert' / 'detect' / 'classify'
            state: 'pending' / 'running' / 'done' / 'failed'
        """
        card = self._step_cards.get(key)
        if card is not None:
            try:
                card.set_status(state)
            except tk.TclError:
                pass

    # ---------------------------------------------------------
    # Step 1 — Data Preparation
    # ---------------------------------------------------------
    def _build_step1_data(self):
        card = StepCard(self.content, "01", "Data Preparation",
                        fonts=self.fonts)
        card.pack(fill=tk.X, padx=Spacing.PAGE_PAD,
                  pady=(Spacing.PAGE_PAD, Spacing.CARD_GAP))

        body = card.body

        # Row 1 — action buttons
        row1 = tk.Frame(body, bg=Colors.BG_CARD)
        row1.pack(fill=tk.X)

        self.btn_add = ttk.Button(row1, text="Import PDF Files",
                                  command=self.import_pdfs)
        self.btn_folder = ttk.Button(row1, text="Import Folder",
                                     command=self.import_pdfs_folder)
        self.btn_open_src = ttk.Button(
            row1, text="Open Source Folder",
            command=lambda: open_path(Config.UPLOAD_FOLDER))

        self.btn_add.pack(side=tk.LEFT)
        self.btn_folder.pack(side=tk.LEFT, padx=(Spacing.FIELD_GAP, 0))
        self.btn_open_src.pack(side=tk.LEFT, padx=(Spacing.FIELD_GAP, 0))

        # Recursive subdirectory scan toggle
        self.var_recurse = tk.BooleanVar(value=True)
        self.check_recurse = ttk.Checkbutton(
            row1, text="Include subfolders",
            variable=self.var_recurse,
            style='TCheckbutton')
        self.check_recurse.pack(side=tk.LEFT,
                                padx=(Spacing.FIELD_GAP * 2, 0))

        # Row 2 — count, muted
        row2 = tk.Frame(body, bg=Colors.BG_CARD)
        row2.pack(fill=tk.X, pady=(Spacing.ROW_GAP, 0))

        self.lbl_count = ttk.Label(row2, text="0 files ready",
                                   style='CardMuted.TLabel')
        self.lbl_count.pack(side=tk.LEFT)

    # ---------------------------------------------------------
    # Step 2 — PDF Conversion
    # ---------------------------------------------------------
    def _build_step2_conversion(self):
        card = StepCard(self.content, "02", "PDF Conversion",
                        fonts=self.fonts)
        card.pack(fill=tk.X, padx=Spacing.PAGE_PAD, pady=Spacing.CARD_GAP)
        # Register so controller can update the badge
        self._step_cards['convert'] = card

        body = card.body

        # Row 1 — controls
        row1 = tk.Frame(body, bg=Colors.BG_CARD)
        row1.pack(fill=tk.X)

        ttk.Label(row1, text="DPI",
                  style='CardBody.TLabel').pack(side=tk.LEFT)

        self.var_dpi = tk.StringVar(value=str(Config.DEFAULT_DPI))
        self.combo_dpi = ttk.Combobox(
            row1,
            textvariable=self.var_dpi,
            values=[str(v) for v in Config.DPI_PRESETS],
            state="readonly",
            width=8,
        )
        self.combo_dpi.pack(side=tk.LEFT, padx=(Spacing.FIELD_GAP, 0))

        # Hint label about render scale
        ttk.Label(row1,
                  text="(higher = sharper but slower / larger files)",
                  style='CardMuted.TLabel').pack(
            side=tk.LEFT, padx=(Spacing.FIELD_GAP, 0))

        self.btn_convert = ttk.Button(
            row1, text="Start Conversion", style='Accent.TButton',
            command=self._on_click_convert,
        )
        self.btn_convert.pack(side=tk.RIGHT)

        # Row 2 — progress bar
        self.prog_convert = ttk.Progressbar(body, mode='determinate')
        self.prog_convert.pack(fill=tk.X, pady=(Spacing.ROW_GAP, 0))

    def _on_click_convert(self):
        try:
            dpi = int(self.var_dpi.get())
        except ValueError:
            self.controller.log(f"⚠️ Invalid DPI value: {self.var_dpi.get()!r}", "warn")
            return
        if dpi <= 0:
            self.controller.log(f"⚠️ DPI must be positive (got {dpi}).", "warn")
            return
        self.controller.start_conversion(dpi)

    # ---------------------------------------------------------
    # Step 3 — Object Detection
    # ---------------------------------------------------------
    def _build_step3_detection(self):
        card = StepCard(self.content, "03", "Object Detection",
                        fonts=self.fonts)
        card.pack(fill=tk.X, padx=Spacing.PAGE_PAD, pady=Spacing.CARD_GAP)
        self._step_cards['detect'] = card

        body = card.body

        # ---- Row 1: Model selector + refresh ----
        row1 = tk.Frame(body, bg=Colors.BG_CARD)
        row1.pack(fill=tk.X)

        ttk.Label(row1, text="Model",
                  style='CardBody.TLabel').pack(side=tk.LEFT)

        self.combo_yolo = ttk.Combobox(row1, state="readonly", width=28)
        self.combo_yolo.pack(side=tk.LEFT, padx=(Spacing.FIELD_GAP, 0))

        # Emoji-or-text label so Linux users without color-emoji fonts
        # don't see a mystery blank button.
        self.btn_yolo_refresh = ttk.Button(
            row1, text=emoji_or_text("↻", "Reload"),
            width=4, style='Toolbar.TButton',
            command=self._refresh_yolo_models)
        self.btn_yolo_refresh.pack(side=tk.LEFT, padx=(4, 0))
        ToolTip(self.btn_yolo_refresh,
                text="Re-scan models/yolo/ for new weights")

        # Start Detection button on the right
        self.btn_detect = ttk.Button(
            row1, text="Start Detection", style='Accent.TButton',
            command=self._on_click_detect)
        self.btn_detect.pack(side=tk.RIGHT)

        # ---- Row 2: Confidence slider with live value ----
        row2 = tk.Frame(body, bg=Colors.BG_CARD)
        row2.pack(fill=tk.X, pady=(Spacing.ROW_GAP, 0))

        ttk.Label(row2, text="Confidence",
                  style='CardBody.TLabel').pack(side=tk.LEFT)

        # Live-value slider: use both the `command` callback (fires while
        # dragging) AND a variable trace (fires for any programmatic
        # `.set()`), so the label is always in sync with the variable.
        self.var_conf = tk.DoubleVar(value=0.25)
        self.var_conf.trace_add('write', self._on_conf_var_changed)
        self.scale_conf = ttk.Scale(
            row2, from_=0.1, to=1.0, variable=self.var_conf,
            orient=tk.HORIZONTAL, length=240,
            command=self._on_conf_change,
        )
        self.scale_conf.pack(side=tk.LEFT, padx=(Spacing.FIELD_GAP, 0))

        self.lbl_conf_value = ttk.Label(
            row2, text="0.25", width=5,
            style='CardValue.TLabel')
        self.lbl_conf_value.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(row2,
                  text="(min box confidence to crop)",
                  style='CardMuted.TLabel').pack(side=tk.LEFT,
                                                  padx=(Spacing.FIELD_GAP, 0))

        # ---- Row 3: progress bar ----
        self.prog_detect = ttk.Progressbar(body, mode='determinate')
        self.prog_detect.pack(fill=tk.X, pady=(Spacing.ROW_GAP, 0))

        # Populate the dropdown from disk
        self._refresh_yolo_models(silent=True)

    def _on_conf_change(self, value):
        """Update the conf-value label as the user drags the slider."""
        try:
            self.lbl_conf_value.config(text=f"{float(value):.2f}")
        except (ValueError, tk.TclError):
            pass

    def _on_conf_var_changed(self, *_args):
        """Fires when the DoubleVar is changed programmatically."""
        try:
            self.lbl_conf_value.config(
                text=f"{float(self.var_conf.get()):.2f}")
        except (ValueError, tk.TclError):
            pass

    def _refresh_yolo_models(self, silent: bool = False):
        """Re-scan models/yolo/ and update the YOLO combobox + cache."""
        self._yolo_cache = Config.scan_yolo_models()
        values = list(self._yolo_cache.keys())
        self.combo_yolo['values'] = values
        if values:
            current = self.combo_yolo.get()
            if current not in values:
                self.combo_yolo.current(0)
        else:
            self.combo_yolo.set('')
        if not silent:
            if values:
                self.controller.log(f"🔄 Found {len(values)} YOLO model(s).", "info")
            else:
                self.controller.log("⚠️ No YOLO models found in models/yolo/.", "warn")

    def _on_click_detect(self):
        display = self.combo_yolo.get()
        if not display:
            self.controller.log("⚠️ Please select a YOLO model first.", "warn")
            return
        # Fall back to a fresh scan if the cache is stale (e.g. user
        # dropped a new weight file without hitting the refresh button).
        if display not in self._yolo_cache:
            self._refresh_yolo_models(silent=True)
        if display not in self._yolo_cache:
            self.controller.log(f"❌ Model not found on disk: {display}", "error")
            return
        # Read from the DoubleVar so we get exactly what the user sees
        # in the label (not a stale value from before the last drag).
        conf = round(float(self.var_conf.get()), 2)
        self.controller.start_detection(display, self._yolo_cache[display], conf)

    # ---------------------------------------------------------
    # Step 4 — Quality Control (Classification)
    # ---------------------------------------------------------
    def _build_step4_classification(self):
        card = StepCard(self.content, "04", "Quality Control",
                        fonts=self.fonts)
        card.pack(fill=tk.X, padx=Spacing.PAGE_PAD, pady=Spacing.CARD_GAP)
        self._step_cards['classify'] = card

        body = card.body

        # Row 1 — Classifier picker
        row1 = tk.Frame(body, bg=Colors.BG_CARD)
        row1.pack(fill=tk.X)

        ttk.Label(row1, text="Classifier",
                  style='CardBody.TLabel').pack(side=tk.LEFT)

        self.combo_cls = ttk.Combobox(row1, state="readonly", width=36)
        self.combo_cls.pack(side=tk.LEFT, padx=(Spacing.FIELD_GAP, 0))

        self.btn_cls_refresh = ttk.Button(
            row1, text=emoji_or_text("↻", "Reload"),
            width=4, style='Toolbar.TButton',
            command=self._refresh_classifier_models)
        self.btn_cls_refresh.pack(side=tk.LEFT, padx=(4, 0))
        ToolTip(self.btn_cls_refresh,
                text="Re-scan models/classifiers/ for new weights")

        self.lbl_arch = ttk.Label(row1, text="arch: —",
                                  style='CardMuted.TLabel')
        self.lbl_arch.pack(side=tk.LEFT, padx=(Spacing.FIELD_GAP, 0))

        self.combo_cls.bind("<<ComboboxSelected>>", self._update_arch_label)

        self.btn_classify = ttk.Button(
            row1, text="Start Filter", style='Accent.TButton',
            command=self._on_click_classify)
        self.btn_classify.pack(side=tk.RIGHT)

        # Row 2 — progress
        self.prog_classify = ttk.Progressbar(body, mode='determinate')
        self.prog_classify.pack(fill=tk.X, pady=(Spacing.ROW_GAP, 0))

        # Populate the dropdown from disk
        self._refresh_classifier_models(silent=True)

    def _refresh_classifier_models(self, silent: bool = False):
        """Re-scan models/classifiers/ and update the classifier combobox + cache."""
        self._cls_cache = Config.scan_classifier_models()
        values = list(self._cls_cache.keys())
        self.combo_cls['values'] = values
        if values:
            current = self.combo_cls.get()
            if current not in values:
                self.combo_cls.current(0)
        else:
            self.combo_cls.set('')
        self._update_arch_label()
        if not silent:
            if values:
                self.controller.log(f"🔄 Found {len(values)} classifier model(s).", "info")
            else:
                self.controller.log("⚠️ No classifier models found in models/classifiers/.", "warn")

    def _update_arch_label(self, _event=None):
        """Show the matched architecture next to the dropdown."""
        display = self.combo_cls.get()
        if display in self._cls_cache:
            _, arch_key = self._cls_cache[display]
            self.lbl_arch.config(text=f"arch: {arch_key}")
        else:
            self.lbl_arch.config(text="arch: —")

    def _on_click_classify(self):
        display = self.combo_cls.get()
        if not display:
            self.controller.log("⚠️ Please select a classifier first.", "warn")
            return
        if display not in self._cls_cache:
            self._refresh_classifier_models(silent=True)
        if display not in self._cls_cache:
            self.controller.log(f"❌ Model not found on disk: {display}", "error")
            return
        model_path, arch_key = self._cls_cache[display]
        self.controller.start_classification(display, model_path, arch_key)

    # ---------------------------------------------------------
    # Widget state management
    # ---------------------------------------------------------
    def interactive_widgets(self):
        """Return a list of widgets that should be disabled while a
        background task is running (so the user cannot start a
        conflicting operation, import PDFs, or wipe the cache)."""
        widgets = [
            self.btn_add, self.btn_folder, self.btn_open_src, self.check_recurse,
            self.combo_dpi, self.btn_convert,
            self.combo_yolo, self.btn_yolo_refresh, self.scale_conf, self.btn_detect,
            self.combo_cls, self.btn_cls_refresh, self.btn_classify,
        ]
        # New log toolbar buttons — only present after _build_logs ran
        for name in ('btn_log_clear', 'btn_log_export'):
            w = getattr(self, name, None)
            if w is not None:
                widgets.append(w)
        return widgets

    def _build_logs(self):
        """System Logs card with a toolbar (filter / search / clear / export)
        and a light-themed Text widget."""
        # Use a Card so it matches the step-card visual language.
        card = Card(self.content)
        card.pack(fill=tk.BOTH, expand=True,
                  padx=Spacing.PAGE_PAD,
                  pady=(Spacing.CARD_GAP, Spacing.PAGE_PAD))

        body = card.body

        # ---- Toolbar row ----
        toolbar = tk.Frame(body, bg=Colors.BG_CARD)
        toolbar.pack(fill=tk.X,
                     padx=Spacing.CARD_PAD_X,
                     pady=(Spacing.CARD_PAD_Y, Spacing.CARD_INNER))

        # Title on the left
        F = self.fonts
        tk.Label(toolbar, text="System Logs",
                 bg=Colors.BG_CARD, fg=Colors.TEXT_HEAD,
                 font=F.CARD_TITLE if F else ('TkDefaultFont', 11, 'bold'),
                 padx=0, pady=0).pack(side=tk.LEFT)

        # Action buttons on the right (Export / Clear)
        self.btn_log_export = ttk.Button(
            toolbar, text="Export…",
            style='Toolbar.TButton',
            command=self._export_log)
        self.btn_log_export.pack(side=tk.RIGHT)
        ToolTip(self.btn_log_export,
                text="Save the current log content to a .txt file")

        self.btn_log_clear = ttk.Button(
            toolbar, text="Clear",
            style='Toolbar.TButton',
            command=self._clear_log)
        self.btn_log_clear.pack(side=tk.RIGHT, padx=(0, 6))
        ToolTip(self.btn_log_clear, text="Erase all messages from the log view")

        # Filter checkboxes in the middle
        self.var_show_info = tk.BooleanVar(value=True)
        self.var_show_warn = tk.BooleanVar(value=True)
        self.var_show_error = tk.BooleanVar(value=True)
        self.var_show_success = tk.BooleanVar(value=True)

        # Inline label
        tk.Label(toolbar, text="Show:",
                 bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED,
                 font=F.MUTED if F else ('TkDefaultFont', 9),
                 padx=0, pady=0).pack(side=tk.LEFT, padx=(Spacing.SECTION_PAD, 4))

        for label, var, tag in (
                ("Info",    self.var_show_info,    'INFO'),
                ("Warn",    self.var_show_warn,    'WARN'),
                ("Error",   self.var_show_error,   'ERROR'),
                ("Success", self.var_show_success, 'SUCCESS'),
        ):
            cb = ttk.Checkbutton(
                toolbar, text=label, variable=var,
                command=lambda t=tag, v=var: self._toggle_log_level(t, v),
                style='TCheckbutton')
            cb.pack(side=tk.LEFT, padx=(0, 4))

        # Separator below toolbar
        tk.Frame(body, height=1, bg=Colors.BORDER_LIGHT).pack(
            fill='x', padx=Spacing.CARD_PAD_X)

        # ---- Text widget area ----
        text_wrap = tk.Frame(body, bg=Colors.BG_CARD)
        text_wrap.pack(fill=tk.BOTH, expand=True,
                       padx=Spacing.CARD_PAD_X,
                       pady=(Spacing.CARD_INNER, Spacing.CARD_PAD_Y))

        # Make Text + Scrollbar look unified
        self.txt_log = tk.Text(
            text_wrap, height=12, state='disabled',
            bg=Colors.BG_LOG, fg=Colors.TEXT_BODY,
            insertbackground=Colors.TEXT_HEAD,
            selectbackground=Colors.PRIMARY,
            selectforeground=Colors.TEXT_ON_DARK,
            font=F.LOG if F else ('TkFixedFont', 9),
            spacing1=2, spacing3=2,                  # line gap above/below
            relief='flat', borderwidth=0,
            highlightthickness=0,
            wrap='word')
        scroll = ttk.Scrollbar(text_wrap, orient='vertical',
                               command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.txt_log.pack(side='left', fill='both', expand=True)

        # Log Tags — academic palette
        self.txt_log.tag_config('TIMESTAMP',
                                foreground=Colors.TEXT_MUTED)
        self.txt_log.tag_config('INFO',
                                foreground=Colors.TEXT_BODY)
        self.txt_log.tag_config('WARN',
                                foreground=Colors.WARNING)
        self.txt_log.tag_config('ERROR',
                                foreground=Colors.DANGER)
        self.txt_log.tag_config('SUCCESS',
                                foreground=Colors.SUCCESS)
        self.txt_log.tag_config('MUTED',
                                foreground=Colors.TEXT_MUTED)

    # ---- Log toolbar handlers ----
    def _toggle_log_level(self, tag, var):
        """Hide / show all messages of a given level by changing the
        tag's `elide` attribute.  Existing lines are affected too."""
        try:
            self.txt_log.tag_config(tag, elide=not bool(var.get()))
        except tk.TclError:
            pass

    def _clear_log(self):
        try:
            self.txt_log.config(state='normal')
            self.txt_log.delete('1.0', tk.END)
            self.txt_log.config(state='disabled')
            self.controller.log("Log cleared.", "info")
        except tk.TclError:
            pass

    def _export_log(self):
        """Save the entire log content to a .txt file chosen by the user."""
        path = filedialog.asksaveasfilename(
            title="Export log",
            defaultextension=".txt",
            initialfile=f"radiolarian_log_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            content = self.txt_log.get('1.0', tk.END)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.controller.log(f"Log exported to {path}", "success")
        except OSError as e:
            self.controller.log(f"Export failed: {e}", "error")

    def import_pdfs(self):
        files = filedialog.askopenfilenames(
            title="Select PDF files to import",
            initialdir=self._last_pdf_dir,
            filetypes=PDF_FILETYPES,
        )
        if files:
            # Remember directory for next time
            self._last_pdf_dir = os.path.dirname(files[0])
            count = 0
            failed = []
            for src in files:
                try:
                    dst = sanitize_filename(os.path.basename(src))
                    shutil.copy(src, os.path.join(Config.UPLOAD_FOLDER, dst))
                    count += 1
                except Exception as e:
                    failed.append(os.path.basename(src))
            if failed:
                self.controller.log(f"Failed to import: {', '.join(failed)}", "warn")
            self.controller.log(f"Imported {count} files.", "success")
            self.update_count()

    def import_pdfs_folder(self):
        """Import every PDF inside a chosen folder (optionally recursive).

        Duplicate handling:
          - Same name AND same byte-size => skip (assumed identical).
          - Same name BUT different size => rename with _2, _3, ... so
            distinct papers that happen to share a filename don't get
            silently dropped.
        """
        folder = filedialog.askdirectory(
            title="Select a folder containing PDF files",
            initialdir=self._last_pdf_dir,
        )
        if not folder:
            return
        self._last_pdf_dir = folder

        recursive = bool(self.var_recurse.get()) if hasattr(self, 'var_recurse') else True

        # Collect candidate paths
        pdfs = []
        if recursive:
            for root, _dirs, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith('.pdf'):
                        pdfs.append(os.path.join(root, f))
        else:
            try:
                pdfs = [os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if f.lower().endswith('.pdf')
                        and os.path.isfile(os.path.join(folder, f))]
            except OSError as e:
                self.controller.log(f"❌ Cannot read folder: {e}", "error")
                return

        if not pdfs:
            self.controller.log(
                f"⚠️ No PDF files found in '{folder}'"
                f"{' (recursive)' if recursive else ''}.",
                "warn"
            )
            return

        copied = 0
        renamed = 0
        skipped = 0
        failed = []
        for src in pdfs:
            try:
                dst_name = sanitize_filename(os.path.basename(src))
                dst_path = os.path.join(Config.UPLOAD_FOLDER, dst_name)

                if os.path.exists(dst_path):
                    # Compare sizes to distinguish "true duplicate" from
                    # "different paper with same filename".  Same size
                    # is a strong but not perfect signal; good enough to
                    # avoid pathological collisions without hashing
                    # potentially-large PDFs.
                    try:
                        same_size = os.path.getsize(src) == os.path.getsize(dst_path)
                    except OSError:
                        same_size = False

                    if same_size:
                        skipped += 1
                        continue

                    # Different file with same name → find a free slot.
                    stem, ext = os.path.splitext(dst_name)
                    new_path = dst_path
                    for i in range(2, 1001):
                        candidate = os.path.join(
                            Config.UPLOAD_FOLDER, f"{stem}_{i}{ext}")
                        if not os.path.exists(candidate):
                            new_path = candidate
                            break
                    else:
                        # Exhausted 1000 names — give up on this file.
                        failed.append(os.path.basename(src))
                        continue
                    shutil.copy(src, new_path)
                    renamed += 1
                else:
                    shutil.copy(src, dst_path)
                    copied += 1
            except Exception:
                failed.append(os.path.basename(src))

        # Build a status message
        parts = [f"Imported {copied} file(s) from folder."]
        if renamed:
            parts.append(f"Renamed {renamed} same-name-but-different-content file(s).")
        if skipped:
            parts.append(f"Skipped {skipped} duplicate(s).")
        if failed:
            preview = ', '.join(failed[:5])
            more = '' if len(failed) <= 5 else f' (+{len(failed) - 5} more)'
            self.controller.log(f"Failed: {preview}{more}", "warn")
        self.controller.log(' '.join(parts),
                            "success" if (copied or renamed) else "warn")
        self.update_count()

    def update_count(self):
        try:
            # Case-insensitive match (consistent with PdfConverter's scan)
            n = len([f for f in os.listdir(Config.UPLOAD_FOLDER)
                     if f.lower().endswith('.pdf')])
        except FileNotFoundError:
            n = 0
        except Exception:
            n = 0
        self.lbl_count.config(text=f"{n} files ready")

    def get_root_frame(self):
        return self.frame

    # ---------------------------------------------------------
    # Drag-and-drop
    # ---------------------------------------------------------
    def _setup_drag_and_drop(self):
        """Register the Pipeline panel as a drop target for files /
        folders.  Uses tkinterdnd2 if available; otherwise this method
        is a no-op and users fall back to the Import buttons."""
        if not DND_AVAILABLE:
            return
        try:
            # tkinterdnd2 requires the root window to be a TkinterDnD.Tk
            # instance.  Our main app inherits from tk.Tk, so we attempt
            # a soft registration which works when tkinterdnd2 has been
            # imported (the constants are still available even if the
            # root isn't a TkinterDnD-aware Tk).
            from tkinterdnd2 import DND_FILES
            widget = self.frame  # ScrollableFrame is a ttk.Frame
            # drop_target_register may not exist on plain ttk widgets if
            # the root isn't TkinterDnD.Tk — guard with hasattr.
            if hasattr(widget, 'drop_target_register'):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            # Any failure here is non-fatal — just log so the developer
            # knows DnD didn't initialise.
            self.controller.log(
                "ℹ️ Drag-and-drop unavailable (install tkinterdnd2 and use "
                "TkinterDnD.Tk root to enable).", "info")

    def _on_drop(self, event):
        """Handle a file/folder drop on the Pipeline panel."""
        # event.data is a Tcl list of paths; some have curly braces around
        # paths with spaces.  splitlist handles both correctly.
        try:
            paths = self.frame.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        self._import_paths(paths)

    def _import_paths(self, paths):
        """Shared import logic used by drag-and-drop and (could be reused
        by command-line)."""
        copied = 0
        failed = []
        for p in paths:
            if os.path.isdir(p):
                # Walk the folder for PDFs
                for root, _dirs, files in os.walk(p):
                    for f in files:
                        if f.lower().endswith('.pdf'):
                            try:
                                dst = sanitize_filename(f)
                                shutil.copy(
                                    os.path.join(root, f),
                                    os.path.join(Config.UPLOAD_FOLDER, dst))
                                copied += 1
                            except Exception:
                                failed.append(f)
            elif os.path.isfile(p) and p.lower().endswith('.pdf'):
                try:
                    dst = sanitize_filename(os.path.basename(p))
                    shutil.copy(p, os.path.join(Config.UPLOAD_FOLDER, dst))
                    copied += 1
                except Exception:
                    failed.append(os.path.basename(p))
        if copied:
            self.controller.log(f"Drop-imported {copied} file(s).", "success")
        if failed:
            self.controller.log(f"Failed: {', '.join(failed[:5])}", "warn")
        self.update_count()


class GalleryPanel:
    """A pagination-based image viewer."""

    SORT_OPTIONS = ('Newest first', 'Oldest first', 'Name A→Z', 'Name Z→A')

    def __init__(self, parent, controller=None):
        self.controller = controller
        self.fonts = getattr(controller, 'fonts', None) if controller else None
        self.frame = ttk.Frame(parent, style='Page.TFrame')

        # ---- Toolbar (single row, light card-style) ----
        tb_card = Card(self.frame)
        tb_card.pack(fill=tk.X,
                     padx=Spacing.PAGE_PAD,
                     pady=(Spacing.PAGE_PAD, Spacing.CARD_GAP))
        tb = tk.Frame(tb_card.body, bg=Colors.BG_CARD)
        tb.pack(fill=tk.X,
                padx=Spacing.CARD_PAD_X,
                pady=(Spacing.CARD_PAD_Y, Spacing.CARD_PAD_Y))

        self.btn_refresh = ttk.Button(
            tb, text=emoji_or_text("↻", "Refresh"),
            style='Toolbar.TButton', command=self.reload, width=8)
        self.btn_refresh.pack(side=tk.LEFT)
        ToolTip(self.btn_refresh, text="Re-scan candidate images")

        self.btn_prev = ttk.Button(
            tb, text="← Prev", style='Toolbar.TButton',
            command=lambda: self.change_page(-1))
        self.btn_prev.pack(side=tk.LEFT, padx=(Spacing.FIELD_GAP, 4))

        self.btn_next = ttk.Button(
            tb, text="Next →", style='Toolbar.TButton',
            command=lambda: self.change_page(1))
        self.btn_next.pack(side=tk.LEFT)

        self.lbl_page = ttk.Label(tb, text="Page 1 / 1",
                                  style='CardMuted.TLabel')
        self.lbl_page.pack(side=tk.LEFT, padx=Spacing.SECTION_PAD)

        # Right side: sort selector + Open Folder
        self.btn_open = ttk.Button(
            tb, text="Open Folder", style='Toolbar.TButton',
            command=lambda: open_path(Config.CROPPED_FOLDER))
        self.btn_open.pack(side=tk.RIGHT)

        self.var_sort = tk.StringVar(value=self.SORT_OPTIONS[0])
        self.combo_sort = ttk.Combobox(
            tb, textvariable=self.var_sort,
            values=self.SORT_OPTIONS, state='readonly', width=14)
        self.combo_sort.pack(side=tk.RIGHT, padx=(0, Spacing.FIELD_GAP))
        self.combo_sort.bind("<<ComboboxSelected>>",
                             lambda e: self.reload())
        ttk.Label(tb, text="Sort:",
                  style='CardMuted.TLabel').pack(
            side=tk.RIGHT, padx=(0, 4))

        # ---- Scrollable image area (Card with embedded scroll) ----
        grid_card = Card(self.frame)
        grid_card.pack(fill=tk.BOTH, expand=True,
                       padx=Spacing.PAGE_PAD,
                       pady=(0, Spacing.PAGE_PAD))

        canvas_holder = grid_card.body

        self.canvas = tk.Canvas(canvas_holder, bg=Colors.BG_CARD,
                                highlightthickness=0, bd=0)
        self.content = tk.Frame(self.canvas, bg=Colors.BG_CARD)
        vbar = ttk.Scrollbar(canvas_holder, command=self.canvas.yview)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                         padx=Spacing.CARD_PAD_X,
                         pady=Spacing.CARD_PAD_Y)
        vbar.pack(side=tk.RIGHT, fill=tk.Y,
                  pady=Spacing.CARD_PAD_Y)
        self.canvas.configure(yscrollcommand=vbar.set)
        self._canvas_window = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>",
                          lambda e: self.canvas.configure(
                              scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfig(
                             self._canvas_window, width=e.width))
        # Mouse-wheel scrolling on the canvas
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

        self.images = []
        self.page = 1
        self.per_page = 20
        self.img_refs = []  # GC protection
        self._render_token = 0

    # ---- Mouse wheel (delegates to ScrollableFrame's pattern) ----
    def _bind_wheel(self, _e):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _e):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event):
        try:
            first, last = self.canvas.yview()
        except tk.TclError:
            return
        if first <= 0.0 and last >= 1.0:
            return
        if getattr(event, 'num', 0) == 4:
            delta = -1
        elif getattr(event, 'num', 0) == 5:
            delta = 1
        else:
            if event.delta == 0:
                return
            delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta, "units")

    def interactive_widgets(self):
        """Widgets to disable while a background task is running."""
        return [self.btn_refresh, self.btn_prev, self.btn_next,
                self.btn_open, self.combo_sort]

    def reload(self):
        # Use os.scandir so we get mtime + filename in ONE syscall per file.
        entries = []
        try:
            with os.scandir(Config.CROPPED_FOLDER) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    if not entry.name.lower().endswith('.jpg'):
                        continue
                    try:
                        entries.append((entry.stat().st_mtime, entry.name))
                    except OSError:
                        continue
            # Apply user-chosen sort order
            sort_choice = self.var_sort.get() if hasattr(self, 'var_sort') else 'Newest first'
            if sort_choice == 'Newest first':
                entries.sort(key=lambda t: t[0], reverse=True)
            elif sort_choice == 'Oldest first':
                entries.sort(key=lambda t: t[0])
            elif sort_choice == 'Name A→Z':
                entries.sort(key=lambda t: t[1])
            elif sort_choice == 'Name Z→A':
                entries.sort(key=lambda t: t[1], reverse=True)
            self.images = [name for _, name in entries]
        except FileNotFoundError:
            self.images = []
        except Exception:
            self.images = []
        self.page = 1
        self.render()

    def change_page(self, d):
        max_p = math.ceil(len(self.images) / self.per_page) or 1
        new_p = self.page + d
        if 1 <= new_p <= max_p:
            self.page = new_p
            self.render()

    def render(self):
        """Render the current page of thumbnails.

        Image loading is CHUNKED via `after()` so the UI stays responsive
        even when loading 20 large JPEGs.  We render one image at a time
        and yield control back to the Tk event loop between each, which
        prevents the multi-hundred-millisecond freeze a synchronous
        implementation would cause on slower disks.
        """
        for w in self.content.winfo_children(): w.destroy()
        self.img_refs = []

        start = (self.page - 1) * self.per_page
        batch = self.images[start: start + self.per_page]

        # ----- Empty state -----
        if not batch:
            empty = tk.Frame(self.content, bg=Colors.BG_CARD)
            empty.pack(expand=True, pady=60)
            # A clean glyph (not emoji — universal coverage)
            tk.Label(empty, text="○",
                     bg=Colors.BG_CARD, fg=Colors.BORDER_MED,
                     font=(self.fonts.APP_TITLE[0] if self.fonts
                           else 'TkDefaultFont', 48)).pack()
            tk.Label(empty, text="No candidate images yet",
                     bg=Colors.BG_CARD, fg=Colors.TEXT_HEAD,
                     font=(self.fonts.CARD_TITLE if self.fonts
                           else ('TkDefaultFont', 11, 'bold'))).pack(
                pady=(8, 2))
            tk.Label(empty, text="Run detection (Step 03) to populate the gallery.",
                     bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED,
                     font=(self.fonts.MUTED if self.fonts
                           else ('TkDefaultFont', 9))).pack()
            total = len(self.images)
            mp = math.ceil(total / self.per_page) or 1
            self.lbl_page.config(text=f"Page {self.page} / {mp}  (0 total)")
            return

        cols = 5
        total_items = len(batch)
        total_rows = (total_items + cols - 1) // cols
        items_in_last_row = total_items % cols
        if items_in_last_row == 0:
            items_in_last_row = cols
        last_row_idx = total_rows - 1
        col_offset = (cols - items_in_last_row) // 2

        total = len(self.images)
        mp = math.ceil(total / self.per_page) or 1
        self.lbl_page.config(
            text=f"Page {self.page} / {mp}  ({total} total) · loading…")

        # Make columns equal width
        for c in range(cols):
            self.content.columnconfigure(c, weight=1, uniform='gallery_cols')

        self._render_token += 1
        ctx = {
            'token': self._render_token,
            'batch': batch,
            'idx': 0,
            'cols': cols,
            'last_row_idx': last_row_idx,
            'items_in_last_row': items_in_last_row,
            'col_offset': col_offset,
            'mp': mp,
            'total': total,
        }
        self.frame.after(0, self._render_one, ctx)

    def _render_one(self, ctx):
        """Load and place a single thumbnail, then schedule the next one."""
        if ctx['token'] != self._render_token:
            return
        idx = ctx['idx']
        batch = ctx['batch']
        if idx >= len(batch):
            self.lbl_page.config(
                text=f"Page {self.page} / {ctx['mp']}  ({ctx['total']} total)")
            return

        fname = batch[idx]
        path = os.path.join(Config.CROPPED_FOLDER, fname)
        try:
            with Image.open(path) as pil:
                pil.thumbnail((140, 140))
                tk_img = ImageTk.PhotoImage(pil)
            self.img_refs.append(tk_img)

            # ----- Thumbnail card (1px border) -----
            # Outer border simulator
            outer = tk.Frame(self.content, bg=Colors.BORDER_LIGHT,
                             highlightthickness=0, bd=0)
            inner = tk.Frame(outer, bg=Colors.BG_CARD,
                             highlightthickness=0, bd=0)
            inner.pack(fill='both', expand=True, padx=1, pady=1)

            # Image area (white bg, centred)
            img_area = tk.Frame(inner, bg=Colors.BG_CARD, height=150)
            img_area.pack(fill='x')
            img_lbl = tk.Label(img_area, image=tk_img,
                               bg=Colors.BG_CARD, bd=0,
                               highlightthickness=0)
            img_lbl.pack(pady=4)

            # Caption (filename) — always visible
            stem = os.path.splitext(fname)[0]
            cap_lbl = tk.Label(
                inner, text=stem,
                bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED,
                font=(self.fonts.MONO_SMALL if self.fonts
                      else ('TkFixedFont', 8)),
                padx=4, pady=4)
            cap_lbl.pack(fill='x')

            # Hover: border colour changes to primary
            def on_enter(_e=None, w=outer):
                w.config(bg=Colors.PRIMARY)
            def on_leave(_e=None, w=outer):
                w.config(bg=Colors.BORDER_LIGHT)

            for w in (outer, inner, img_area, img_lbl, cap_lbl):
                w.bind("<Enter>", on_enter)
                w.bind("<Leave>", on_leave)
                w.bind("<Double-1>",
                       lambda e, p=path: self._open_file(p))

            # Grid placement
            row = idx // ctx['cols']
            col = idx % ctx['cols']
            if (row == ctx['last_row_idx']
                    and ctx['items_in_last_row'] < ctx['cols']):
                outer.grid(row=row, column=col + ctx['col_offset'],
                           padx=6, pady=6, sticky='nsew')
            else:
                outer.grid(row=row, column=col,
                           padx=6, pady=6, sticky='nsew')

            ToolTip(img_lbl, text=f"{fname}\nDouble-click to open")
        except Exception:
            pass

        ctx['idx'] += 1
        self.frame.after(0, self._render_one, ctx)

    def _open_file(self, path):
        """Open a file with the system's default application.
        Delegates to utils.open_path (which uses subprocess on Mac/Linux,
        not os.system, to avoid shell-injection on weird filenames)."""
        open_path(path)

    def get_root_frame(self):
        return self.frame