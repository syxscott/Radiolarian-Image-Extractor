# -*- coding: utf-8 -*-
"""
@file: ui_main.py
@description: The main application window and controller.
              Manages the thread pool, UI updates, and tab orchestration.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
from datetime import datetime

from config import Config
from backend_signals import WorkerSignals
from backend_tasks import PdfConverter, ObjectDetector, ImageClassifier
from ui_panels import ProcessingPanel, GalleryPanel
from ui_platform import apply_tk_scaling, get_ui_font, get_mono_font
from ui_theme import Colors, Spacing, get_fonts
from ui_widgets import SectionHeader, HSeparator, VSeparator
from utils import clear_cache_directories


# Cap log messages drained per poll cycle so a burst of 1000 logs doesn't
# lock the main thread for several seconds.
MAX_LOGS_PER_CYCLE = 50
# Hard cap on log Text widget line count to bound memory.
MAX_LOG_LINES = 5000
LOG_TRIM_CHUNK = 1000   # delete this many oldest lines when we exceed the cap


class RadiolarianApp(tk.Tk):
    """
    Main Application Window class.
    """

    def __init__(self):
        super().__init__()
        self.title("Radiolarian Image Extraction System")
        self.geometry("1100x750")
        self.minsize(900, 600)

        # Tk scaling for HiDPI (Win DPI awareness is applied in main.py
        # before the Tk root is created).
        apply_tk_scaling(self)

        self._set_window_icon()

        # Initialize Backend Infrastructure
        Config.ensure_directories()

        # Thread Communication
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.signals = WorkerSignals(self.log_queue, self.progress_queue)
        self.current_worker = None
        self.worker_logic = None

        # Tk fires <<NotebookTabChanged>> once during initial tab population.
        # We ignore it until the UI is fully ready.
        self._ui_ready = False

        self._configure_styles()
        self._build_layout()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(100, self._mark_ready)
        self.after(100, self._process_queues)

    # ---------------------------------------------------------
    # Initial setup helpers
    # ---------------------------------------------------------
    def _mark_ready(self):
        """Allow tab-change events to actually do work."""
        self._ui_ready = True

    def _set_window_icon(self):
        """Try a few common icon locations; silently ignore if none found.
        Supports both .ico (Win) and .png (cross-platform via iconphoto)."""
        base = Config.BASE_DIR
        # 1. Native .ico for Windows
        if sys.platform.startswith('win'):
            ico = os.path.join(base, 'icon.ico')
            if os.path.exists(ico):
                try:
                    self.iconbitmap(ico)
                    return
                except tk.TclError:
                    pass
        # 2. .png via iconphoto (works on Win/Linux/macOS)
        png = os.path.join(base, 'icon.png')
        if os.path.exists(png):
            try:
                img = tk.PhotoImage(file=png)
                # Keep a reference so it isn't GC'd
                self._icon_ref = img
                self.iconphoto(True, img)
                return
            except tk.TclError:
                pass
        # 3. No icon file — accept the Tk default rather than crash

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Platform-aware font picking
        ui_font = get_ui_font()
        mono_font = get_mono_font()
        self._ui_font = ui_font
        self._mono_font = mono_font

        # Build the fonts namespace and stash it for child panels to use
        self.fonts = get_fonts(ui_font, mono_font)
        F = self.fonts

        # =====================================================
        # Global defaults — sets the page colour for any default ttk widget
        # =====================================================
        self.configure(bg=Colors.BG_PAGE)
        style.configure('.', background=Colors.BG_PAGE,
                        foreground=Colors.TEXT_BODY,
                        font=F.BODY)

        # =====================================================
        # Frames — page, card, sidebar, status bar
        # =====================================================
        style.configure('Page.TFrame',      background=Colors.BG_PAGE)
        style.configure('Card.TFrame',      background=Colors.BG_CARD)
        style.configure('Sidebar.TFrame',   background=Colors.BG_SIDEBAR)
        style.configure('StatusBar.TFrame', background=Colors.BG_STATUSBAR)
        style.configure('Toolbar.TFrame',   background=Colors.BG_CARD)

        # =====================================================
        # Labels — by context
        # =====================================================
        style.configure('AppTitle.TLabel',
                        background=Colors.BG_SIDEBAR,
                        foreground=Colors.TEXT_HEAD,
                        font=F.APP_TITLE)
        style.configure('AppSubtitle.TLabel',
                        background=Colors.BG_SIDEBAR,
                        foreground=Colors.TEXT_MUTED,
                        font=F.APP_SUBTITLE)
        style.configure('SidebarKey.TLabel',
                        background=Colors.BG_SIDEBAR,
                        foreground=Colors.TEXT_MUTED,
                        font=F.SIDEBAR_KEY,
                        anchor='w')
        style.configure('SidebarValue.TLabel',
                        background=Colors.BG_SIDEBAR,
                        foreground=Colors.TEXT_HEAD,
                        font=F.SIDEBAR_VALUE,
                        anchor='e')
        style.configure('CardBody.TLabel',
                        background=Colors.BG_CARD,
                        foreground=Colors.TEXT_BODY,
                        font=F.BODY)
        style.configure('CardMuted.TLabel',
                        background=Colors.BG_CARD,
                        foreground=Colors.TEXT_MUTED,
                        font=F.MUTED)
        style.configure('CardValue.TLabel',
                        background=Colors.BG_CARD,
                        foreground=Colors.TEXT_HEAD,
                        font=F.MONO_VALUE)
        style.configure('Status.TLabel',
                        background=Colors.BG_STATUSBAR,
                        foreground=Colors.TEXT_BODY,
                        font=F.STATUS,
                        padding=(Spacing.STATUS_PAD_X, Spacing.STATUS_PAD_Y))
        style.configure('StatusValue.TLabel',
                        background=Colors.BG_STATUSBAR,
                        foreground=Colors.TEXT_HEAD,
                        font=F.STATUS_VALUE,
                        padding=(Spacing.STATUS_PAD_X, Spacing.STATUS_PAD_Y))

        # =====================================================
        # Buttons — Primary / Secondary / Danger
        # =====================================================
        # Primary (the green "Start X" buttons in the original code use
        # this).  Now a deep academic blue — no green, no teal.
        style.configure('Accent.TButton',
                        background=Colors.PRIMARY,
                        foreground=Colors.TEXT_ON_DARK,
                        font=F.BODY_BOLD,
                        padding=(Spacing.BTN_PAD_X, Spacing.BTN_PAD_Y),
                        relief='flat',
                        borderwidth=0)
        style.map('Accent.TButton',
                  background=[('active',   Colors.PRIMARY_HOVER),
                              ('pressed',  Colors.PRIMARY_ACTIVE),
                              ('disabled', Colors.DISABLED_BG)],
                  foreground=[('disabled', Colors.DISABLED_FG)])

        # Secondary (everything else — flat white with thin border)
        style.configure('TButton',
                        background=Colors.BG_CARD,
                        foreground=Colors.TEXT_BODY,
                        font=F.BODY,
                        padding=(Spacing.BTN_PAD_X, Spacing.BTN_PAD_Y),
                        relief='flat',
                        bordercolor=Colors.BORDER_MED,
                        borderwidth=1)
        style.map('TButton',
                  background=[('active',   Colors.BG_HOVER_DARK),
                              ('pressed',  Colors.BG_HOVER_DARK),
                              ('disabled', Colors.DISABLED_BG)],
                  foreground=[('disabled', Colors.DISABLED_FG)],
                  bordercolor=[('active', Colors.BORDER_MED)])

        # Toolbar buttons (smaller, used in log/gallery toolbars)
        style.configure('Toolbar.TButton',
                        background=Colors.BG_CARD,
                        foreground=Colors.TEXT_BODY,
                        font=F.BODY_SMALL,
                        padding=(8, 3),
                        relief='flat',
                        bordercolor=Colors.BORDER_LIGHT,
                        borderwidth=1)
        style.map('Toolbar.TButton',
                  background=[('active', Colors.BG_HOVER_DARK)])

        # Danger (STOP) — muted dark red, prominent only when enabled
        style.configure('Stop.TButton',
                        background=Colors.DANGER,
                        foreground=Colors.TEXT_ON_DARK,
                        font=F.BODY_BOLD,
                        padding=(Spacing.BTN_PAD_X, Spacing.BTN_PAD_Y),
                        relief='flat',
                        borderwidth=0)
        style.map('Stop.TButton',
                  background=[('active',   Colors.DANGER_HOVER),
                              ('pressed',  Colors.DANGER_HOVER),
                              ('disabled', Colors.DISABLED_BG)],
                  foreground=[('disabled', Colors.DISABLED_FG)])

        # =====================================================
        # Inputs — Combobox / Entry / Scale / Checkbutton / Progressbar
        # =====================================================
        style.configure('TCombobox',
                        fieldbackground=Colors.BG_CARD,
                        background=Colors.BG_CARD,
                        foreground=Colors.TEXT_BODY,
                        bordercolor=Colors.BORDER_MED,
                        lightcolor=Colors.BORDER_MED,
                        darkcolor=Colors.BORDER_MED,
                        arrowcolor=Colors.TEXT_MUTED,
                        padding=4)
        style.map('TCombobox',
                  fieldbackground=[('readonly', Colors.BG_CARD),
                                   ('disabled', Colors.DISABLED_BG)],
                  foreground=[('disabled', Colors.DISABLED_FG)],
                  bordercolor=[('focus', Colors.BORDER_FOCUS)])

        style.configure('TEntry',
                        fieldbackground=Colors.BG_CARD,
                        foreground=Colors.TEXT_BODY,
                        bordercolor=Colors.BORDER_MED,
                        lightcolor=Colors.BORDER_MED,
                        darkcolor=Colors.BORDER_MED,
                        insertcolor=Colors.TEXT_HEAD,
                        padding=4)

        style.configure('TCheckbutton',
                        background=Colors.BG_CARD,
                        foreground=Colors.TEXT_BODY,
                        font=F.BODY)
        style.map('TCheckbutton',
                  background=[('active', Colors.BG_HOVER)])

        style.configure('Horizontal.TScale',
                        background=Colors.BG_CARD,
                        troughcolor=Colors.BG_HOVER_DARK,
                        bordercolor=Colors.BORDER_LIGHT,
                        lightcolor=Colors.PRIMARY,
                        darkcolor=Colors.PRIMARY)

        # Progress bar — one consistent style, no per-step colours
        style.configure('Horizontal.TProgressbar',
                        background=Colors.PRIMARY,
                        troughcolor=Colors.BG_HOVER_DARK,
                        bordercolor=Colors.BORDER_LIGHT,
                        lightcolor=Colors.PRIMARY,
                        darkcolor=Colors.PRIMARY,
                        thickness=6)

        # =====================================================
        # Notebook / LabelFrame
        # =====================================================
        style.configure('TNotebook',
                        background=Colors.BG_PAGE,
                        borderwidth=0,
                        tabposition='n')
        style.configure('TNotebook.Tab',
                        background=Colors.BG_PAGE,
                        foreground=Colors.TEXT_MUTED,
                        font=F.BODY,
                        padding=(18, 8),
                        borderwidth=0)
        style.map('TNotebook.Tab',
                  background=[('selected', Colors.BG_CARD),
                              ('active',   Colors.BG_HOVER)],
                  foreground=[('selected', Colors.TEXT_HEAD),
                              ('active',   Colors.TEXT_BODY)])

        # LabelFrame (only used by Log/Gallery — step cards are custom)
        style.configure('TLabelframe',
                        background=Colors.BG_CARD,
                        bordercolor=Colors.BORDER_LIGHT,
                        relief='flat',
                        borderwidth=1)
        style.configure('TLabelframe.Label',
                        background=Colors.BG_CARD,
                        foreground=Colors.TEXT_HEAD,
                        font=F.CARD_TITLE)

        # Make the PanedWindow sash thick enough to grab on Linux —
        # default clam sash is 1-2px which is essentially unusable.
        style.configure('TPanedwindow', background=Colors.BG_PAGE)
        style.configure('Sash', sashthickness=6, gripcount=0)

        # =====================================================
        # Scrollbar
        # =====================================================
        style.configure('Vertical.TScrollbar',
                        background=Colors.BG_PAGE,
                        troughcolor=Colors.BG_PAGE,
                        bordercolor=Colors.BG_PAGE,
                        arrowcolor=Colors.TEXT_MUTED,
                        gripcount=0)

    def _build_layout(self):
        # Pack the status bar FIRST so it always stays visible at the
        # bottom even when the window is shrunk below the natural size
        # of the paned-window contents.
        self._build_status_bar()

        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL,
                                     style='TPanedwindow')
        self.paned.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar()

        self.notebook = ttk.Notebook(self.paned, style='TNotebook')
        self.paned.add(self.notebook, weight=3)

        self.panel_proc = ProcessingPanel(self.notebook, self)
        self.panel_gallery = GalleryPanel(self.notebook, self)

        self.notebook.add(self.panel_proc.get_root_frame(),
                          text="  Pipeline  ")
        self.notebook.add(self.panel_gallery.get_root_frame(),
                          text="  Gallery  ")

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Force the sash to its desired position once Tk has computed
        # initial geometry — otherwise the paned window may give the
        # sidebar more space than its requested width.
        self.after(50, lambda: self._lock_sash_position())

        self._refresh_stats()

    def _lock_sash_position(self):
        try:
            self.paned.sashpos(0, Spacing.SIDEBAR_W)
        except tk.TclError:
            pass

    # ---------------------------------------------------------
    # Sidebar — academic dashboard style
    # ---------------------------------------------------------
    def _build_sidebar(self):
        """Layout (top → bottom):
            App title block
            ─────────────────
            STATISTICS  (live counts)
            ─────────────────
            HARDWARE    (device / GPUs / CPUs)
            ─────────────────
            (spacer)
            STOP / Clear Cache buttons
            ─────────────────
            SHORTCUTS hint
        """
        F = self.fonts

        sidebar = tk.Frame(self.paned, bg=Colors.BG_SIDEBAR,
                           highlightthickness=0, bd=0,
                           width=Spacing.SIDEBAR_W)
        self.paned.add(sidebar, weight=0)
        sidebar.pack_propagate(False)  # respect the fixed width

        pad = Spacing.SIDEBAR_PAD

        # App title block
        title_wrap = tk.Frame(sidebar, bg=Colors.BG_SIDEBAR)
        title_wrap.pack(fill='x', padx=pad, pady=(pad + 4, pad))

        ttk.Label(title_wrap, text="Radiolarian",
                  style='AppTitle.TLabel').pack(anchor='w')
        ttk.Label(title_wrap, text="Image Extraction Toolkit",
                  style='AppSubtitle.TLabel').pack(anchor='w')

        HSeparator(sidebar).pack(fill='x', padx=pad)

        # STATISTICS section
        self._stats_section = tk.Frame(sidebar, bg=Colors.BG_SIDEBAR)
        self._stats_section.pack(fill='x',
                                 padx=pad,
                                 pady=(Spacing.SECTION_GAP, pad))

        SectionHeader(self._stats_section, "Statistics",
                      fonts=F).pack(fill='x', pady=(0, 6))

        # Keep references to the value labels so we can update them
        self._stat_labels = {}
        for key in ('PDFs', 'Pages', 'Crops', 'Rejected'):
            row = tk.Frame(self._stats_section, bg=Colors.BG_SIDEBAR)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text=key,
                      style='SidebarKey.TLabel').pack(side='left')
            lbl = ttk.Label(row, text='—', style='SidebarValue.TLabel')
            lbl.pack(side='right')
            self._stat_labels[key] = lbl

        HSeparator(sidebar).pack(fill='x', padx=pad)

        # HARDWARE section
        hw_section = tk.Frame(sidebar, bg=Colors.BG_SIDEBAR)
        hw_section.pack(fill='x', padx=pad,
                        pady=(Spacing.SECTION_GAP, pad))

        SectionHeader(hw_section, "Hardware", fonts=F).pack(
            fill='x', pady=(0, 6))

        hw_values = {
            'Device':  'CUDA' if Config.NUM_GPUS > 0 else 'CPU',
            'GPUs':    str(Config.NUM_GPUS),
            'CPUs':    str(Config.CPU_COUNT),
        }
        for key, val in hw_values.items():
            row = tk.Frame(hw_section, bg=Colors.BG_SIDEBAR)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text=key,
                      style='SidebarKey.TLabel').pack(side='left')
            ttk.Label(row, text=val,
                      style='SidebarValue.TLabel').pack(side='right')

        HSeparator(sidebar).pack(fill='x', padx=pad)

        # SHORTCUTS hint — packed at BOTTOM so action buttons sit above it
        bottom_wrap = tk.Frame(sidebar, bg=Colors.BG_SIDEBAR)
        bottom_wrap.pack(side='bottom', fill='x',
                         padx=pad, pady=(pad, pad))

        SectionHeader(bottom_wrap, "Shortcuts", fonts=F).pack(
            fill='x', pady=(0, 6))

        sc_text = ("Ctrl+I    Import\n"
                   "F5         Refresh\n"
                   "Esc        Stop\n"
                   "Ctrl+Q   Quit")
        tk.Label(bottom_wrap, text=sc_text,
                 bg=Colors.BG_SIDEBAR, fg=Colors.TEXT_MUTED,
                 font=F.MONO_SMALL, justify='left',
                 padx=0, pady=0).pack(anchor='w')

        HSeparator(sidebar).pack(side='bottom', fill='x', padx=pad)

        # Action buttons (above shortcuts, below stats spacer)
        actions_wrap = tk.Frame(sidebar, bg=Colors.BG_SIDEBAR)
        actions_wrap.pack(side='bottom', fill='x',
                          padx=pad, pady=(pad, Spacing.SECTION_GAP))

        self.btn_stop = ttk.Button(actions_wrap, text="Stop Task",
                                   style='Stop.TButton',
                                   command=self.stop_current_task,
                                   state='disabled')   # only enabled when busy
        self.btn_stop.pack(fill='x', pady=(0, 6))

        self.btn_clear = ttk.Button(actions_wrap, text="Clear Cache",
                                    command=self.clear_cache)
        self.btn_clear.pack(fill='x')

    # ---------------------------------------------------------
    # Multi-cell status bar (VS-Code-style bottom strip)
    # ---------------------------------------------------------
    def _build_status_bar(self):
        F = self.fonts if hasattr(self, 'fonts') else None

        bar = tk.Frame(self, bg=Colors.BG_STATUSBAR,
                       highlightthickness=0, bd=0)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        # A thin top border so the bar visually detaches from the page
        top_sep = tk.Frame(bar, height=1, bg=Colors.BORDER_LIGHT)
        top_sep.pack(side='top', fill='x')

        inner = tk.Frame(bar, bg=Colors.BG_STATUSBAR)
        inner.pack(side='top', fill='x')

        # Cell 1 — current task status (live)
        self.status_var = tk.StringVar(value="● Ready")
        self._lbl_status = ttk.Label(inner, textvariable=self.status_var,
                                     style='Status.TLabel')
        self._lbl_status.pack(side='left')

        VSeparator(inner).pack(side='left', fill='y')

        # Cell 2 — live counts
        self.statbar_counts_var = tk.StringVar(value="— PDFs · — crops")
        ttk.Label(inner, textvariable=self.statbar_counts_var,
                  style='Status.TLabel').pack(side='left')

        VSeparator(inner).pack(side='left', fill='y')

        # Cell 3 — hardware
        hw_text = (f"{'CUDA' if Config.NUM_GPUS > 0 else 'CPU'} · "
                   f"{Config.NUM_GPUS} GPU · {Config.CPU_COUNT} CPU")
        ttk.Label(inner, text=hw_text, style='Status.TLabel').pack(side='left')

        # Cell 4 — clock (right-aligned)
        self.statbar_clock_var = tk.StringVar(value="")
        ttk.Label(inner, textvariable=self.statbar_clock_var,
                  style='StatusValue.TLabel').pack(side='right')

        # Tick the clock
        self._tick_clock()

    def _tick_clock(self):
        self.statbar_clock_var.set(datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ---------------------------------------------------------
    # Live stats — sidebar + status-bar synchronised
    # ---------------------------------------------------------
    def _refresh_stats(self):
        """Recompute file-system counts and push them to both the
        sidebar Statistics panel and the bottom status-bar."""
        try:
            n_pdfs = sum(1 for f in os.listdir(Config.UPLOAD_FOLDER)
                         if f.lower().endswith('.pdf'))
        except OSError:
            n_pdfs = 0
        try:
            n_pages = sum(1 for f in os.listdir(Config.PAGES_FOLDER)
                          if f.lower().endswith(('.jpg', '.png')))
        except OSError:
            n_pages = 0
        try:
            n_crops = sum(1 for f in os.listdir(Config.CROPPED_FOLDER)
                          if f.lower().endswith('.jpg'))
        except OSError:
            n_crops = 0
        try:
            n_rej = sum(1 for f in os.listdir(Config.REJECTED_FOLDER)
                        if f.lower().endswith('.jpg'))
        except OSError:
            n_rej = 0

        # Sidebar
        for k, v in (('PDFs', n_pdfs), ('Pages', n_pages),
                     ('Crops', n_crops), ('Rejected', n_rej)):
            if k in self._stat_labels:
                self._stat_labels[k].config(text=f"{v:,}")

        # Status bar
        self.statbar_counts_var.set(
            f"{n_pdfs:,} PDFs · {n_crops:,} crops · {n_rej:,} rejected")

    def _bind_shortcuts(self):
        """Keyboard shortcuts.  Bound at the toplevel so they work no
        matter which widget has focus."""
        self.bind_all("<Control-q>", lambda e: self._on_close())
        self.bind_all("<Control-Q>", lambda e: self._on_close())
        self.bind_all("<Escape>", lambda e: self.stop_current_task())
        self.bind_all("<F5>", lambda e: self._refresh_all())
        self.bind_all("<Control-i>", lambda e: self.panel_proc.import_pdfs())
        self.bind_all("<Control-I>", lambda e: self.panel_proc.import_pdfs())

    def _refresh_all(self):
        """Refresh everything (model lists + gallery + stats)."""
        self.panel_proc._refresh_yolo_models(silent=True)
        self.panel_proc._refresh_classifier_models(silent=True)
        self.panel_proc.update_count()
        self.panel_gallery.reload()
        self._refresh_stats()
        self.log("Refreshed model lists, gallery and stats.", "info")

    # ---------------------------------------------------------
    # Threading Logic
    # ---------------------------------------------------------
    def start_conversion(self, dpi):
        self._run_task(
            PdfConverter(self.signals, dpi),
            status=f"● Converting PDFs (DPI={dpi})…",
            step_key='convert',
        )

    def start_detection(self, model_name, model_path, conf):
        """model_name: display name; model_path: absolute path on disk."""
        self._run_task(
            ObjectDetector(self.signals, model_name, model_path, float(conf)),
            status=f"● Detecting fossils ({model_name}, conf={conf})…",
            step_key='detect',
        )

    def start_classification(self, model_name, model_path, arch_key):
        """model_name: display name; model_path: absolute path; arch_key: matched arch."""
        self._run_task(
            ImageClassifier(self.signals, model_name, model_path, arch_key),
            status=f"● Filtering candidates ({model_name})…",
            step_key='classify',
        )

    def _run_task(self, worker_instance, status="● Processing…", step_key=None):
        if self.current_worker and self.current_worker.is_alive():
            messagebox.showwarning("Busy", "A task is already running.")
            return

        self.worker_logic = worker_instance
        self.current_worker = threading.Thread(target=worker_instance.run,
                                               daemon=True)
        self._current_step_key = step_key   # remembered for completion update

        # Mark the corresponding step as running
        if step_key:
            self.panel_proc.set_step_status(step_key, 'running')

        self.current_worker.start()

        self._toggle_ui(False)
        self.status_var.set(status)

        # Enable the STOP button (disabled by default)
        try:
            self.btn_stop.config(state='normal')
        except tk.TclError:
            pass

    def stop_current_task(self):
        if self.worker_logic:
            self.worker_logic.request_stop()
            self.log("Stopping task…", "warn")
            self.status_var.set("● Stopping…")

    def _process_queues(self):
        # Process at most MAX_LOGS_PER_CYCLE messages per cycle.  Anything
        # beyond that waits for the next poll so we never block the main
        # loop for more than ~50 short insertions at a time.
        log_count = 0
        while log_count < MAX_LOGS_PER_CYCLE and not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log(msg['text'], msg['level'])
            log_count += 1

        # Progress queue: also bounded, but progress msgs are small.
        prog_count = 0
        while prog_count < MAX_LOGS_PER_CYCLE and not self.progress_queue.empty():
            try:
                msg = self.progress_queue.get_nowait()
            except queue.Empty:
                break
            task, val = msg['task'], msg['value']
            if task == 'conversion':
                self.panel_proc.prog_convert['value'] = val
            elif task == 'detection':
                self.panel_proc.prog_detect['value'] = val
            elif task == 'classification':
                self.panel_proc.prog_classify['value'] = val
            prog_count += 1

        # Handle task finish
        if self.current_worker and not self.current_worker.is_alive():
            self.current_worker = None

            # Mark the just-finished step as done (status badge)
            if getattr(self, '_current_step_key', None):
                self.panel_proc.set_step_status(
                    self._current_step_key, 'done')
                self._current_step_key = None

            self._toggle_ui(True)
            self.status_var.set("● Ready")
            self.panel_proc.update_count()  # Refresh file counts
            # Also refresh the gallery so newly-classified results appear
            self.panel_gallery.reload()
            # Refresh sidebar + status-bar stats
            self._refresh_stats()
            # Disable STOP again
            try:
                self.btn_stop.config(state='disabled')
            except tk.TclError:
                pass

        self.after(100, self._process_queues)

    def log(self, text, level='info'):
        """Writes to the text widget in the Processing Panel.

        Trims the oldest log lines when the widget exceeds MAX_LOG_LINES
        so long-running sessions don't accumulate unbounded memory.
        """
        txt = self.panel_proc.txt_log
        txt.config(state='normal')

        # Trim old lines if we're over the cap
        try:
            num_lines = int(txt.index('end-1c').split('.')[0])
            if num_lines > MAX_LOG_LINES:
                end = f"{LOG_TRIM_CHUNK + 1}.0"
                txt.delete('1.0', end)
                txt.insert('1.0',
                           f"  [...{LOG_TRIM_CHUNK} earlier lines trimmed...]\n",
                           'MUTED')
        except (tk.TclError, ValueError):
            pass

        ts = datetime.now().strftime("%H:%M:%S")
        lvl = level.upper()
        tag = lvl if lvl in ('INFO', 'WARN', 'ERROR', 'SUCCESS') else 'INFO'

        # Time stamp in its own (muted) tag, body in level tag
        txt.insert(tk.END, f"  {ts}  ", 'TIMESTAMP')
        txt.insert(tk.END, f"{text}\n", tag)
        txt.see(tk.END)
        txt.config(state='disabled')

    def _toggle_ui(self, enable):
        """Enable / disable all interactive widgets across both tabs and
        the sidebar so a running task cannot be corrupted by another
        action (import, clear-cache, switch detection model, etc.).

        Comboboxes need to be reset to 'readonly' (not 'normal'),
        otherwise the user can type free-form text into the model
        dropdown when the task finishes.
        """
        state = 'normal' if enable else 'disabled'
        readonly_state = 'readonly' if enable else 'disabled'

        all_widgets = (self.panel_proc.interactive_widgets()
                       + self.panel_gallery.interactive_widgets()
                       + [self.btn_clear])
        for w in all_widgets:
            try:
                if isinstance(w, ttk.Combobox):
                    w.config(state=readonly_state)
                else:
                    w.config(state=state)
            except tk.TclError:
                # Some widgets (e.g. ttk.Scale) only support a subset
                # of states; ignore the ones that don't apply.
                pass

    def _on_tab_changed(self, _event):
        """Auto-refresh the Gallery when the user switches to it so they
        see the latest candidate images without a manual click.

        Ignored until the UI is fully initialised so Tk's spurious
        initial <<NotebookTabChanged>> event doesn't kick off a Gallery
        reload before things are wired up.
        """
        if not self._ui_ready:
            return
        try:
            tab_text = self.notebook.tab(self.notebook.select(), "text").strip()
        except tk.TclError:
            return
        if tab_text == "Gallery":
            self.panel_gallery.reload()

    def clear_cache(self):
        # Guard: the controller normally disables this button while a task
        # is running, but be defensive in case the click races the toggle.
        if self.current_worker and self.current_worker.is_alive():
            messagebox.showwarning("Busy",
                                   "Cannot clear cache while a task is running.")
            return
        if messagebox.askyesno("Confirm", "Delete all processed files?"):
            clear_cache_directories(
                [Config.PAGES_FOLDER, Config.CROPPED_FOLDER,
                 Config.REJECTED_FOLDER, Config.LOGS_FOLDER])
            self.log("Cache cleared.", "success")
            self.panel_proc.update_count()
            self.panel_gallery.reload()
            # Reset every step badge back to pending
            for k in ('convert', 'detect', 'classify'):
                self.panel_proc.set_step_status(k, 'pending')
            # Refresh sidebar + status-bar stats
            self._refresh_stats()
            # Re-scan model folders in case the user dropped new weights in
            self.panel_proc._refresh_yolo_models(silent=True)
            self.panel_proc._refresh_classifier_models(silent=True)

    # ---------------------------------------------------------
    # Clean shutdown
    # ---------------------------------------------------------
    def _on_close(self):
        """Handle the user closing the window.

        - If a task is running, ask for confirmation so we don't silently
          kill the worker thread and corrupt the output files.
        - Tell the worker to stop, give it a short grace period to flush,
          then force-destroy if it hasn't returned.
        """
        if self.current_worker and self.current_worker.is_alive():
            if not messagebox.askyesno(
                    "Task Running",
                    "A task is currently running.\n\n"
                    "Stop it and exit anyway?\n"
                    "(Unsaved progress may be lost.)"):
                return
            self.stop_current_task()
            self.status_var.set("Shutting down — finishing active work...")
            # Poll for shutdown; force-quit after at most 5 seconds
            self._shutdown_deadline = 50  # 50 * 100ms = 5s
            self._wait_and_destroy()
        else:
            self.destroy()

    def _wait_and_destroy(self):
        """Tail-recursive shutdown poller — gives the worker up to 5s
        to flush its file handles cleanly before we force-quit."""
        if self.current_worker is None or not self.current_worker.is_alive():
            self.destroy()
            return
        self._shutdown_deadline -= 1
        if self._shutdown_deadline <= 0:
            # Time's up — destroy anyway.  Daemon threads will be killed
            # when the interpreter exits; we accept that.
            self.destroy()
            return
        self.after(100, self._wait_and_destroy)