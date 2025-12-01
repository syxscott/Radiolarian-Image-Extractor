# -*- coding: utf-8 -*-
"""
@file: ui_main.py
@description: The main application window and controller.
              Manages the thread pool, UI updates, and tab orchestration.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
from datetime import datetime

from config import Config
from backend_signals import WorkerSignals
from backend_tasks import PdfConverter, ObjectDetector, ImageClassifier
from ui_panels import ProcessingPanel, GalleryPanel
from utils import clear_cache_directories, create_placeholder_models


class RadiolarianApp(tk.Tk):
    """
    Main Application Window class.
    """

    def __init__(self):
        super().__init__()
        self.title("Radiolarian Image Extraction System")
        self.geometry("1100x750")

        # Initialize Backend Infrastructure
        Config.ensure_directories()
        # Create dummy models if missing so app doesn't crash on start
        create_placeholder_models(list(Config.YOLO_MODELS.values()) + list(Config.CLASSIFIER_MODELS.values()))

        # Thread Communication
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.signals = WorkerSignals(self.log_queue, self.progress_queue)
        self.current_worker = None
        self.worker_logic = None

        # Setup UI
        self._configure_styles()
        self._build_layout()

        # Start Message Polling Loop
        self.after(100, self._process_queues)

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Accent.TButton', background='#00796B', foreground='white')
        style.map('Accent.TButton', background=[('active', '#004D40')])
        style.configure('Status.TLabel', background='#37474F', foreground='white', font=('Consolas', 9))

    def _build_layout(self):
        # 1. Sidebar
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(self.paned, width=200, relief=tk.RIDGE)
        self.paned.add(sidebar, weight=0)

        ttk.Label(sidebar, text="RADIOLARIAN\nTOOLKIT", font=("Segoe UI", 16, "bold"), justify=tk.CENTER).pack(pady=20)
        tk.Button(sidebar, text="STOP TASKS", bg="#D32F2F", fg="white", font=("Segoe UI", 10, "bold"),
                  command=self.stop_current_task).pack(fill=tk.X, padx=10, pady=20)

        ttk.Button(sidebar, text="Clear Cache", command=self.clear_cache).pack(fill=tk.X, padx=10, pady=5)

        # 2. Tabs
        self.notebook = ttk.Notebook(self.paned)
        self.paned.add(self.notebook, weight=3)

        self.panel_proc = ProcessingPanel(self.notebook, self)
        self.panel_gallery = GalleryPanel(self.notebook)

        self.notebook.add(self.panel_proc.get_root_frame(), text=" Pipeline ")
        self.notebook.add(self.panel_gallery.get_root_frame(), text=" Gallery ")

        # 3. Status Bar
        self.status_var = tk.StringVar(value="System Ready")
        ttk.Label(self, textvariable=self.status_var, style='Status.TLabel', padding=5).pack(side=tk.BOTTOM, fill=tk.X)

    # --- Threading Logic ---
    def start_conversion(self, dpi):
        self._run_task(PdfConverter(self.signals, dpi))

    def start_detection(self, model, conf):
        self._run_task(ObjectDetector(self.signals, model, float(conf)))

    def start_classification(self, model):
        self._run_task(ImageClassifier(self.signals, model))

    def _run_task(self, worker_instance):
        if self.current_worker and self.current_worker.is_alive():
            messagebox.showwarning("Busy", "A task is already running.")
            return

        self.worker_logic = worker_instance
        self.current_worker = threading.Thread(target=worker_instance.run, daemon=True)
        self.current_worker.start()

        self._toggle_ui(False)
        self.status_var.set("Processing...")

    def stop_current_task(self):
        if self.worker_logic:
            self.worker_logic.request_stop()
            self.log("Stopping task...", "warn")

    def _process_queues(self):
        # 1. Handle Logs
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log(msg['text'], msg['level'])

        # 2. Handle Progress
        while not self.progress_queue.empty():
            msg = self.progress_queue.get()
            task, val = msg['task'], msg['value']
            # Update specific progress bars
            if task == 'conversion':
                self.panel_proc.prog_convert['value'] = val
            elif task == 'detection':
                self.panel_proc.prog_detect['value'] = val
            elif task == 'classification':
                self.panel_proc.prog_classify['value'] = val

        # 3. Handle Task Finish
        if self.current_worker and not self.current_worker.is_alive():
            self.current_worker = None
            self._toggle_ui(True)
            self.status_var.set("Ready")
            self.panel_proc.update_count()  # Refresh file counts

        self.after(100, self._process_queues)

    def log(self, text, level='info'):
        """Writes to the text widget in the Processing Panel."""
        txt = self.panel_proc.txt_log
        txt.config(state='normal')
        ts = datetime.now().strftime("%H:%M:%S")
        tag = level.upper() if level.upper() in ['INFO', 'WARN', 'ERROR', 'SUCCESS'] else 'INFO'
        txt.insert(tk.END, f"[{ts}] {text}\n", tag)
        txt.see(tk.END)
        txt.config(state='disabled')

    def _toggle_ui(self, enable):
        state = 'normal' if enable else 'disabled'
        p = self.panel_proc
        p.btn_convert.config(state=state)
        p.btn_detect.config(state=state)
        p.btn_classify.config(state=state)

    def clear_cache(self):
        if messagebox.askyesno("Confirm", "Delete all processed files?"):
            clear_cache_directories(
                [Config.PAGES_FOLDER, Config.CROPPED_FOLDER, Config.REJECTED_FOLDER, Config.LOGS_FOLDER])
            self.log("Cache cleared.", "success")
            self.panel_proc.update_count()
            self.panel_gallery.reload()