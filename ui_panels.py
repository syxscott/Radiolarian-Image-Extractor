# -*- coding: utf-8 -*-
"""
@file: ui_panels.py
@description: Implementation of the specific content tabs (Processing & Gallery).
@author: Yaxuan Shen
@date: 2025-10-01
"""

import os
import math
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

from config import Config
from ui_widgets import ToolTip, ScrollableFrame
from utils import sanitize_filename
import shutil


class ProcessingPanel:
    """The main dashboard for running the data pipeline."""

    def __init__(self, parent, controller):
        self.controller = controller  # Reference to main app to start threads
        self.frame = ScrollableFrame(parent)
        self.content = self.frame.scrollable_frame

        self._build_step1_data()
        self._build_step2_conversion()
        self._build_step3_detection()
        self._build_step4_classification()
        self._build_logs()

    def _build_step1_data(self):
        step = ttk.LabelFrame(self.content, text=" 1. Data Preparation ", padding=10)
        step.pack(fill=tk.X, padx=10, pady=5)

        btn_add = ttk.Button(step, text="Import PDF Files", command=self.import_pdfs)
        btn_open = ttk.Button(step, text="Open Source Folder", command=lambda: os.startfile(Config.UPLOAD_FOLDER))

        btn_add.pack(side=tk.LEFT, padx=5)
        btn_open.pack(side=tk.LEFT, padx=5)

        self.lbl_count = ttk.Label(step, text="0 files ready")
        self.lbl_count.pack(side=tk.LEFT, padx=15)

    def _build_step2_conversion(self):
        step = ttk.LabelFrame(self.content, text=" 2. PDF Conversion ", padding=10)
        step.pack(fill=tk.X, padx=10, pady=5)

        f = ttk.Frame(step)
        f.pack(fill=tk.X)

        ttk.Label(f, text="DPI:").pack(side=tk.LEFT)
        self.var_dpi = tk.IntVar(value=3)
        tk.Spinbox(f, from_=1, to=10, textvariable=self.var_dpi, width=5).pack(side=tk.LEFT, padx=5)

        self.btn_convert = ttk.Button(f, text="Start Conversion", style='Accent.TButton',
                                      command=lambda: self.controller.start_conversion(self.var_dpi.get()))
        self.btn_convert.pack(side=tk.RIGHT)

        self.prog_convert = ttk.Progressbar(step, mode='determinate')
        self.prog_convert.pack(fill=tk.X, pady=(10, 0))

    def _build_step3_detection(self):
        step = ttk.LabelFrame(self.content, text=" 3. Object Detection ", padding=10)
        step.pack(fill=tk.X, padx=10, pady=5)

        f = ttk.Frame(step)
        f.pack(fill=tk.X)

        ttk.Label(f, text="Model:").pack(side=tk.LEFT)
        self.combo_yolo = ttk.Combobox(f, values=list(Config.YOLO_MODELS.keys()), state="readonly")
        if Config.YOLO_MODELS: self.combo_yolo.current(0)
        self.combo_yolo.pack(side=tk.LEFT, padx=5)

        ttk.Label(f, text="Conf:").pack(side=tk.LEFT, padx=(10, 0))
        self.scale_conf = ttk.Scale(f, from_=0.1, to=1.0, value=0.25, orient=tk.HORIZONTAL)
        self.scale_conf.pack(side=tk.LEFT, padx=5)

        self.btn_detect = ttk.Button(f, text="Start Detection", style='Accent.TButton',
                                     command=lambda: self.controller.start_detection(self.combo_yolo.get(),
                                                                                     self.scale_conf.get()))
        self.btn_detect.pack(side=tk.RIGHT)

        self.prog_detect = ttk.Progressbar(step, mode='determinate')
        self.prog_detect.pack(fill=tk.X, pady=(10, 0))

    def _build_step4_classification(self):
        step = ttk.LabelFrame(self.content, text=" 4. Quality Control ", padding=10)
        step.pack(fill=tk.X, padx=10, pady=5)

        f = ttk.Frame(step)
        f.pack(fill=tk.X)

        ttk.Label(f, text="Classifier:").pack(side=tk.LEFT)
        self.combo_cls = ttk.Combobox(f, values=list(Config.CLASSIFIER_MODELS.keys()), state="readonly")
        if Config.CLASSIFIER_MODELS: self.combo_cls.current(0)
        self.combo_cls.pack(side=tk.LEFT, padx=5)

        self.btn_classify = ttk.Button(f, text="Start Filter", style='Accent.TButton',
                                       command=lambda: self.controller.start_classification(self.combo_cls.get()))
        self.btn_classify.pack(side=tk.RIGHT)

        self.prog_classify = ttk.Progressbar(step, mode='determinate')
        self.prog_classify.pack(fill=tk.X, pady=(10, 0))

    def _build_logs(self):
        lf = ttk.LabelFrame(self.content, text=" System Logs ", padding=10)
        lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.txt_log = tk.Text(lf, height=10, state='disabled', bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        # Log Tags
        self.txt_log.tag_config('INFO', foreground='#d4d4d4')
        self.txt_log.tag_config('WARN', foreground='#FFA500')
        self.txt_log.tag_config('ERROR', foreground='#FF6B6B')
        self.txt_log.tag_config('SUCCESS', foreground='#4CAF50')

    def import_pdfs(self):
        files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if files:
            count = 0
            for src in files:
                try:
                    dst = sanitize_filename(os.path.basename(src))
                    shutil.copy(src, os.path.join(Config.UPLOAD_FOLDER, dst))
                    count += 1
                except:
                    pass
            self.controller.log(f"Imported {count} files.", "success")
            self.update_count()

    def update_count(self):
        n = len([f for f in os.listdir(Config.UPLOAD_FOLDER) if f.endswith('.pdf')])
        self.lbl_count.config(text=f"{n} files ready")

    def get_root_frame(self):
        return self.frame


class GalleryPanel:
    """A pagination-based image viewer."""

    def __init__(self, parent):
        self.frame = ttk.Frame(parent)

        # Toolbar
        tb = ttk.Frame(self.frame, padding=5)
        tb.pack(fill=tk.X)
        ttk.Button(tb, text="Refresh", command=self.reload).pack(side=tk.LEFT)
        ttk.Button(tb, text="< Prev", command=lambda: self.change_page(-1)).pack(side=tk.LEFT, padx=10)
        ttk.Button(tb, text="Next >", command=lambda: self.change_page(1)).pack(side=tk.LEFT)
        self.lbl_page = ttk.Label(tb, text="Page 1/1")
        self.lbl_page.pack(side=tk.LEFT, padx=15)
        ttk.Button(tb, text="Open Folder", command=lambda: os.startfile(Config.CROPPED_FOLDER)).pack(side=tk.RIGHT)

        # Canvas
        self.canvas = tk.Canvas(self.frame, bg="white")
        self.content = ttk.Frame(self.canvas)
        vbar = ttk.Scrollbar(self.frame, command=self.canvas.yview)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=vbar.set)
        self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.images = []
        self.page = 1
        self.per_page = 20
        self.img_refs = []  # GC protection

    def reload(self):
        self.images = sorted([f for f in os.listdir(Config.CROPPED_FOLDER) if f.endswith('.jpg')],
                             key=lambda x: os.path.getmtime(os.path.join(Config.CROPPED_FOLDER, x)),
                             reverse=True)
        self.page = 1
        self.render()

    def change_page(self, d):
        max_p = math.ceil(len(self.images) / self.per_page) or 1
        new_p = self.page + d
        if 1 <= new_p <= max_p:
            self.page = new_p
            self.render()

    def render(self):
        for w in self.content.winfo_children(): w.destroy()
        self.img_refs = []

        start = (self.page - 1) * self.per_page
        batch = self.images[start: start + self.per_page]

        if not batch:
            ttk.Label(self.content, text="No images found.").pack(pady=20)
            return

        cols = 5
        for idx, fname in enumerate(batch):
            path = os.path.join(Config.CROPPED_FOLDER, fname)
            try:
                pil = Image.open(path)
                pil.thumbnail((150, 150))
                tk_img = ImageTk.PhotoImage(pil)
                self.img_refs.append(tk_img)

                cell = ttk.Frame(self.content, relief="solid", borderwidth=1)
                cell.grid(row=idx // cols, column=idx % cols, padx=5, pady=5)

                l = ttk.Label(cell, image=tk_img)
                l.pack()
                l.bind("<Double-1>", lambda e, p=path: os.startfile(p))
                ToolTip(l, text=fname)
            except:
                pass

        total = len(self.images)
        mp = math.ceil(total / self.per_page) or 1
        self.lbl_page.config(text=f"Page {self.page}/{mp} (Total: {total})")

    def get_root_frame(self):
        return self.frame