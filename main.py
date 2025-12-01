# -*- coding: utf-8 -*-
"""
@file: main.py
@description: Entry point for the Radiolarian Image Extraction System.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import sys
from ui_main import RadiolarianApp

def main():
    # Windows-specific fix for multiprocessing support in frozen/compiled apps
    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.freeze_support()

    app = RadiolarianApp()
    app.mainloop()

if __name__ == "__main__":
    main()