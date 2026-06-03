# -*- coding: utf-8 -*-
"""
@file: main.py
@description: Entry point for the Radiolarian Image Extraction System.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import sys
from ui_platform import enable_hidpi_awareness
from ui_main import RadiolarianApp


def main():
    # Windows multiprocessing support for frozen/compiled apps
    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.freeze_support()

    # Enable HiDPI awareness BEFORE creating the Tk root so the OS gives us
    # real pixel dimensions instead of bitmap-scaling the UI on 4K monitors.
    enable_hidpi_awareness()

    app = RadiolarianApp()
    app.mainloop()


if __name__ == "__main__":
    main()