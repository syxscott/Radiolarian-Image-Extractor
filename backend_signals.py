# -*- coding: utf-8 -*-
"""
@file: backend_signals.py
@description: Defines the communication protocol between background worker threads
              and the main GUI thread using thread-safe queues.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import queue


class WorkerSignals:
    """
    Proxy class to handle logging and progress updates from background threads.
    Decouples the logic layer from the UI layer.
    """

    def __init__(self, log_queue: queue.Queue, progress_queue: queue.Queue):
        self.log_queue = log_queue
        self.progress_queue = progress_queue

    def log(self, message: str, level: str = "info"):
        """
        Sends a log message to the GUI.

        Args:
            message (str): The text to display.
            level (str): 'info', 'warn', 'error', or 'success'.
        """
        self.log_queue.put({"text": message, "level": level})
        # Also print to console for debugging
        print(f"[{level.upper()}] {message}")

    def progress(self, task_name: str, value: int):
        """
        Updates the progress bar for a specific task.

        Args:
            task_name (str): 'conversion', 'detection', or 'classification'.
            value (int): Progress percentage (0-100).
        """
        self.progress_queue.put({"task": task_name, "value": value})