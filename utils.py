# -*- coding: utf-8 -*-
"""
@file: utils.py
@description: Helper functions for file system operations and string manipulation.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import os
import re
import shutil


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Sanitizes a filename to ensure it is safe for the file system.
    Removes special characters and truncates if necessary.

    Args:
        filename (str): The original filename.
        max_length (int): Maximum allowed length.

    Returns:
        str: A safe, clean filename.
    """
    base, ext = os.path.splitext(filename)
    # Remove non-alphanumeric chars except dots and dashes
    safe_base = re.sub(r'[^\w\s.-]', '', base).strip()
    # Replace whitespace with underscores
    safe_base = re.sub(r'[\s._-]+', '_', safe_base)

    # Truncate to avoid OS limits
    if len(safe_base.encode('utf-8')) > max_length - len(ext.encode('utf-8')):
        safe_base = safe_base[:max_length - len(ext)]

    return f"{safe_base}{ext}"


def clear_cache_directories(folders: list):
    """
    Deletes content of specified folders and recreates them.

    Args:
        folders (list): List of directory paths to clean.
    """
    for folder in folders:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                os.makedirs(folder)
            except Exception as e:
                print(f"Error cleaning {folder}: {e}")


def create_placeholder_models(model_paths: list):
    """
    Creates empty dummy files for models to allow the GUI to launch
    without the actual heavy model weights present (for demo purposes).
    """
    for path in model_paths:
        if not os.path.exists(path):
            try:
                with open(path, 'w') as f:
                    f.write("Placeholder model file.")
            except Exception:
                pass