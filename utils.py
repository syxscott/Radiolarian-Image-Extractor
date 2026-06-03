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
import subprocess
import sys
import time


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Sanitizes a filename to ensure it is safe for the file system.
    Removes special characters and truncates if necessary.

    If, after stripping invalid characters, the base name would be empty
    (e.g. user imported a file literally named "!!!.pdf"), a unique
    fallback name is generated so the file is never imported as a
    hidden file like ".pdf".

    Handles pathological inputs:
      - "..pdf"      -> base='', ext='.pdf' (preserves intended extension)
      - "..."        -> generates 'unnamed_<ts>'
      - "..foo.pdf"  -> base='foo', ext='.pdf'

    Args:
        filename (str): The original filename.
        max_length (int): Maximum allowed length.

    Returns:
        str: A safe, clean filename.
    """
    # ----- Edge-case: strip leading dots so splitext doesn't mis-parse -----
    # `os.path.splitext('..pdf')` returns ('..pdf', '') — losing the
    # intended extension. We pre-strip leading dots, then re-split.
    working = filename.lstrip('.')
    if not working:
        # Input was nothing but dots
        return f"unnamed_{int(time.time() * 1000)}"

    base, ext = os.path.splitext(working)
    # If after stripping dots there is still no extension AND the remaining
    # text looks like a typical extension (short, alphanumeric), the user
    # almost certainly meant it as the extension.  Examples:
    #   "..pdf" -> working="pdf" -> ext=".pdf", base=""
    #   "..jpg" -> working="jpg" -> ext=".jpg", base=""
    if not ext and base and base.isalnum() and 1 <= len(base) <= 5:
        ext = '.' + base
        base = ''

    # Remove non-alphanumeric chars except dots and dashes
    safe_base = re.sub(r'[^\w\s.-]', '', base).strip()
    # Replace whitespace / dots / underscores / dashes runs with single _
    safe_base = re.sub(r'[\s._-]+', '_', safe_base)
    # Strip leading/trailing underscores left over from collapsed dots
    safe_base = safe_base.strip('_')

    # Fallback: if everything got stripped (e.g. "!!!.pdf"),
    # generate a unique name so we never produce ".pdf" or "_pdf".
    if not safe_base:
        safe_base = f"unnamed_{int(time.time() * 1000)}"

    # Truncate to avoid OS limits (use byte-based truncation for non-ASCII chars)
    ext_bytes = ext.encode('utf-8')
    max_base_bytes = max(1, max_length - len(ext_bytes))
    base_bytes = safe_base.encode('utf-8')
    if len(base_bytes) > max_base_bytes:
        base_bytes = base_bytes[:max_base_bytes]
        safe_base = base_bytes.decode('utf-8', errors='ignore')
        if not safe_base:  # paranoid: all multi-byte truncation
            safe_base = f"unnamed_{int(time.time() * 1000)}"

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
            except Exception as e:
                print(f"Error removing {folder}: {e}")
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            print(f"Error creating {folder}: {e}")


def open_path(path):
    """
    Opens a file or folder with the system's default application (cross-platform).

    Uses subprocess (not os.system) to avoid shell-injection on paths
    that contain double-quotes or other shell metacharacters.

    Failures (path missing, permission denied, no default handler, etc.)
    are logged to stderr instead of raising, so callers in the UI event
    loop don't get blown up by a bad click.
    """
    try:
        if sys.platform.startswith('win'):
            # os.startfile is the Windows-native call; it does not go through
            # a shell so it's safe even for paths with weird characters.
            os.startfile(path)
        elif sys.platform.startswith('darwin'):
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    except (OSError, FileNotFoundError) as e:
        # e.g. path doesn't exist, NUL char in path, no associated handler.
        print(f"[open_path] Cannot open {path!r}: {type(e).__name__}: {e}",
              file=sys.stderr)