# -*- coding: utf-8 -*-
"""
@file: ui_platform.py
@description: Cross-platform UI helpers — font picking, HiDPI awareness,
              emoji-safe button labels, etc.  Centralises platform-specific
              quirks so the rest of the UI code stays clean.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import sys
import tkinter as tk
import tkinter.font as tkfont


# ---------------------------------------------------------
# HiDPI awareness
# ---------------------------------------------------------
def enable_hidpi_awareness():
    """Tell Windows we are DPI-aware so the UI isn't bitmap-scaled
    (blurry/tiny) on 4K monitors.  No-op on Linux / macOS where the
    behaviour is correct out of the box.

    Must be called BEFORE creating the Tk root window.
    """
    if not sys.platform.startswith('win'):
        return
    try:
        # Windows 8.1+ : per-monitor DPI awareness
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            # Windows 7 fallback
            from ctypes import windll
            windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def apply_tk_scaling(root):
    """Scale Tkinter to match the system DPI.  Called after the Tk root
    is created.  On a 96-DPI monitor `scaling` is 1.0; on a 192-DPI
    (4K-equivalent) monitor it is 2.0."""
    try:
        # Tk's default scaling assumes 72 DPI; multiply by the real
        # screen DPI to get widget sizes that match the OS.
        dpi = root.winfo_fpixels('1i')   # pixels per inch
        scale = dpi / 72.0
        # Clamp to sensible range
        scale = max(1.0, min(scale, 3.0))
        root.tk.call('tk', 'scaling', scale)
    except Exception:
        pass


# ---------------------------------------------------------
# Cross-platform fonts
# ---------------------------------------------------------
# Resolved on first call and cached so we don't pay the font-enumeration
# cost on every widget construction.
_CACHED_FONTS = {}


def _pick_font(candidates, fallback):
    """Return the first candidate font that exists on this system."""
    if not _CACHED_FONTS.get('_init'):
        try:
            _CACHED_FONTS['_families'] = set(tkfont.families())
        except Exception:
            _CACHED_FONTS['_families'] = set()
        _CACHED_FONTS['_init'] = True
    families = _CACHED_FONTS['_families']
    for c in candidates:
        if c in families:
            return c
    return fallback


def get_ui_font():
    """Default proportional UI font (titles, labels, buttons)."""
    if 'ui' not in _CACHED_FONTS:
        _CACHED_FONTS['ui'] = _pick_font(
            ['Segoe UI', 'Ubuntu', 'Cantarell', 'DejaVu Sans',
             'Liberation Sans', 'Helvetica'],
            'TkDefaultFont',
        )
    return _CACHED_FONTS['ui']


def get_mono_font():
    """Default monospace font (log window, code)."""
    if 'mono' not in _CACHED_FONTS:
        _CACHED_FONTS['mono'] = _pick_font(
            ['Consolas', 'Cascadia Code', 'DejaVu Sans Mono',
             'Ubuntu Mono', 'Liberation Mono', 'Menlo', 'Courier New'],
            'TkFixedFont',
        )
    return _CACHED_FONTS['mono']


# ---------------------------------------------------------
# Emoji-safe symbols for buttons.
# Used where a single emoji is the entire label — without a color-emoji
# font (minimal Linux installs), the button would otherwise appear as
# a blank square and leave the user with no clue what it does.
# Log messages keep their decorative emoji because surrounding text
# still makes them readable even when a glyph is missing.
# ---------------------------------------------------------
def can_render_color_emoji():
    """Best-effort check for color-emoji support.  Returns True on
    Windows 10+ and macOS, False on most Linux installs unless the user
    has installed `fonts-noto-color-emoji` or similar.

    Heuristic — we look for a known emoji font family.
    """
    if sys.platform.startswith('win') or sys.platform == 'darwin':
        return True
    if not _CACHED_FONTS.get('_init'):
        try:
            _CACHED_FONTS['_families'] = set(tkfont.families())
        except Exception:
            _CACHED_FONTS['_families'] = set()
        _CACHED_FONTS['_init'] = True
    families = _CACHED_FONTS['_families']
    return any(name in families for name in (
        'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji',
        'Twitter Color Emoji', 'EmojiOne Color',
    ))


def emoji_or_text(emoji_char: str, text_fallback: str) -> str:
    """Return the emoji on platforms that can render it, otherwise a
    plain-text fallback.  Use this for button labels where a blank
    square would leave the user with no clue what the button does.

    Example:
        text=emoji_or_text("🔄", "Reload")
    """
    if can_render_color_emoji():
        return emoji_char
    return text_fallback
