# -*- coding: utf-8 -*-
"""
@file: ui_widgets.py
@description: Custom Tkinter widgets used throughout the application.
              All widgets are tested to behave consistently on Windows,
              Ubuntu/Linux (X11 + Wayland-via-XWayland) and macOS.

              Widget catalogue:
                ToolTip          — cross-platform hover tooltip
                ScrollableFrame  — vertically scrollable Frame container
                Card             — white panel with a 1-pixel border
                StepCard         — Card with header (number+title+badge)
                StatusBadge      — 4-state status indicator
                SectionHeader    — small uppercase label for sidebar sections
                VSeparator       — 1px vertical separator for status bar
@author: Yaxuan Shen
@date: 2025-10-01
"""

import sys
import tkinter as tk
from tkinter import ttk


class ToolTip(object):
    """
    Creates a pop-up tooltip for a given widget.

    Cross-platform notes:
      - Uses `wm_overrideredirect(True)` so no window decoration.
      - Adds `-topmost` so the tooltip doesn't get covered by the main
        window on GNOME / KDE / i3.
      - Adds the `-type tooltip` X11 hint where supported so window
        managers treat it as a proper tooltip (no focus stealing, no
        taskbar entry on KDE).
    """

    def __init__(self, widget, text='widget info'):
        self.waittime = 500  # milliseconds
        self.wraplength = 180  # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        # Hide tooltip when the widget itself is destroyed (otherwise the
        # toplevel becomes an orphan with no parent on Linux).
        self.widget.bind("<Destroy>", lambda e: self.hidetip())
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            try:
                self.widget.after_cancel(id)
            except tk.TclError:
                pass

    def showtip(self, event=None):
        # Use the widget's root-coordinates + its rendered size to pin the
        # tooltip just below it.  `self.widget.bbox("insert")` only makes
        # sense for Text widgets — on ttk.Label it returns (0,0,0,0) and
        # the tooltip would end up in the top-left of the screen.
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        except tk.TclError:
            # Widget already destroyed (race with rapid mouse movement)
            return
        # Avoid creating a second toplevel if one is already up
        if self.tw is not None:
            return
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        # Stay above the main window — without this, GNOME and some other
        # Linux WMs will lower the tooltip behind the application window.
        try:
            self.tw.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        # X11 hint: tell the WM this is a tooltip so it doesn't appear in
        # the taskbar (KDE) and doesn't steal focus.  Ignored on Windows.
        if sys.platform.startswith('linux'):
            try:
                self.tw.wm_attributes("-type", "tooltip")
            except tk.TclError:
                pass
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("TkDefaultFont", 9, "normal"),
                         wraplength=self.wraplength)
        label.pack(ipadx=2, ipady=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            try:
                tw.destroy()
            except Exception:
                pass


class ScrollableFrame(ttk.Frame):
    """
    A Frame that includes a vertical scrollbar.

    Cross-platform behaviour:
      - Mouse-wheel scrolling works on all three event models
        (Windows / macOS use `<MouseWheel>`, Linux X11 uses
        `<Button-4>` / `<Button-5>`).
      - The inner frame tracks canvas width so labelframes / widgets
        packed inside actually fill the available horizontal space.
      - Wheel binding is scoped to hover (Enter/Leave on canvas)
        instead of `bind_all`, so multiple scrollable frames in the
        same app don't fight over the wheel event.
    """

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical",
                                       command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Keep the window-id so we can resize it when the canvas resizes
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.scrollable_frame.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Bind on Enter so wheel only affects the canvas the cursor is over.
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, _event):
        """Update scroll region when inner frame size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Resize the embedded frame to match canvas width so content
        fills horizontally instead of hugging the left edge."""
        self.canvas.itemconfig(self._window_id, width=event.width)

    def _bind_mousewheel(self, _event):
        # Windows / macOS
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Linux X11
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        # Only scroll if the canvas actually has overflow — otherwise wheel
        # events would still be consumed and could feel "stuck".
        try:
            first, last = self.canvas.yview()
        except tk.TclError:
            return
        if first <= 0.0 and last >= 1.0:
            return

        # Direction across the 3 event styles
        if getattr(event, 'num', 0) == 4:        # Linux scroll up
            delta = -1
        elif getattr(event, 'num', 0) == 5:      # Linux scroll down
            delta = 1
        else:
            # Windows: event.delta is multiple of 120
            # macOS:   event.delta is small (1-3), still works with sign
            if event.delta == 0:
                return
            delta = -1 if event.delta > 0 else 1

        self.canvas.yview_scroll(delta, "units")

# =====================================================================
# Visual primitives for the academic / scientific look
# =====================================================================
from ui_theme import Colors, Spacing, STATUS_BADGE


class Card(tk.Frame):
    """A white card with a 1-pixel light-grey border.

    Implementation note: ttk.Frame in the clam theme cannot reliably
    render a thin border with a chosen colour (the bordercolor option
    is partially ignored).  We therefore build the border the
    bullet-proof way — a tk.Frame whose background IS the border colour,
    with a 1-pixel inner padding revealing the body Frame underneath.

    Use the `.body` attribute to add child widgets:
        card = Card(parent)
        card.pack(fill='x', padx=10, pady=8)
        ttk.Label(card.body, text="hello").pack()
    """

    def __init__(self, master, **kwargs):
        super().__init__(master,
                         bg=Colors.BORDER_LIGHT,   # 'border' = outer bg
                         highlightthickness=0,
                         bd=0,
                         **kwargs)
        self.body = tk.Frame(self, bg=Colors.BG_CARD,
                             highlightthickness=0, bd=0)
        # 1-pixel padding reveals the outer Frame's background = a
        # crisp 1px border on all four sides.
        self.body.pack(fill='both', expand=True, padx=1, pady=1)


class StatusBadge(tk.Label):
    """A small text+symbol label that has 4 visual states:
        pending  → grey ○
        running  → blue ⟳   (static — no animation per design spec)
        done     → green ✓
        failed   → red ✗

    Use `set_state('running')` to transition.  Width is fixed so
    the surrounding header doesn't shift when the text changes."""

    def __init__(self, master, state='pending', bg=None):
        if bg is None:
            bg = Colors.BG_CARD
        symbol, color = STATUS_BADGE[state]
        super().__init__(master, text=symbol, fg=color, bg=bg,
                         font=('TkDefaultFont', 9),
                         anchor='e', width=12,
                         padx=0, pady=0)
        self._state = state
        self._bg = bg

    def set_state(self, state):
        """Transition to one of: 'pending' / 'running' / 'done' / 'failed'."""
        if state not in STATUS_BADGE:
            return
        symbol, color = STATUS_BADGE[state]
        self.config(text=symbol, fg=color)
        self._state = state

    def get_state(self):
        return self._state


class StepCard(Card):
    """A Card with a step-numbered header.

    Layout:
        ┌──────────────────────────────────────┐
        │  01  Title text             ✓ Done   │  ← header
        │  ──────────────────────────────────  │  ← separator
        │                                       │
        │  <user-supplied content via .body>    │
        │                                       │
        └──────────────────────────────────────┘

    Use the `.body` attribute (inherited from Card) for content.
    Use `set_status('running'|'done'|'failed'|'pending')` to update
    the badge from outside (e.g. when a task starts/ends)."""

    def __init__(self, master, step_num: str, title: str,
                 fonts=None, **kwargs):
        super().__init__(master, **kwargs)
        # `fonts` is the Fonts namespace from ui_theme.get_fonts() —
        # callers pass it in so we don't recompute font lookup per card.
        if fonts is None:
            class _Fallback:
                STEP_NUMBER = ('TkFixedFont', 12, 'bold')
                CARD_TITLE  = ('TkDefaultFont', 11, 'bold')
                BADGE       = ('TkDefaultFont', 9)
            fonts = _Fallback

        # Layout structure: header / separator / content stacked
        # vertically inside the Card body.  We rebind `self.body` to
        # the content area at the end so external callers add widgets
        # there (not into the header).
        outer = self.body
        outer.config(bg=Colors.BG_CARD)

        # Header
        header = tk.Frame(outer, bg=Colors.BG_CARD)
        header.pack(fill='x', padx=Spacing.CARD_PAD_X,
                    pady=(Spacing.CARD_PAD_Y, Spacing.CARD_INNER))

        # Step number (e.g. "01")
        tk.Label(header, text=step_num,
                 bg=Colors.BG_CARD, fg=Colors.PRIMARY,
                 font=fonts.STEP_NUMBER, padx=0, pady=0).pack(side='left')

        # Title
        tk.Label(header, text=title,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_HEAD,
                 font=fonts.CARD_TITLE,
                 padx=0, pady=0).pack(side='left', padx=(10, 0))

        # Status badge (right-aligned)
        self.badge = StatusBadge(header, state='pending')
        self.badge.config(font=fonts.BADGE)
        self.badge.pack(side='right')

        # Thin separator line
        sep = tk.Frame(outer, height=1, bg=Colors.BORDER_LIGHT)
        sep.pack(fill='x', padx=Spacing.CARD_PAD_X)

        # Content area — THIS replaces `.body` for external use
        content = tk.Frame(outer, bg=Colors.BG_CARD)
        content.pack(fill='x', padx=Spacing.CARD_PAD_X,
                     pady=(Spacing.CARD_INNER, Spacing.CARD_PAD_Y))

        # Re-bind .body so caller adds widgets into content, not header
        self.body = content

    def set_status(self, state):
        """Update the header status badge.  Convenience proxy."""
        self.badge.set_state(state)


class SectionHeader(tk.Label):
    """Small uppercase label used as a sidebar section divider.

        STATISTICS
        ──────────
    """

    def __init__(self, master, text, fonts=None, bg=None):
        if bg is None:
            bg = Colors.BG_SIDEBAR
        font = fonts.SECTION_LABEL if fonts else ('TkDefaultFont', 8, 'bold')
        super().__init__(master, text=text.upper(),
                         bg=bg, fg=Colors.TEXT_MUTED,
                         font=font, anchor='w', padx=0, pady=0)


class VSeparator(tk.Frame):
    """1-pixel vertical separator for use inside horizontal toolbars /
    status bars.  Height matches its container automatically."""

    def __init__(self, master, height=18, color=None):
        if color is None:
            color = Colors.BORDER_MED
        super().__init__(master, width=1, height=height,
                         bg=color, highlightthickness=0, bd=0)


class HSeparator(tk.Frame):
    """1-pixel horizontal separator for use between sidebar sections."""

    def __init__(self, master, color=None):
        if color is None:
            color = Colors.BORDER_LIGHT
        super().__init__(master, height=1, bg=color,
                         highlightthickness=0, bd=0)
