# -*- coding: utf-8 -*-
"""
@file: ui_theme.py
@description: Centralised design tokens for the application.

              Single source of truth for colours, typography and spacing.
              The look is intentionally MONOCHROMATIC + ONE accent colour,
              following SCI-paper / lab-software aesthetics (think MATLAB,
              R, Mathematica, Origin).  No gradients, no shadows, no
              rounded corners (clam ttk theme cannot render them).

              Usage:
                  from ui_theme import Colors, Spacing, get_fonts
                  Fonts = get_fonts(ui_family, mono_family)
                  widget.config(bg=Colors.BG_CARD, fg=Colors.TEXT_HEAD,
                                font=Fonts.CARD_TITLE)
@author: Yaxuan Shen
@date: 2025-10-01
"""


# =====================================================================
# Colour palette — monochromatic grayscale + a single deep-blue accent
# =====================================================================
class Colors:
    """Named colours used throughout the UI.  Any new colour MUST be
    added here, NOT hard-coded at the call site, so the palette can be
    audited in one place."""

    # ---- The ONE accent ----
    PRIMARY        = '#1E3A8A'    # deep academic blue
    PRIMARY_HOVER  = '#1E40AF'
    PRIMARY_ACTIVE = '#1E3A8A'

    # ---- Greyscale (text + structure) ----
    TEXT_HEAD      = '#111827'    # near-black for titles
    TEXT_BODY      = '#374151'    # primary running text
    TEXT_MUTED     = '#6B7280'    # secondary / muted captions
    TEXT_DISABLED  = '#9CA3AF'    # disabled state
    TEXT_ON_DARK   = '#FFFFFF'

    # ---- Borders / separators (subtle) ----
    BORDER_LIGHT   = '#E5E7EB'    # default 1px card border
    BORDER_MED     = '#D1D5DB'    # stronger when needed
    BORDER_FOCUS   = PRIMARY      # focused input

    # ---- Backgrounds (off-white scale) ----
    BG_PAGE        = '#F5F5F4'    # outer page (subtle warm grey)
    BG_CARD        = '#FFFFFF'    # white card
    BG_SIDEBAR     = '#F9FAFB'    # near-white sidebar
    BG_STATUSBAR   = '#F3F4F6'    # status bar
    BG_HOVER       = '#F9FAFB'    # subtle hover
    BG_HOVER_DARK  = '#E5E7EB'    # button hover
    BG_LOG         = '#FAFAFA'    # log window (light, not dark)

    # ---- Semantic colours (muted dark variants — no neon) ----
    SUCCESS        = '#166534'    # dark green
    WARNING        = '#92400E'    # dark amber
    DANGER         = '#991B1B'    # dark red (STOP / errors only)
    DANGER_HOVER   = '#7F1D1D'
    INFO           = '#1E40AF'

    # ---- Disabled control bg ----
    DISABLED_BG    = '#F3F4F6'
    DISABLED_FG    = '#9CA3AF'


# =====================================================================
# Spacing — geometric ratios, used to compute paddings/margins
# =====================================================================
class Spacing:
    """Spacing tokens (pixels).  Multiples of 2/4 keep things tidy at
    different DPI levels."""

    # Cards
    CARD_PAD_X    = 16            # left/right padding inside a card
    CARD_PAD_Y    = 14            # top/bottom padding inside a card
    CARD_GAP      = 10            # vertical gap between adjacent cards
    CARD_INNER    = 8             # gap between header and body inside a card

    # Form rows
    FIELD_GAP     = 8             # horizontal gap between label and input
    ROW_GAP       = 8             # vertical gap between form rows

    # Sections (sidebar)
    SECTION_GAP   = 18            # space between sidebar sections
    SECTION_PAD   = 12            # padding inside a section

    # Buttons
    BTN_PAD_X     = 12
    BTN_PAD_Y     = 6

    # Outer
    PAGE_PAD      = 16            # outer padding of the page
    SIDEBAR_W     = 220           # sidebar width
    SIDEBAR_PAD   = 14            # sidebar padding

    # Status bar
    STATUS_PAD_X  = 10
    STATUS_PAD_Y  = 4

    # Log / Gallery toolbars
    TOOLBAR_PAD   = 6


# =====================================================================
# Fonts — must be built after Tk root exists (so font-family lookup
# can pick a real available family).  Call `get_fonts(ui, mono)` once
# in app __init__ and pass the resulting class around.
# =====================================================================
def get_fonts(ui_family: str, mono_family: str):
    """Build a Fonts namespace given the picked UI + mono font families.

    All sizes are in points (Tk treats negative numbers as pixels;
    positive numbers as points, scaled by the DPI setting we already
    applied with `apply_tk_scaling`)."""

    class Fonts:
        # Application chrome
        APP_TITLE      = (ui_family, 14, 'bold')
        APP_SUBTITLE   = (ui_family, 9)

        # Sidebar
        SECTION_LABEL  = (ui_family, 8, 'bold')     # "STATISTICS" etc.
        SIDEBAR_KEY    = (ui_family, 9)
        SIDEBAR_VALUE  = (mono_family, 9, 'bold')

        # Card
        CARD_TITLE     = (ui_family, 11, 'bold')
        STEP_NUMBER    = (mono_family, 12, 'bold')
        BADGE          = (ui_family, 9)

        # Body
        BODY           = (ui_family, 10)
        BODY_BOLD      = (ui_family, 10, 'bold')
        BODY_SMALL     = (ui_family, 9)
        MUTED          = (ui_family, 9)
        MONO_VALUE     = (mono_family, 10)
        MONO_SMALL     = (mono_family, 9)

        # Log / status
        LOG            = (mono_family, 9)
        STATUS         = (ui_family, 9)
        STATUS_VALUE   = (mono_family, 9)

    return Fonts


# =====================================================================
# Status badge symbol+colour pairs (used by StatusBadge widget)
# =====================================================================
STATUS_BADGE = {
    'pending': ('○ Pending', Colors.TEXT_MUTED),
    'running': ('⟳ Running', Colors.PRIMARY),
    'done':    ('✓ Done',    Colors.SUCCESS),
    'failed':  ('✗ Failed',  Colors.DANGER),
}
