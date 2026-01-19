"""Dark theme color constants for the Data Room Graph application."""


class Colors:
    """Color palette for the dark theme with purple accents."""

    # Base Colors (Dark Theme)
    BG_PRIMARY = "#1a1a2e"      # Main window background
    BG_SECONDARY = "#16213e"    # Sidebar, panels
    BG_TERTIARY = "#1f1f3d"     # Cards, elevated surfaces
    BG_INPUT = "#252547"        # Input fields, text areas
    BORDER = "#3d3d6b"          # Borders, dividers
    BORDER_LIGHT = "#4a4a7d"    # Hover borders

    # Text Colors
    TEXT_PRIMARY = "#e8e8f0"    # Primary text
    TEXT_SECONDARY = "#a0a0b8"  # Secondary/muted text
    TEXT_DISABLED = "#6b6b85"   # Disabled text

    # Accent Colors (Purple)
    ACCENT_PRIMARY = "#8b5cf6"  # Primary buttons, links, selections
    ACCENT_HOVER = "#a78bfa"    # Hover states
    ACCENT_ACTIVE = "#7c3aed"   # Active/pressed states
    ACCENT_MUTED = "#4c1d95"    # Subtle accent backgrounds

    # Status Colors
    SUCCESS = "#22c55e"         # Success states, connected
    WARNING = "#f59e0b"         # Warning states, in progress
    ERROR = "#ef4444"           # Error states, disconnected
    INFO = "#3b82f6"            # Info states


class GraphColors:
    """Node colors optimized for dark background."""

    DATA_ROOM = "#4a5568"
    FOLDER = "#5a6678"
    DOCUMENT = "#60a5fa"       # Brighter blue for dark bg
    CHUNK = "#34d399"          # Brighter green for dark bg
    FUND = "#fbbf24"           # Brighter amber for dark bg
    MANAGER = "#a78bfa"        # Purple to match accent
    PERSON = "#f87171"         # Brighter red for dark bg
    VEHICLE = "#fb923c"        # Brighter orange for dark bg
    SERVICE_PROVIDER = "#2dd4bf"  # Brighter teal for dark bg
    INVESTOR = "#38bdf8"       # Brighter sky blue for dark bg
    ENTITY = "#94a3b8"         # Slate gray


# CSS-ready color strings for inline styles
STYLE_COLORS = {
    # Backgrounds
    "bg_primary": Colors.BG_PRIMARY,
    "bg_secondary": Colors.BG_SECONDARY,
    "bg_tertiary": Colors.BG_TERTIARY,
    "bg_input": Colors.BG_INPUT,

    # Borders
    "border": Colors.BORDER,
    "border_light": Colors.BORDER_LIGHT,

    # Text
    "text_primary": Colors.TEXT_PRIMARY,
    "text_secondary": Colors.TEXT_SECONDARY,
    "text_disabled": Colors.TEXT_DISABLED,

    # Accents
    "accent": Colors.ACCENT_PRIMARY,
    "accent_hover": Colors.ACCENT_HOVER,
    "accent_active": Colors.ACCENT_ACTIVE,
    "accent_muted": Colors.ACCENT_MUTED,

    # Status
    "success": Colors.SUCCESS,
    "warning": Colors.WARNING,
    "error": Colors.ERROR,
    "info": Colors.INFO,
}
