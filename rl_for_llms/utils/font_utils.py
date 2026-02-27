from matplotlib import font_manager

from rl_for_llms.utils.path_utils import get_thesis_fonts_dir


def get_thesis_font_families() -> dict[str, list[str]]:
    """Return the default thesis font families for serif, sans-serif, and monospace."""
    return {
        "serif": ["Merriweather", "DejaVu Serif", "Times New Roman"],
        "sans-serif": ["Public Sans", "DejaVu Sans", "Helvetica"],
        "monospace": ["Inconsolata", "DejaVu Sans Mono"],
    }


def load_thesis_fonts() -> None:
    """Load custom thesis TTF fonts into matplotlib's font manager."""
    fonts_dir = get_thesis_fonts_dir()
    if fonts_dir.exists():
        for font_file in fonts_dir.glob("*.ttf"):
            font_manager.fontManager.addfont(str(font_file))
