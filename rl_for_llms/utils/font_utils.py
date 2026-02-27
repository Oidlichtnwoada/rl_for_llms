from matplotlib import font_manager

from rl_for_llms.utils.path_utils import get_thesis_fonts_dir


def get_chart_font_families() -> dict[str, list[str]]:
    """Return Computer Modern font families for chart rendering."""
    return {
        "serif": ["cmr10"],
        "sans-serif": ["cmss10"],
        "monospace": ["cmtt10"],
    }


def load_thesis_fonts() -> None:
    """Load custom thesis TTF fonts into matplotlib's font manager."""
    fonts_dir = get_thesis_fonts_dir()
    if fonts_dir.exists():
        for font_file in fonts_dir.glob("*.ttf"):
            font_manager.fontManager.addfont(str(font_file))
