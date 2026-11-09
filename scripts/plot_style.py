"""Publication styling in physical units; configure before creating artists.

Full-width figures occupy 12.2 cm without LaTeX scaling. ``aspect`` is width /
height; grid dimensions describe panel topology, never extra physical width.
Stroke widths are Matplotlib points (as in the original IQM notebook).
"""
from pathlib import Path

import matplotlib as mpl

TEXT_WIDTH_CM = 12.2
TEXT_WIDTH_IN = TEXT_WIDTH_CM / 2.54
TEX_PT_TO_MPL_PT = 72 / 72.27
FONT_SIZE_TEX_PT = 7.5
FONT_SIZE = FONT_SIZE_TEX_PT * TEX_PT_TO_MPL_PT
FONT_FAMILY = "CMU Serif"
AXIS_WIDTH = 0.8
GRID_WIDTH = 0.5
LINE_WIDTH = 1.3
TASK_BOUNDARY_WIDTH = 0.5
TICK_WIDTH = AXIS_WIDTH
TASK_LABEL_OFFSET_PT = 18
TASK_XLABEL_PAD_PT = 32
STROKES = {
    "curve": LINE_WIDTH,
    "raw_curve": 0.5,  # subordinate unsmoothed trace beneath a smoothed curve
    "dense_raw_curve": 0.2,
    "task_boundary": TASK_BOUNDARY_WIDTH,
    "zoom_connector": 1.0,
    "interpolation_path": LINE_WIDTH,
    "band_edge": 0.0,
    "grid": GRID_WIDTH,
}
# Ratios include room for labels and, where used, a shared bottom legend.
ASPECTS = {
    "curve": 2.0,
    "zoom": 1.25,
    "cropped_curve": 1.4,
    "aggregate": 1.65,
    "comparison": 2.5,
    "individual": 14 / 9,
    "training_paths": 3.0,
    "combined": 4.8 / 2.42,
    "heatmap": 1.0,
    "heatmap_row": 0.75,
    "heatmap_curve": 5 / 7,
    "benchmark": 200 / 120,
}
# Minimum row height for YAML curves with per-panel axis labels and legends.
CURVE_GRID_ROW_HEIGHT_IN = 2.0
WIDTH_FRACTIONS = {"full": 1.0, "half": 0.5, "third": 1 / 3}
# Change this to "below" to put the comparison legend on its own row.
COMPARISON_LEGEND_PLACEMENT = "beside_xlabel"


def figure_size(aspect="curve", *, width="full", rows=1, cols=1):
    """Return inches at placement width; width may be a name or fraction."""
    fraction = WIDTH_FRACTIONS[width] if isinstance(width, str) else width
    ratio = ASPECTS[aspect] if isinstance(aspect, str) else aspect
    if not (0 < fraction <= 1 and ratio > 0 and rows > 0 and cols > 0):
        raise ValueError("Width must be in (0, 1]; aspect and grid dimensions positive.")
    inches = TEXT_WIDTH_IN * fraction
    height = inches / ratio * rows / cols
    if aspect in ("curve", "cropped_curve") and cols > 1:
        height = max(height, rows * CURVE_GRID_ROW_HEIGHT_IN)
    return inches, height


def apply_style():
    """Apply all recurring typography and strokes without changing colors."""
    settings = {
        "font.family": FONT_FAMILY, "mathtext.fontset": "cm",
        "text.usetex": False, "axes.formatter.use_mathtext": True,
        "axes.linewidth": AXIS_WIDTH, "grid.linewidth": GRID_WIDTH,
        "lines.linewidth": LINE_WIDTH, "patch.linewidth": AXIS_WIDTH,
        "lines.markeredgewidth": TICK_WIDTH,
        "xtick.major.width": TICK_WIDTH, "ytick.major.width": TICK_WIDTH,
        "xtick.minor.width": TICK_WIDTH, "ytick.minor.width": TICK_WIDTH,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "savefig.bbox": None, "savefig.pad_inches": 0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    for key in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
                "ytick.labelsize", "legend.fontsize", "legend.title_fontsize",
                "figure.titlesize", "figure.labelsize"):
        settings[key] = FONT_SIZE
    mpl.rcParams.update(settings)


def bottom_legend(fig, handles, labels, *, ncol=3, placement="below"):
    """Put a legend below the axes, optionally beside the figure x-axis label."""
    if placement not in ("below", "beside_xlabel"):
        raise ValueError("placement must be 'below' or 'beside_xlabel'")
    if placement == "beside_xlabel":
        xlabel = fig._supxlabel
        if xlabel is None:
            raise ValueError("beside_xlabel requires fig.supxlabel(...) first")
        legend = fig.legend(handles, labels, frameon=False, loc="center right",
                            ncol=ncol, bbox_to_anchor=(0.98, 0))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        canvas_width = fig.get_figwidth() * fig.dpi
        canvas_height = fig.get_figheight() * fig.dpi
        legend_box = legend.get_window_extent(renderer)
        label_box = xlabel.get_window_extent(renderer)
        gap = 8 * fig.dpi / 72  # points; independent of export DPI
        left = 0.98 - legend_box.width / canvas_width
        label_right = left - gap / canvas_width
        if label_right * canvas_width < label_box.width + gap:
            raise ValueError("X label and legend do not fit beside each other at this width")
        strip = (max(legend_box.height, label_box.height) + 12 * fig.dpi / 72) / canvas_height
        center_y = strip / 2
        fig.get_layout_engine().set(rect=(0, strip, 1, 1 - strip))
        legend.set_bbox_to_anchor((0.98, center_y), transform=fig.transFigure)
        fig.supxlabel(xlabel.get_text(), x=label_right, y=center_y,
                     ha="right", va="center")
        return legend
    legend_height = 0.25 / fig.get_figheight()
    strip = legend_height * (2 if fig._supxlabel is not None else 1)
    fig.get_layout_engine().set(rect=(0, strip, 1, 1 - strip))
    if fig._supxlabel is not None:
        fig.supxlabel(fig._supxlabel.get_text(), y=legend_height + 0.01)
    return fig.legend(handles, labels, frameon=False, loc="lower center",
                      ncol=ncol, bbox_to_anchor=(0.5, 0))


def export_figure(fig, path, **kwargs):
    """Save the configured canvas exactly, never crop or rescale its artists."""
    if kwargs.pop("bbox_inches", None) is not None:
        raise ValueError("Tight cropping changes placement width; fix the layout instead.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(path, bbox_inches=None, **kwargs)
    return path


def line_width(line, panel, defaults):
    """Resolve a named YAML visual role, preserving the old scope precedence."""
    for config in (line, panel, defaults):
        if "stroke_role" in config:
            return STROKES[config["stroke_role"]]
        if "linewidth" in config:
            raise ValueError("Replace linewidth with a named stroke_role from plot_style.STROKES")
    return LINE_WIDTH

# SVG benchmark artwork uses a 200-unit design coordinate system. These roles
# are illustration details, not data curves or axes; retain their hierarchy.
SVG_DESIGN_WIDTH = 200
SVG_UNITS_PER_PT = SVG_DESIGN_WIDTH / (TEXT_WIDTH_IN * 72)
SVG_FONT_SIZE = FONT_SIZE * SVG_UNITS_PER_PT
SCHEMATIC_STROKES = {
    "detail": 0.35, "box": 0.3, "rail": 0.45, "pendulum": 0.85,
    "branch": 0.4, "task_arrow": 0.55, "task_box": 0.28,
}
