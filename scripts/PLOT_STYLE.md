# Publication plot style

`plot_style.py` is the shared source for Matplotlib typography, physical dimensions,
strokes, bottom-legend placement and fixed-canvas export. Apply the style before
creating artists:

```python
from scripts.plot_style import apply_style, figure_size, export_figure
apply_style()
fig, ax = plt.subplots(figsize=figure_size("curve"))
# Plot data as usual.
export_figure(fig, "output/plots/example.pdf")
```

Use `width="half"`, `width="third"`, or a fraction in `(0, 1]` for an intentionally
narrow placement. YAML accepts `defaults.width_fraction`; `grid` still controls
panel topology. Figure dimensions are configured before drawing, not scaled during
export. No producer uses tight export bounding boxes. For constrained-layout
figures with a shared legend, `bottom_legend` reserves space inside the canvas.

## Constants and units

- Text width: **12.2 cm**, 4.8031496063 inches. The given **347.12354 TeX pt** is
  approximately equivalent; PDF uses 72-point inches, so the measured full-width
  MediaBox is **345.82677165 PDF pt**, not 347.12354 PDF pt.
- Text: **7.5 TeX pt × 72 / 72.27 = 7.4719800747 Matplotlib pt**, including titles,
  labels, ticks, legend titles, legends, figure labels and annotations.
- CMU Serif; Computer Modern math (`cm`), including scientific notation. PDF/PS
  TrueType embedding and SVG live text are centralized.
- Strokes in Matplotlib points: axes and major/minor ticks **1**, grid **0.5**,
  ordinary curves **1.5**, dashed task boundaries **0.7**.
- Named roles: raw curve **0.5**, dense raw curve **0.2**, zoom connectors **1**,
  interpolation paths **1.5**, uncertainty-band edges **0**. YAML numeric
  `linewidth` alternatives were removed; distinct raw traces use `stroke_role`.
- Named width/height ratios: curve 2; zoom 1.25; cropped curve 1.4; aggregate 1.65;
  comparison 2.5; individual 14/9; training paths 3; combined 4.8/2.42;
  heatmap 1; heatmap row .75; heatmap with curve 5/7; benchmark 200/120.
  Grid ratios account for rows/columns; YAML curve grids reserve at least 2 inches
  per row for per-panel labels and legends. Ratios include margins and legends.

