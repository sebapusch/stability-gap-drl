"""Publication contract: physical PDF canvas, uniform text, and import modes."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import plot_style as style


class PlotStyleTests(unittest.TestCase):
    def test_export_width_ignores_external_tight_setting(self):
        for width in (1, .5):
            style.apply_style()
            fig, ax = plt.subplots(figsize=style.figure_size(width=width))
            ax.plot([0, 1], [0, 1])
            ax.set_xlabel(r"Timesteps ($\times 10^5$)")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "figure.pdf"
                with matplotlib.rc_context({"savefig.bbox": "tight"}):
                    style.export_figure(fig, path)
                box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", path.read_bytes())
                self.assertIsNotNone(box)
                self.assertAlmostEqual(float(box[3]) - float(box[1]), 12.2 / 2.54 * 72 * width, places=6)
            plt.close(fig)

    def test_text_and_stroke_contract(self):
        style.apply_style()
        fig, ax = plt.subplots()
        ax.set(title="Title", xlabel="X", ylabel="Y")
        ax.plot([0, 1], [0, 1], label="Curve")
        legend = ax.legend(title="Legend")
        annotation = ax.annotate("Annotation", (.5, .5))
        fig.canvas.draw()
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label, annotation,
                 legend.get_title(), *legend.get_texts(), *ax.get_xticklabels(), *ax.get_yticklabels()]
        for text in texts:
            self.assertAlmostEqual(text.get_fontsize(), 7.5 * 72 / 72.27)
        self.assertEqual(ax.lines[0].get_linewidth(), style.LINE_WIDTH)
        self.assertEqual(ax.spines['left'].get_linewidth(), style.AXIS_WIDTH)
        self.assertEqual(style.line_width({'stroke_role': 'raw_curve'}, {}, {}), .5)
        plt.close(fig)

    def test_notebook_bootstrap_from_both_working_directories(self):
        for path in (ROOT / 'scripts').glob('*.ipynb'):
            cells = [c for c in json.loads(path.read_text())['cells'] if c['cell_type'] == 'code']
            for cell in cells:
                compile(''.join(cell['source']), str(path), 'exec')
            bootstrap = ''.join(cells[0]['source']).split('apply_style()')[0] + 'apply_style()'
            for cwd in (ROOT, ROOT / 'scripts'):
                # Emulate a kernel holding old values, then rerun the setup cell.
                regression = bootstrap + "\n" + "\n".join([
                    "expected_width = plot_style.LINE_WIDTH",
                    "expected_boundary = plot_style.STROKES['task_boundary']",
                    "plot_style.LINE_WIDTH = 99",
                    "plot_style.STROKES['task_boundary'] = 99",
                    "plot_style.apply_style()",
                    bootstrap,
                    "assert plot_style.LINE_WIDTH == expected_width",
                    "assert STROKES['task_boundary'] == expected_boundary",
                    "import matplotlib.pyplot as plt",
                    "fig, ax = plt.subplots()",
                    "line, = ax.plot([0, 1], [0, 1])",
                    "assert line.get_linewidth() == expected_width",
                    "plt.close(fig)",
                ])
                subprocess.run([sys.executable, '-c', regression], cwd=cwd, check=True, capture_output=True)

    def test_all_yaml_roles_are_valid(self):
        from scripts.plot_iqm import load_config
        for path in (ROOT / 'scripts/plots').rglob('*.yaml'):
            cfg = load_config(path)
            for panel in cfg['plots']:
                for line in panel['lines']:
                    style.line_width(line, panel, cfg.get('defaults', {}))


if __name__ == '__main__':
    unittest.main()
