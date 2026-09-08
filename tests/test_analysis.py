from pathlib import Path
import tempfile
import unittest

import pandas as pd

from analysis.generate_appendix_figures import build_analysis


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "processed" / "Table_A0_cleaned_raw_results.csv"


class AnalysisTests(unittest.TestCase):
    def test_processed_results_rebuild_all_appendix_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = build_analysis(
                RESULTS,
                Path(directory),
                generate_figures=False,
            )

            self.assertEqual(summary["records"], 532)
            self.assertEqual(summary["baselines"], 8)
            self.assertEqual(summary["matrix_rows"], 60)
            self.assertEqual(summary["best_worst_rows"], 60)
            table_dir = Path(directory, "tables")
            rebuilt = pd.read_csv(table_dir / "Table_A3_full_results_with_deltas.csv")
            self.assertEqual(len(rebuilt), 532)


if __name__ == "__main__":
    unittest.main()
