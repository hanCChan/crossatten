"""Complete and incomplete Chinese reporting, using synthetic artifacts only."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s07_conclusion_test", ROOT / "07-wifi-osu/07-19-conclusion.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ConclusionTest(unittest.TestCase):
    def test_incomplete_stage_cannot_generate_conclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            here = Path(temp)
            (here / "acceptance.json").write_text(json.dumps(dict(status="PASS")))
            (here / "stage_status.json").write_text(json.dumps(dict(wifi_complete=False, osu_complete=True)))
            with patch.object(report, "HERE", here), self.assertRaisesRegex(ValueError, "complete audited"):
                report.main()
            self.assertFalse((here / "CONCLUSION_ZH.md").exists())

    def test_all_pipelines_and_negative_osu_are_disclosed(self):
        with tempfile.TemporaryDirectory() as temp:
            here = Path(temp)
            (here / "acceptance.json").write_text(json.dumps(dict(status="PASS")))
            (here / "stage_status.json").write_text(json.dumps(dict(wifi_complete=True, osu_complete=True)))
            rows, contrasts = [], []
            for dataset in ("ManyRx", "ManySig"):
                for pipeline in report.PIPELINES:
                    for task in ("CR", "CRD"):
                        rows.append(dict(
                            dataset=dataset, pipeline=pipeline, task=task, rx_accuracy_pct=85,
                            rx_macro_f1_pct=83, seed_sd_pp=1, worst_rx_seed_mean_pct=60,
                            parameters="90420", packet_accuracy_pct=84))
                    for baseline in ("W1", "IB-RFF", "WaveMLP", "FAN_source_selected"):
                        contrasts.append(dict(
                            dataset=dataset, pipeline=pipeline, baseline=baseline, delta_pp=-3,
                            ci95_low_pp=-4, ci95_high_pp=-2, improved_receivers=1, receiver_count=12))
            write_csv(here / "wifi_final_metrics.csv", rows)
            write_csv(here / "wifi_comparison_intervals.csv", contrasts)
            write_csv(here / "osu_summary.csv", [
                dict(method="hstem_hstu", accuracy_pct=1.42, tx_equal_pct=2,
                     macro_f1_pct=1, params=143367)])
            (here / "costs.json").write_text(json.dumps(dict(
                gpu="synthetic GPU", torch="synthetic", rows=[
                    dict(model="WiSig:ManySig:frequency:pooled_concat", params=90420, mflops=1,
                         cpu4t_b1_ms=1, gpu_b1_ms=1, gpu_b1_p95_ms=2,
                         gpu_b256_packets_per_second=1000)])))
            with patch.object(report, "HERE", here):
                report.main()
            text = (here / "CONCLUSION_ZH.md").read_text()
            for label in report.PIPELINES.values():
                self.assertIn(label, text)
            self.assertIn("1.42", text)
            self.assertIn("不是配对种子检验", text)
            self.assertIn("参数少不等于推理快", text)
            self.assertIn("最终配置相同", text)
            self.assertIn("不认证全球 SOTA", text)
            self.assertIn("85.00% / 85.00%", text)
            self.assertIn("-3.00 / -3.00 pp", text)
            self.assertIn("两项均值仍低于 IB-RFF 和 WaveMLP", text)
            self.assertIn("本轮没有达成 WiSig / OSU 双 SOTA", text)


if __name__ == "__main__":
    unittest.main()
