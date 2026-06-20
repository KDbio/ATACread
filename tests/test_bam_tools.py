import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pyBigWig
    import pysam
except ImportError:  # pragma: no cover - optional dependency
    pyBigWig = None
    pysam = None

from atacread.bam_tools import run_bam_downstream
from atacread.cli import main as cli_main


@unittest.skipUnless(pysam is not None and pyBigWig is not None,
                     "requires pysam and pyBigWig")
class BamToolsIntegrationTest(unittest.TestCase):
    def _pair(self, name, start, fragment_length=100, mapq=60,
              duplicate=False):
        read_length = 50
        mate_start = start + fragment_length - read_length
        seq = "A" * read_length
        qual = pysam.qualitystring_to_array("I" * read_length)

        read1 = pysam.AlignedSegment()
        read1.query_name = name
        read1.query_sequence = seq
        read1.flag = 99 | (1024 if duplicate else 0)
        read1.reference_id = 0
        read1.reference_start = start
        read1.mapping_quality = mapq
        read1.cigar = ((0, read_length),)
        read1.next_reference_id = 0
        read1.next_reference_start = mate_start
        read1.template_length = fragment_length
        read1.query_qualities = qual

        read2 = pysam.AlignedSegment()
        read2.query_name = name
        read2.query_sequence = seq
        read2.flag = 147 | (1024 if duplicate else 0)
        read2.reference_id = 0
        read2.reference_start = mate_start
        read2.mapping_quality = mapq
        read2.cigar = ((0, read_length),)
        read2.next_reference_id = 0
        read2.next_reference_start = start
        read2.template_length = -fragment_length
        read2.query_qualities = qual
        return read1, read2

    def _write_bam(self, path):
        header = {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 5000}],
        }
        records = []
        for i in range(10):
            records.extend(self._pair(f"peak_{i}", 2450 + i))
        for i, start in enumerate((500, 520, 4380, 4400)):
            records.extend(self._pair(f"edge_{i}", start))
        records.extend(self._pair("middle", 1500))
        records.extend(self._pair("duplicate", 2460, duplicate=True))
        records.extend(self._pair("low_mapq", 2470, mapq=5))
        records.sort(key=lambda r: (r.reference_id, r.reference_start, r.is_read2))
        with pysam.AlignmentFile(path, "wb", header=header) as bam:
            for read in records:
                bam.write(read)

    def test_bam_qc_frip_tss_count_and_bigwig(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bam = tmp / "synthetic.bam"
            peaks = tmp / "peaks.bed"
            tss = tmp / "tss.bed"
            out = tmp / "out"
            self._write_bam(bam)
            peaks.write_text("chr1\t2400\t2600\tpeak1\n", encoding="utf-8")
            tss.write_text("chr1\t2500\t2501\tgene1\t0\t+\n", encoding="utf-8")

            result = run_bam_downstream(
                [bam],
                regions_bed=peaks,
                output_dir=out,
                sample_names=["sample1"],
                min_mapq=30,
                make_bigwig=True,
                bigwig_bin_size=25,
                tss_regions=tss,
                auto_index=True,
            )

            self.assertTrue((tmp / "synthetic.bam.bai").exists())
            self.assertTrue(Path(result["qc_csv"]).exists())
            self.assertTrue(Path(result["fragment_summary_csv"]).exists())
            self.assertTrue(Path(result["frip_summary_csv"]).exists())
            self.assertTrue(Path(result["tss_enrichment_summary_csv"]).exists())
            self.assertTrue(Path(result["count_matrix_csv"]).exists())
            self.assertEqual(int(result["count_matrix"].loc[0, "sample1"]), 10)

            import pandas as pd

            frip = pd.read_csv(result["frip_summary_csv"]).iloc[0]
            self.assertEqual(int(frip["usable_units"]), 15)
            self.assertEqual(int(frip["in_peak_units"]), 10)
            self.assertAlmostEqual(float(frip["frip"]), 10 / 15, places=6)

            tss_summary = pd.read_csv(result["tss_enrichment_summary_csv"]).iloc[0]
            self.assertTrue(np.isfinite(float(tss_summary["tss_enrichment"])))
            self.assertGreater(float(tss_summary["tss_enrichment"]), 1.0)

            bw_path = Path(result["bigwig_files"][0])
            self.assertTrue(bw_path.exists())
            with pyBigWig.open(str(bw_path)) as bw:
                values = np.asarray(bw.values("chr1", 2400, 2600), dtype=float)
                self.assertGreater(float(np.nanmax(values)), 0.0)

    def test_sample_name_count_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            bam = Path(tmp) / "synthetic.bam"
            self._write_bam(bam)
            with self.assertRaisesRegex(ValueError, "数量"):
                run_bam_downstream(
                    [bam, bam],
                    output_dir=Path(tmp) / "out",
                    sample_names=["only_one"],
                    make_bigwig=False,
                )

    def test_cli_allows_qc_without_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bam = tmp / "synthetic.bam"
            out = tmp / "qc_only"
            self._write_bam(bam)
            cli_main([
                "bam",
                "--bam", str(bam),
                "--sample-names", "sample1",
                "--no-bigwig",
                "-o", str(out),
            ])
            self.assertTrue((out / "bam_qc_summary.csv").exists())
            self.assertTrue((out / "fragment_summary.csv").exists())
            self.assertFalse((out / "count_matrix.csv").exists())


if __name__ == "__main__":
    unittest.main()
