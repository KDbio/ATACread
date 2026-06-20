import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyBigWig
except ImportError:  # pragma: no cover - optional dependency
    pyBigWig = None

from atacread.read import RNAReader, attach_exon_intervals, GTFAnnotationCache
from atacread.signal_utils import (
    binned_permutation_test,
    _comparison_items,
    _deviation_items,
    overall_deviation_tests,
    plot_gene_signals,
)
from atacread.cli import main as cli_main


class SignalStatisticsTest(unittest.TestCase):
    def test_paired_sign_flip_uses_configurable_thresholds(self):
        high = np.full(400, 2.0)
        low = np.full(400, 1.0)

        relaxed = binned_permutation_test(
            high,
            low,
            n_permutations=200,
            significance_level=0.10,
            log2fc_threshold=0.25,
        )
        strict_effect = binned_permutation_test(
            high,
            low,
            n_permutations=200,
            significance_level=0.10,
            log2fc_threshold=1.5,
        )

        self.assertEqual(relaxed["test"], "paired_binned_sign_flip")
        self.assertTrue(relaxed["significant"])
        self.assertEqual(relaxed["change_call"], "higher_a")
        self.assertFalse(strict_effect["significant"])

    def test_multiple_pairwise_calls_are_numbered_for_plotting(self):
        results = pd.DataFrame([
            {"comparison_id": 1, "sample_a": "A", "sample_b": "B", "significant": True},
            {"comparison_id": 2, "sample_a": "A", "sample_b": "C", "significant": False},
            {"comparison_id": 3, "sample_a": "B", "sample_b": "C", "significant": True},
        ])
        self.assertEqual(
            _comparison_items(results),
            ["1 A vs B: YES", "2 A vs C: NO", "3 B vs C: YES"],
        )

        deviation = overall_deviation_tests(
            {
                "A": np.ones(400),
                "B": np.full(400, 1.05),
                "C": np.full(400, 3.0),
            },
            n_permutations=200,
        )
        self.assertEqual(
            _deviation_items(deviation),
            ["1 A: NO", "2 B: NO", "3 C: YES"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "comparisons.png"
            signals = {
                "A": np.linspace(0, 1, 100),
                "B": np.linspace(0, 2, 100),
                "C": np.linspace(0, 0.5, 100),
            }
            plot_gene_signals(
                "GENE1",
                signals,
                signals,
                signals,
                output,
                comparison_results={
                    "atac_promoter": results,
                    "atac_genebody": results,
                    "rna": results,
                },
                deviation_results={
                    "atac_promoter": deviation,
                    "atac_genebody": deviation,
                    "rna": deviation,
                },
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)


class GTFCacheTest(unittest.TestCase):
    def test_cache_reuses_gene_and_exon_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gtf = tmp / "genes.gtf"
            cache_path = tmp / "genes.sqlite"
            gtf.write_text(
                "chr1\ttest\tgene\t101\t300\t.\t+\t.\t"
                'gene_id "GENE1.1"; gene_name "GENE1"; gene_type "protein_coding";\n'
                "chr1\ttest\texon\t101\t150\t.\t+\t.\t"
                'gene_id "GENE1.1"; transcript_id "TX1"; gene_name "GENE1";\n'
                "chr1\ttest\texon\t201\t250\t.\t+\t.\t"
                'gene_id "GENE1.1"; transcript_id "TX1"; gene_name "GENE1";\n'
                "chr2\ttest\tgene\t501\t700\t.\t-\t.\t"
                'gene_id "GENE2.1"; gene_name "GENE2"; gene_type "lncRNA";\n'
                "chr2\ttest\texon\t601\t700\t.\t-\t.\t"
                'gene_id "GENE2.1"; transcript_id "TX2"; gene_name "GENE2";\n',
                encoding="utf-8",
            )

            cache = GTFAnnotationCache(gtf, cache_file=cache_path)
            first = cache.read(
                queries=["GENE1"],
                promoter_upstream=50,
                promoter_downstream=20,
            )
            first_mtime = cache_path.stat().st_mtime_ns
            second = cache.read(
                queries=["GENE1"],
                promoter_upstream=80,
                promoter_downstream=30,
            )

            self.assertEqual(first_mtime, cache_path.stat().st_mtime_ns)
            self.assertEqual(int(first.iloc[0]["exon_length"]), 100)
            self.assertEqual(int(first.iloc[0]["promoter_start"]), 50)
            self.assertEqual(int(second.iloc[0]["promoter_start"]), 20)
            self.assertEqual(len(cache.read(indices=[1])), 1)

            cli_main(["gtf-index", "--gtf", str(gtf)])
            self.assertTrue(Path(str(gtf) + ".atacread.sqlite").exists())


@unittest.skipUnless(pyBigWig is not None, "requires pyBigWig")
class RNAExonReaderTest(unittest.TestCase):
    def test_rna_reader_excludes_introns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gtf = tmp / "genes.gtf"
            bw_path = tmp / "rna.bw"
            gtf.write_text(
                "chr1\ttest\tgene\t1\t100\t.\t+\t.\t"
                'gene_id "GENE1.1"; gene_name "GENE1"; gene_type "protein_coding";\n'
                "chr1\ttest\texon\t1\t10\t.\t+\t.\t"
                'gene_id "GENE1.1"; transcript_id "TX1"; gene_name "GENE1";\n'
                "chr1\ttest\texon\t91\t100\t.\t+\t.\t"
                'gene_id "GENE1.1"; transcript_id "TX1"; gene_name "GENE1";\n',
                encoding="utf-8",
            )

            with pyBigWig.open(str(bw_path), "w") as bw:
                bw.addHeader([("chr1", 100)])
                bw.addEntries(
                    ["chr1", "chr1", "chr1"],
                    [0, 10, 90],
                    ends=[10, 90, 100],
                    values=[2.0, 100.0, 2.0],
                )

            genes = pd.DataFrame([{
                "gene_id": "GENE1.1",
                "gene_id_base": "GENE1",
                "chrom": "chr1",
                "start": 0,
                "end": 100,
                "strand": "+",
            }])
            genes = attach_exon_intervals(gtf, genes)
            gene = genes.iloc[0].to_dict()

            with RNAReader([bw_path], sample_names=["RNA1"]) as reader:
                signal = reader.fetch_gene(gene)["rna_RNA1_gene_body_signal"]

            self.assertEqual(len(signal), 20)
            self.assertTrue(np.allclose(signal, 2.0))
            self.assertEqual(int(genes.iloc[0]["exon_length"]), 20)


if __name__ == "__main__":
    unittest.main()
