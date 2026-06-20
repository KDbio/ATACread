import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

try:
    import pyBigWig
except ImportError:  # pragma: no cover - optional dependency
    pyBigWig = None

from atacread.read import (
    RNAReader,
    attach_exon_intervals,
    GTFAnnotationCache,
    FastaIndex,
    fasta_read,
    configure_rna_regions,
)
from atacread.signal_utils import (
    binned_permutation_test,
    _comparison_items,
    _deviation_items,
    overall_deviation_tests,
    plot_gene_signals,
)
from atacread.cli import main as cli_main
from atacread.multiomics import run_paired


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
                "chr1\ttest\texon\t101\t120\t.\t+\t.\t"
                'gene_id "GENE1.1"; transcript_id "TXALT.2"; gene_name "GENE1";\n'
                "chr1\ttest\texon\t221\t250\t.\t+\t.\t"
                'gene_id "GENE1.1"; transcript_id "TXALT.2"; gene_name "GENE1";\n'
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

            union = configure_rna_regions(first, mode="exon_union")
            transcript = configure_rna_regions(
                first,
                mode="transcript",
                transcript_ids=["TXALT"],
            )
            self.assertEqual(int(union.iloc[0]["rna_length"]), 100)
            self.assertEqual(int(transcript.iloc[0]["rna_length"]), 50)
            self.assertEqual(transcript.iloc[0]["rna_transcript_id"], "TXALT.2")

            cli_main(["gtf-index", "--gtf", str(gtf)])
            self.assertTrue(Path(str(gtf) + ".atacread.sqlite").exists())


class FastaIndexTest(unittest.TestCase):
    def test_fai_build_and_targeted_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fasta = tmp / "genome.fa"
            fasta.write_bytes(
                b">chr1 description\nACGT\nACGT\nAC\n"
                b">chr2\nTTTT\nGGGG\n"
            )
            index = FastaIndex(fasta)
            index.build()
            first_mtime = Path(index.index_file).stat().st_mtime_ns

            result = fasta_read(fasta, keep_chroms=["chr2"], use_cache=False)
            self.assertEqual(set(result), {"chr2"})
            self.assertEqual(result["chr2"]["+"], "TTTTGGGG")
            self.assertEqual(index.fetch_chromosome("chr1"), "ACGTACGTAC")

            index.build()
            self.assertEqual(first_mtime, Path(index.index_file).stat().st_mtime_ns)
            cli_main(["fasta-index", "--fasta", str(fasta)])


class PairedReuseTest(unittest.TestCase):
    def test_paired_builds_features_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            metadata = tmp / "metadata.csv"
            metadata.write_text(
                "sample,assay,group\n"
                "A1,ATAC,control\nA2,ATAC,treat\n"
                "R1,RNA,control\nR2,RNA,treat\n",
                encoding="utf-8",
            )
            features = pd.DataFrame([{
                "gene_id": "GENE1.1",
                "gene_name": "GENE1",
                "chrom": "chr1",
                "strand": "+",
                "promoter_seq": "A" * 20,
                "atac_A1_promoter_signal": np.ones(40),
                "atac_A2_promoter_signal": np.full(40, 2.0),
                "atac_A1_gene_body_signal": np.ones(40),
                "atac_A2_gene_body_signal": np.full(40, 2.0),
                "rna_R1_gene_body_signal": np.ones(40),
                "rna_R2_gene_body_signal": np.full(40, 2.0),
            }])
            bundle = (features, ["A1", "A2"], ["R1", "R2"])

            with mock.patch(
                "atacread.multiomics._build_features",
                return_value=bundle,
            ) as build_features, mock.patch(
                "atacread.multiomics.plot_gene_signals"
            ):
                run_paired(
                    "genes.gtf",
                    "genome.fa",
                    ["a1.bw", "a2.bw"],
                    ["r1.bw", "r2.bw"],
                    metadata,
                    genes=["GENE1"],
                    output_dir=tmp / "out",
                    atac_names=["A1", "A2"],
                    rna_names=["R1", "R2"],
                    n_permutations=20,
                )
            self.assertEqual(build_features.call_count, 1)


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

            gene["rna_intervals"] = [(0, 10)]
            with RNAReader([bw_path], sample_names=["RNA1"]) as reader:
                transcript_signal = reader.fetch_gene(gene)["rna_RNA1_gene_body_signal"]
            self.assertEqual(len(transcript_signal), 10)
            self.assertTrue(np.allclose(transcript_signal, 2.0))


if __name__ == "__main__":
    unittest.main()
