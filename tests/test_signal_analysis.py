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
    BigWigReader,
    RNAReader,
    attach_exon_intervals,
    GTFAnnotationCache,
    FastaIndex,
    fasta_read,
    configure_rna_regions,
    validate_bigwig_files,
)
from atacread.signal_utils import (
    binned_permutation_test,
    _comparison_items,
    _displayed_comparison_items,
    _deviation_items,
    overall_deviation_tests,
    plot_gene_signals,
    resolve_genes,
)
from atacread.cli import (
    _automatic_output_dir,
    _prepare_output_dir,
    _unique_sample_names,
    main as cli_main,
)
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
        self.assertEqual(_displayed_comparison_items(results), _comparison_items(results))

        four_sample_results = pd.DataFrame([
            {"sample_a": "A", "sample_b": "B", "significant": False},
            {"sample_a": "A", "sample_b": "C", "significant": False},
            {"sample_a": "A", "sample_b": "D", "significant": True},
            {"sample_a": "B", "sample_b": "C", "significant": False},
            {"sample_a": "B", "sample_b": "D", "significant": True},
            {"sample_a": "C", "sample_b": "D", "significant": True},
        ])
        self.assertEqual(_displayed_comparison_items(four_sample_results), [])

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

            with self.assertRaisesRegex(ValueError, "MISSING_GENE"):
                resolve_genes(gtf, genes=["GENE1", "MISSING_GENE"])
            with self.assertRaisesRegex(ValueError, "999"):
                resolve_genes(gtf, genes=[999], include_exons=True)


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


class AutoCliTest(unittest.TestCase):
    def _write_reference_files(self, folder):
        (folder / "genes.gtf").write_text(
            'chr1\ttest\tgene\t1\t1\t.\t+\t.\tgene_id "G1"; gene_name "G1";\n',
            encoding="utf-8",
        )
        (folder / "genome.fa").write_text(">chr1\nA\n", encoding="utf-8")

    def test_automatic_output_name_and_collision(self):
        self.assertEqual(
            _automatic_output_dir("compare", atac_files=["sample_ATAC1.bigWig"]),
            "output_sample_ATAC1_compare",
        )
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "output_sample_compare"
            first.mkdir()
            second = _prepare_output_dir(first, make_unique=True)
            self.assertEqual(second.name, "output_sample_compare_2")

    def test_duplicate_stems_get_unique_sample_names(self):
        names = _unique_sample_names([
            "/data/control/sample.bw",
            "/data/treated/sample.bw",
        ])
        self.assertEqual(names, ["control_sample", "treated_sample"])

    def test_auto_compare_discovers_files_and_output_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "experiment"
            data.mkdir()
            self._write_reference_files(root)
            (data / "sample_ATAC1.bigWig").touch()
            (data / "sample_ATAC2.bigWig").touch()
            (data / "sample_RNA1.bigWig").touch()

            output = root / "compare_output"
            with mock.patch("atacread.cli.validate_bigwig_files"), mock.patch(
                "atacread.cli.run_profile"
            ) as run_profile:
                cli_main([
                    "auto", "--task", "compare", "--data", str(data),
                    "--genes", "POU5F1",
                    "--output-dir", str(output),
                ])

            kwargs = run_profile.call_args.kwargs
            self.assertEqual(kwargs["genes"], ["POU5F1"])
            self.assertEqual(kwargs["output_dir"], str(output))
            self.assertEqual(len(kwargs["atac_files"]), 2)
            self.assertEqual(len(kwargs["rna_files"]), 1)
            self.assertEqual(kwargs["n_permutations"], 200)

    def test_auto_catalog_uses_hidden_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            self._write_reference_files(data)
            (data / "batch_ATAC.bigWig").touch()
            (data / "batch_RNA.bigWig").touch()

            output = data / "catalog_output"
            with mock.patch("atacread.cli.validate_bigwig_files"), mock.patch(
                "atacread.cli.run_catalog"
            ) as run_catalog:
                cli_main([
                    "auto", "--task", "catalog", "--data", str(data),
                    "--output-dir", str(output),
                ])

            kwargs = run_catalog.call_args.kwargs
            self.assertEqual(kwargs["output_dir"], str(output))
            self.assertEqual(kwargs["promoter_upstream"], 200)
            self.assertEqual(kwargs["promoter_downstream"], 200)

    def test_auto_bam_runs_qc_and_bigwig_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / "sample_ATAC.bam").write_bytes(b"BAM")
            (data / "consensus_peaks.bed").write_text("chr1\t0\t1\n", encoding="utf-8")
            metadata = data / "metadata.csv"
            metadata.write_text(
                "sample,group\nsample_ATAC,control\n",
                encoding="utf-8",
            )
            count_matrix = data / "count_matrix.csv"
            output = data / "bam_output"

            with mock.patch(
                "atacread.cli.run_bam_downstream",
                return_value={"count_matrix_csv": str(count_matrix)},
            ) as run_bam, mock.patch(
                "atacread.cli.pydeseq2_differential"
            ) as run_deseq2:
                cli_main([
                    "auto", "--task", "bam", "--data", str(data),
                    "--output-dir", str(output),
                ])

            kwargs = run_bam.call_args.kwargs
            self.assertEqual(kwargs["output_dir"], str(output))
            self.assertTrue(kwargs["make_bigwig"])
            self.assertEqual(kwargs["bigwig_bin_size"], 50)
            self.assertTrue(str(kwargs["regions_bed"]).endswith("consensus_peaks.bed"))
            self.assertEqual(run_deseq2.call_args.kwargs["condition_col"], "group")

    def test_failure_writes_manifest_and_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            self._write_reference_files(data)
            (data / "sample_ATAC.bigWig").touch()
            output = data / "failed_output"
            with mock.patch("atacread.cli.validate_bigwig_files"), mock.patch(
                "atacread.cli.run_profile", side_effect=RuntimeError("simulated failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                    cli_main([
                        "auto", "--task", "compare", "--data", str(data),
                        "--genes", "G1", "--output-dir", str(output),
                    ])
            manifest = pd.read_json(output / "run_manifest.json", typ="series")
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("simulated failure", (output / "run_error.log").read_text())


@unittest.skipUnless(pyBigWig is not None, "requires pyBigWig")
class BigWigValidationTest(unittest.TestCase):
    def test_reference_length_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fasta = tmp / "genome.fa"
            bigwig = tmp / "sample.bw"
            fasta.write_text(">chr1\n" + "A" * 100 + "\n", encoding="utf-8")
            with pyBigWig.open(str(bigwig), "w") as bw:
                bw.addHeader([("chr1", 90)])
                bw.addEntries(["chr1"], [0], ends=[90], values=[1.0])
            with self.assertRaisesRegex(ValueError, "参考基因组版本"):
                validate_bigwig_files([bigwig], fasta_file=fasta)

    def test_duplicate_sample_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = []
            for index in range(2):
                path = tmp / f"sample{index}.bw"
                with pyBigWig.open(str(path), "w") as bw:
                    bw.addHeader([("chr1", 10)])
                    bw.addEntries(["chr1"], [0], ends=[10], values=[1.0])
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "重复"):
                BigWigReader(paths, sample_names=["same", "same"])


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
