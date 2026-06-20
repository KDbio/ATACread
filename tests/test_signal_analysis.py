import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyBigWig
except ImportError:  # pragma: no cover - optional dependency
    pyBigWig = None

from atacread.read import RNAReader, attach_exon_intervals
from atacread.signal_utils import binned_permutation_test


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
