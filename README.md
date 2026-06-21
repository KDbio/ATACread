# ATACread

ATACread is a Python package for gene-level ATAC-seq/RNA-seq bigWig reading,
plotting, raw-signal comparison, and BAM-derived ATAC quality control.

RNA profiles use the merged union of all GTF exons when exon annotations are
available, so introns do not dilute conventional RNA-seq comparisons. Raw
signal comparisons use a paired binned sign-flip test and report p-values plus
effect sizes. Exploratory defaults are `p <= 0.10` and
`abs(log2 fold change) >= 0.25`.

For three or more tracks, each sample is also tested against the pointwise
median profile of the group. Plot captions show both overall-deviation calls
and numbered pairwise calls. Results are written to
`overall_deviation_tests.csv` and `raw_permutation_tests.csv`.
Pairwise calls are omitted from the figure when a panel contains more than
three tracks, but all pairwise rows remain available in the CSV.

When RNA analysis first uses a GTF, ATACread creates a sidecar SQLite index
named `<annotation.gtf>.atacread.sqlite`. The first build scans the annotation
once; later gene-name, Ensembl-ID, and gene-index queries reuse the cache. It
is rebuilt automatically when the source GTF size or modification time changes.

FASTA access uses a standard `<genome.fa>.fai` index. The first run builds it;
later runs seek directly to requested chromosomes instead of scanning the
whole genome FASTA.

The index can also be prepared before an analysis:

```bash
atacread gtf-index --gtf annotation.gtf
atacread fasta-index --fasta genome.fa
```

## Installation

```bash
pip install "ATACread[bigwig] @ git+https://github.com/KDbio/ATACread.git"
```

Install BAM QC, bigWig conversion, and PyDESeq2 support:

```bash
pip install "ATACread[all] @ git+https://github.com/KDbio/ATACread.git"
```

## Main commands

```text
atacread catalog   Full-GTF gene summary table
atacread profile   Multi-gene plots and raw-signal permutation tests
atacread paired    Paired ATAC/RNA direction analysis
atacread bam       BAM QC, fragment length, FRiP, TSS, counts, and bigWig
atacread deseq2    PyDESeq2 analysis of an integer count matrix
atacread gtf-index Build or validate the reusable GTF SQLite index
atacread fasta-index Build or validate the standard FASTA .fai index
```

Multiple genes can be supplied as a text file containing one gene name,
Ensembl ID, or GTF gene index per line:

```bash
atacread profile -f genes.txt --gtf annotation.gtf --fasta genome.fa \
  --atac "atac1.bw,atac2.bw" --rna "rna1.bw,rna2.bw" \
  --significance-level 0.10 --lfc-threshold 0.25
```

RNA uses the union of all annotated exons by default. To analyze one explicit
transcript per selected gene:

```bash
atacread profile -g "GENE1,GENE2" --gtf annotation.gtf --fasta genome.fa \
  --rna "rna1.bw,rna2.bw" --rna-region-mode transcript \
  --transcripts "ENST000001,ENST000002"
```

Versionless transcript IDs are accepted. `paired` uses the same options and
reuses the feature data already produced for its profile plots.

Run the full BAM workflow:

```bash
atacread bam \
  --bam "sample1.bam,sample2.bam" \
  --sample-names "sample1,sample2" \
  --regions consensus_peaks.bed \
  --tss-regions genes_tss.bed \
  --bigwig-bin-size 50 \
  -o output_bam
```

Coordinate-sorted BAM files are required. Missing BAM indexes are created
automatically unless `--no-auto-index` is used. PyDESeq2 requires raw integer
counts from BAM/peak regions; bigWig signal values are not count data.

The detailed Chinese project notes and complete examples are in
`PYTHON ATAC RNA-seq一站式读取.md`.
