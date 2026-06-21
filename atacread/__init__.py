"""ATACread: ATAC/RNA bigWig reading, plotting, and raw-signal statistics."""

from .multiomics import (
    extract_multiomics_features,
    run_catalog,
    run_profile,
    run_paired,
)
from .bam_tools import (
    bam_qc,
    fragment_length_distribution,
    frip_score,
    tss_enrichment,
    count_matrix_from_bams,
    bam_to_bigwig,
    pydeseq2_differential,
    run_bam_downstream,
)
from .read import FastaIndex, GTFAnnotationCache, validate_bigwig_files

__version__ = "0.7.0"

__all__ = [
    "extract_multiomics_features",
    "run_catalog",
    "run_profile",
    "run_paired",
    "bam_qc",
    "fragment_length_distribution",
    "frip_score",
    "tss_enrichment",
    "count_matrix_from_bams",
    "bam_to_bigwig",
    "pydeseq2_differential",
    "run_bam_downstream",
    "FastaIndex",
    "GTFAnnotationCache",
    "validate_bigwig_files",
]
