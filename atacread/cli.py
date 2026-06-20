"""Command line interface for ATACread."""

import argparse
from pathlib import Path

from .multiomics import run_catalog, run_profile, run_paired
from .bam_tools import run_bam_downstream, pydeseq2_differential


def _split_csv(value):
    if value is None or value == "":
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


def auto_find(work_dir, suffixes, keywords=None):
    results = []
    for p in Path(work_dir).rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not any(name.endswith(s) for s in suffixes):
            continue
        if keywords and not any(k.lower() in name for k in keywords):
            continue
        results.append(str(p))
    return sorted(results)


def detect_inputs(work_dir, gtf=None, fasta=None, atac=None, rna=None):
    if not gtf:
        cands = auto_find(work_dir, [".gtf", ".gtf.gz"])
        gtf = cands[0] if cands else None
    if not fasta:
        cands = auto_find(work_dir, [".fa", ".fasta", ".fna"])
        fasta = cands[0] if cands else None
    if not atac:
        atac = auto_find(work_dir, [".bw", ".bigwig"], keywords=["atac"])
    else:
        atac = _split_csv(atac) or []
    if not rna:
        rna = auto_find(work_dir, [".bw", ".bigwig"], keywords=["rna"])
    else:
        rna = _split_csv(rna) or []

    print(f"GTF   : {gtf}")
    print(f"FASTA : {fasta}")
    print(f"ATAC  : {len(atac)} files")
    print(f"RNA   : {len(rna)} files")
    return gtf, fasta, atac, rna


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="atacread",
        description="ATAC/RNA bigWig reading, gene-region summaries, plots, and raw-signal tests.",
    )
    parser.add_argument(
        "mode",
        choices=["catalog", "profile", "paired", "bam", "deseq2", "gtf-index"],
    )
    parser.add_argument("-w", "--work-dir", default=".", help="Folder used for automatic file discovery.")
    parser.add_argument("-o", "--output-dir", default=None, help="Output folder.")
    parser.add_argument("-g", "--genes", default=None, help="Comma-separated gene names, gene IDs, or gene indices.")
    parser.add_argument("-f", "--gene-file", default=None, help="Text file with one gene name/ID/index per line.")
    parser.add_argument("-m", "--metadata", default=None, help="Metadata CSV for paired/deseq2 modes.")
    parser.add_argument("--gtf", default=None, help="GTF annotation file.")
    parser.add_argument("--fasta", default=None, help="Genome FASTA file.")
    parser.add_argument("--atac", default=None, help="Comma-separated ATAC bigWig files.")
    parser.add_argument("--rna", default=None, help="Comma-separated RNA bigWig files.")
    parser.add_argument("--atac-names", default=None, help="Comma-separated ATAC sample names.")
    parser.add_argument("--rna-names", default=None, help="Comma-separated RNA sample names.")
    parser.add_argument("--promoter-upstream", type=int, default=200, help="Promoter upstream length.")
    parser.add_argument("--promoter-downstream", type=int, default=200, help="Promoter downstream length.")
    parser.add_argument("--bin-size", default="auto", help="Permutation-test bin size: auto or an integer.")
    parser.add_argument("--n-permutations", type=int, default=200, help="Permutation count for raw-signal tests.")
    parser.add_argument("--significance-level", type=float, default=0.10,
                        help="Exploratory p-value cutoff (default: 0.10).")
    parser.add_argument("--lfc-threshold", type=float, default=0.25,
                        help="Minimum absolute log2 fold change (default: 0.25).")
    parser.add_argument("--pca", action="store_true", help="Run RNA PCA in paired mode.")
    parser.add_argument("--no-classify", action="store_true", help="Do not classify gene states in catalog mode.")
    parser.add_argument("--rebuild-gtf-cache", action="store_true",
                        help="Force rebuilding the GTF SQLite index.")

    parser.add_argument("--bam", default=None, help="Comma-separated BAM files for BAM mode.")
    parser.add_argument("--regions", default=None, help="BED regions for BAM counting.")
    parser.add_argument("--tss-regions", default=None,
                        help="BED or GTF used for BAM TSS enrichment.")
    parser.add_argument("--sample-names", default=None, help="Comma-separated sample names for BAM mode.")
    parser.add_argument("--min-mapq", type=int, default=30)
    parser.add_argument("--bigwig-bin-size", type=int, default=50)
    parser.add_argument("--no-bigwig", action="store_true")
    parser.add_argument("--no-auto-index", action="store_true",
                        help="Do not create a missing index for coordinate-sorted BAM files.")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Keep duplicate reads/fragments in BAM-derived analyses.")

    parser.add_argument("--count-matrix", default=None, help="Count matrix CSV for deseq2 mode.")
    parser.add_argument("--condition-col", default="condition")
    parser.add_argument("--reference-level", default=None, help="Reference:treatment, e.g. control:treat.")

    args = parser.parse_args(argv)
    out = args.output_dir or f"output_{args.mode}"
    genes = _split_csv(args.genes)
    atac_names = _split_csv(args.atac_names)
    rna_names = _split_csv(args.rna_names)

    if args.mode == "gtf-index":
        from .read import GTFAnnotationCache

        gtf = args.gtf
        if not gtf:
            candidates = auto_find(args.work_dir, [".gtf", ".gtf.gz"])
            gtf = candidates[0] if candidates else None
        if not gtf:
            raise ValueError("gtf-index mode requires --gtf or a GTF in --work-dir")
        cache = GTFAnnotationCache(gtf)
        print(cache.build(force=args.rebuild_gtf_cache))
        return

    if args.mode == "bam":
        bams = _split_csv(args.bam) if args.bam else auto_find(args.work_dir, [".bam"])
        outputs = run_bam_downstream(
            bams,
            args.regions,
            out,
            sample_names=_split_csv(args.sample_names),
            min_mapq=args.min_mapq,
            make_bigwig=not args.no_bigwig,
            bigwig_bin_size=args.bigwig_bin_size,
            tss_regions=args.tss_regions,
            auto_index=not args.no_auto_index,
            keep_duplicates=args.keep_duplicates,
        )
        print({k: v for k, v in outputs.items() if k != "count_matrix"})
        return

    if args.mode == "deseq2":
        if not args.count_matrix or not args.metadata:
            raise ValueError("deseq2 mode requires --count-matrix and --metadata")
        reference = tuple(args.reference_level.split(":")) if args.reference_level else None
        result = pydeseq2_differential(
            args.count_matrix,
            args.metadata,
            condition_col=args.condition_col,
            reference_level=reference,
            output_csv=str(Path(out) / "pydeseq2_results.csv"),
        )
        print(result.head())
        return

    gtf, fasta, atac, rna = detect_inputs(args.work_dir, args.gtf, args.fasta, args.atac, args.rna)

    if args.mode == "catalog":
        run_catalog(
            gtf,
            fasta,
            atac,
            rna,
            output_dir=out,
            promoter_upstream=args.promoter_upstream,
            promoter_downstream=args.promoter_downstream,
            atac_names=atac_names,
            rna_names=rna_names,
            classify_state=not args.no_classify,
        )
    elif args.mode == "profile":
        run_profile(
            gtf,
            fasta,
            atac,
            rna,
            genes=genes,
            gene_file=args.gene_file,
            output_dir=out,
            promoter_upstream=args.promoter_upstream,
            promoter_downstream=args.promoter_downstream,
            atac_names=atac_names,
            rna_names=rna_names,
            bin_size=args.bin_size,
            n_permutations=args.n_permutations,
            significance_level=args.significance_level,
            lfc_threshold=args.lfc_threshold,
        )
    elif args.mode == "paired":
        if not args.metadata:
            raise ValueError("paired mode requires --metadata")
        run_paired(
            gtf,
            fasta,
            atac,
            rna,
            args.metadata,
            genes=genes,
            gene_file=args.gene_file,
            output_dir=out,
            promoter_upstream=args.promoter_upstream,
            promoter_downstream=args.promoter_downstream,
            atac_names=atac_names,
            rna_names=rna_names,
            make_pca=args.pca,
            n_permutations=args.n_permutations,
            significance_level=args.significance_level,
            lfc_threshold=args.lfc_threshold,
        )


if __name__ == "__main__":
    main()
