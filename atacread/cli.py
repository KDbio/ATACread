"""Command line interface for ATACread."""

import argparse
import json
import os
import re
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from .multiomics import run_catalog, run_profile, run_paired
from .bam_tools import run_bam_downstream, pydeseq2_differential
from .read import validate_bigwig_files


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


def _find_data_files(data_dir, suffixes, keywords=None):
    """Prefer direct children; recurse only when the folder has no matches."""
    data_dir = Path(data_dir)

    def matches(path):
        name = path.name.lower()
        if not any(name.endswith(suffix.lower()) for suffix in suffixes):
            return False
        return not keywords or any(keyword.lower() in name for keyword in keywords)

    direct = sorted(str(path) for path in data_dir.iterdir() if path.is_file() and matches(path))
    return direct or auto_find(data_dir, suffixes, keywords=keywords)


def _find_reference_file(data_dir, suffixes):
    """Find GTF/FASTA in the data folder or up to three parent folders."""
    current = Path(data_dir).resolve()
    for _ in range(4):
        candidates = sorted(
            str(path) for path in current.iterdir()
            if path.is_file() and any(path.name.lower().endswith(s.lower()) for s in suffixes)
        )
        if candidates:
            return candidates[0]
        if current.parent == current:
            break
        current = current.parent
    return None


def _automatic_output_dir(task, atac_files=None, bam_files=None):
    sources = atac_files or bam_files or []
    stem = Path(sources[0]).stem if sources else "atacread"
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).strip(" ._")
    stem = stem[:120] or "atacread"
    return f"output_{stem}_{task}"


def _unique_sample_names(paths):
    """Create stable names without silently overwriting duplicate stems."""
    stems = [Path(path).stem for path in paths]
    counts = {stem: stems.count(stem) for stem in set(stems)}
    names = []
    used = set()
    for path, stem in zip(paths, stems):
        candidate = stem
        if counts[stem] > 1:
            candidate = f"{Path(path).parent.name}_{stem}"
        base = candidate
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        names.append(candidate)
    return names


def _prepare_output_dir(path, make_unique=False):
    candidate = Path(path)
    if candidate.exists() and candidate.is_file():
        raise ValueError(f"输出路径是文件而不是目录: {candidate}")
    if make_unique and candidate.exists():
        base = candidate
        suffix = 2
        while candidate.exists():
            candidate = Path(f"{base}_{suffix}")
            suffix += 1
        print(f"[auto] 默认输出目录已存在，改用: {candidate}")
    candidate.mkdir(parents=True, exist_ok=True)
    probe = candidate / ".atacread_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(f"输出目录不可写: {candidate}: {exc}") from exc
    return candidate


def _write_manifest(output_dir, payload):
    path = Path(output_dir) / "run_manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return str(path)


def _check_nonempty_files(paths, label):
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} 文件不存在: {path}")
        if Path(path).stat().st_size == 0:
            raise ValueError(f"{label} 文件为空: {path}")


def run_auto_task(task, data_dir, genes=None, gene_file=None, output_dir=None):
    """One-stop wrapper with conservative file discovery and stable defaults."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data 目录不存在: {data_dir}")

    task = str(task).lower()
    if task not in {"bam", "catalog", "compare"}:
        raise ValueError("--task 必须是 bam、catalog 或 compare")

    gtf = _find_reference_file(data_dir, [".gtf", ".gtf.gz"])
    fasta = _find_reference_file(data_dir, [".fa", ".fasta", ".fna"])
    atac = _find_data_files(data_dir, [".bw", ".bigwig"], keywords=["atac"])
    rna = _find_data_files(data_dir, [".bw", ".bigwig"], keywords=["rna"])
    bams = _find_data_files(data_dir, [".bam"])
    all_bigwigs = _find_data_files(data_dir, [".bw", ".bigwig"])
    unclassified_bigwigs = sorted(set(all_bigwigs) - set(atac) - set(rna))
    automatic_output = output_dir is None
    output_dir = output_dir or _automatic_output_dir(
        task, atac_files=atac if task != "bam" else None, bam_files=bams
    )
    output_dir = _prepare_output_dir(output_dir, make_unique=automatic_output)
    atac_names = _unique_sample_names(atac)
    rna_names = _unique_sample_names(rna)
    bam_names = _unique_sample_names(bams)

    print(f"[auto] task   : {task}")
    print(f"[auto] data   : {data_dir}")
    print(f"[auto] output : {output_dir}")
    print(f"[auto] ATAC/RNA/BAM: {len(atac)}/{len(rna)}/{len(bams)}")
    if unclassified_bigwigs:
        print(f"[auto] 警告: {len(unclassified_bigwigs)} 个 bigWig 文件名不含 ATAC/RNA，已忽略")

    warnings = []
    if unclassified_bigwigs:
        warnings.append(
            f"忽略 {len(unclassified_bigwigs)} 个文件名不含 ATAC/RNA 的 bigWig"
        )
    manifest = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "inputs": {
            "gtf": gtf,
            "fasta": fasta,
            "atac_bigwigs": atac,
            "rna_bigwigs": rna,
            "bam_files": bams,
            "unclassified_bigwigs": unclassified_bigwigs,
            "genes": genes,
            "gene_file": gene_file,
        },
        "sample_names": {
            "atac": atac_names,
            "rna": rna_names,
            "bam": bam_names,
        },
        "defaults": {
            "promoter_upstream": 200,
            "promoter_downstream": 200,
            "n_permutations": 200,
            "significance_level": 0.10,
            "lfc_threshold": 0.25,
            "rna_region_mode": "exon_union",
            "bam_min_mapq": 30,
            "bigwig_bin_size": 50,
        },
        "warnings": warnings,
    }
    _write_manifest(output_dir, manifest)

    try:
        if task == "bam":
            if not bams:
                raise FileNotFoundError(f"{data_dir} 中没有找到 BAM")
            _check_nonempty_files(bams, "BAM")
            free_space = shutil.disk_usage(output_dir).free
            bam_bytes = sum(Path(path).stat().st_size for path in bams)
            if free_space < bam_bytes:
                warning = "输出磁盘剩余空间小于 BAM 总大小，bigWig 生成可能失败"
                warnings.append(warning)
                print(f"[auto] 警告: {warning}")

            beds = _find_data_files(data_dir, [".bed", ".bed.gz"])
            peak_beds = [
                path for path in beds
                if any(word in Path(path).name.lower() for word in ("peak", "region", "consensus"))
            ]
            regions = peak_beds[0] if peak_beds else (beds[0] if beds else None)
            if regions:
                _check_nonempty_files([regions], "BED")
            tss_beds = [path for path in beds if "tss" in Path(path).name.lower()]
            tss_regions = gtf or (tss_beds[0] if tss_beds else None)
            outputs = run_bam_downstream(
                bams,
                regions_bed=regions,
                output_dir=str(output_dir),
                sample_names=bam_names,
                min_mapq=30,
                make_bigwig=True,
                bigwig_bin_size=50,
                tss_regions=tss_regions,
                auto_index=True,
                keep_duplicates=False,
            )

            metadata_candidates = _find_data_files(
                data_dir, [".csv"], keywords=["metadata", "design", "sample"]
            )
            count_matrix = outputs.get("count_matrix_csv")
            if count_matrix and metadata_candidates:
                deseq_csv = str(output_dir / "pydeseq2_results.csv")
                try:
                    metadata_columns = pd.read_csv(metadata_candidates[0], nrows=1).columns
                    condition_col = (
                        "condition" if "condition" in metadata_columns
                        else "group" if "group" in metadata_columns
                        else "condition"
                    )
                    pydeseq2_differential(
                        count_matrix,
                        metadata_candidates[0],
                        condition_col=condition_col,
                        output_csv=deseq_csv,
                    )
                    outputs["deseq2_csv"] = deseq_csv
                except Exception as exc:
                    warning = f"PyDESeq2 差异分析已跳过: {type(exc).__name__}: {exc}"
                    warnings.append(warning)
                    print(f"[auto] {warning}")
            else:
                warning = "未同时找到 peak BED 和 metadata CSV，跳过 PyDESeq2"
                warnings.append(warning)
                print(f"[auto] {warning}")
            result = outputs
        else:
            if not gtf or not fasta:
                raise FileNotFoundError(
                    "catalog/compare 需要 data 目录或其父目录中存在 GTF 和 FASTA"
                )
            _check_nonempty_files([gtf], "GTF")
            _check_nonempty_files([fasta], "FASTA")
            if not atac and not rna:
                raise FileNotFoundError(f"{data_dir} 中没有找到 ATAC/RNA bigWig")
            validate_bigwig_files(atac + rna, fasta_file=fasta)

            if task == "catalog":
                result = run_catalog(
                    gtf,
                    fasta,
                    atac_files=atac,
                    rna_files=rna,
                    output_dir=str(output_dir),
                    promoter_upstream=200,
                    promoter_downstream=200,
                    atac_names=atac_names or None,
                    rna_names=rna_names or None,
                )
            else:
                if not genes and not gene_file:
                    gene_files = _find_data_files(data_dir, [".txt"], keywords=["gene"])
                    gene_file = gene_files[0] if gene_files else None
                    manifest["inputs"]["gene_file"] = gene_file
                if not genes and not gene_file:
                    raise ValueError("compare 需要 --genes 或 data 目录中的 gene*.txt")
                if gene_file:
                    _check_nonempty_files([gene_file], "基因列表")
                result = run_profile(
                    gtf,
                    fasta,
                    atac_files=atac,
                    rna_files=rna,
                    genes=genes,
                    gene_file=gene_file,
                    output_dir=str(output_dir),
                    promoter_upstream=200,
                    promoter_downstream=200,
                    atac_names=atac_names or None,
                    rna_names=rna_names or None,
                    bin_size="auto",
                    n_permutations=200,
                    significance_level=0.10,
                    lfc_threshold=0.25,
                    rna_region_mode="exon_union",
                )
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        _write_manifest(output_dir, manifest)
        (output_dir / "run_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[auto] 运行失败: {exc}")
        print(f"[auto] 详情: {output_dir / 'run_error.log'}")
        raise

    manifest.update({
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "output_files": sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*") if path.is_file()
        ),
    })
    _write_manifest(output_dir, manifest)
    return result


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


def _validate_analysis_inputs(
    mode, gtf, fasta, atac, rna, genes=None, gene_file=None,
    atac_names=None, rna_names=None, metadata=None,
):
    """Fail at the CLI boundary with actionable input errors."""
    missing = [label for label, value in (("GTF", gtf), ("FASTA", fasta)) if not value]
    if missing:
        raise FileNotFoundError(
            f"{mode} 模式缺少 {', '.join(missing)}；请显式指定文件或检查 --work-dir"
        )
    _check_nonempty_files([gtf], "GTF")
    _check_nonempty_files([fasta], "FASTA")
    if not atac and not rna:
        raise FileNotFoundError(f"{mode} 模式至少需要一个 ATAC 或 RNA bigWig")
    validate_bigwig_files((atac or []) + (rna or []), fasta_file=fasta)

    for label, names, files in (
        ("ATAC", atac_names, atac or []),
        ("RNA", rna_names, rna or []),
    ):
        if names is not None and len(names) != len(files):
            raise ValueError(
                f"{label} 样本名数量 ({len(names)}) 与文件数量 ({len(files)}) 不一致"
            )

    if mode in {"profile", "paired"}:
        if gene_file:
            _check_nonempty_files([gene_file], "基因列表")
        if not genes and not gene_file:
            raise ValueError(f"{mode} 模式需要 --genes 或 --gene-file")
    if mode == "paired":
        if not metadata:
            raise ValueError("paired 模式需要 --metadata")
        _check_nonempty_files([metadata], "metadata")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="atacread",
        description="One-stop ATAC/RNA analysis plus advanced explicit modes.",
    )
    parser.add_argument(
        "mode",
        choices=[
            "catalog", "profile", "paired", "bam", "deseq2",
            "gtf-index", "fasta-index", "auto",
        ],
    )
    parser.add_argument("-w", "--work-dir", default=".", help="Folder used for automatic file discovery.")
    parser.add_argument("--task", choices=["bam", "catalog", "compare"], default=None,
                        help="One-stop task used by auto mode.")
    parser.add_argument("--data", "--data-dir", dest="data_dir", default=None,
                        help="Input folder used by auto mode.")
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
    parser.add_argument(
        "--rna-region-mode",
        choices=["exon_union", "transcript"],
        default="exon_union",
        help="RNA region: merged gene exons or explicitly selected transcripts.",
    )
    parser.add_argument(
        "--transcripts",
        default=None,
        help="Comma-separated transcript IDs used with --rna-region-mode transcript.",
    )
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
    parser.add_argument("--rebuild-fasta-index", action="store_true",
                        help="Force rebuilding the FASTA .fai index.")

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
    transcript_ids = _split_csv(args.transcripts)

    if args.mode == "auto":
        if not args.task or not args.data_dir:
            raise ValueError("auto 模式必须指定 --task 和 --data")
        run_auto_task(
            args.task,
            args.data_dir,
            genes=genes,
            gene_file=args.gene_file,
            output_dir=args.output_dir,
        )
        return

    if args.mode == "fasta-index":
        from .read import FastaIndex

        fasta = args.fasta
        if not fasta:
            candidates = auto_find(args.work_dir, [".fa", ".fasta", ".fna"])
            fasta = candidates[0] if candidates else None
        if not fasta:
            raise ValueError("fasta-index mode requires --fasta or a FASTA in --work-dir")
        index = FastaIndex(fasta)
        print(index.build(force=args.rebuild_fasta_index))
        return

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
    _validate_analysis_inputs(
        args.mode,
        gtf,
        fasta,
        atac,
        rna,
        genes=genes,
        gene_file=args.gene_file,
        atac_names=atac_names,
        rna_names=rna_names,
        metadata=args.metadata,
    )

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
            rna_region_mode=args.rna_region_mode,
            transcript_ids=transcript_ids,
        )
    elif args.mode == "paired":
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
            rna_region_mode=args.rna_region_mode,
            transcript_ids=transcript_ids,
        )


if __name__ == "__main__":
    main()
