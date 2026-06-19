"""BAM-level ATAC/RNA utilities.

This module starts from aligned BAM files. It keeps the project Python-first:
QC and count matrices are implemented with pysam, bigWig output uses pyBigWig,
and differential testing can optionally use PyDESeq2.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _require_pysam():
    try:
        import pysam
    except ImportError as exc:
        raise ImportError("需要安装 pysam: pip install pysam") from exc
    return pysam


def _require_pybigwig():
    try:
        import pyBigWig
    except ImportError as exc:
        raise ImportError("需要安装 pyBigWig: pip install pyBigWig") from exc
    return pyBigWig


def _sample_name(path):
    name = Path(path).name
    for suffix in (".bam", ".sam", ".cram"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _is_primary_usable(read, min_mapq=0, keep_duplicates=False):
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        return False
    if read.mapping_quality < min_mapq:
        return False
    if read.is_duplicate and not keep_duplicates:
        return False
    return True


def bam_qc(bam_file, output_dir=None, sample_name=None, min_mapq=30,
           mito_names=("chrM", "MT", "M")):
    """
    Basic BAM QC.

    Returns one-row DataFrame with mapping, duplicate, paired-end, MAPQ and
    mitochondrial-read summaries. Works without a BAM index by scanning records.
    """
    pysam = _require_pysam()
    sample_name = sample_name or _sample_name(bam_file)
    counts = {
        "sample": sample_name,
        "bam": str(bam_file),
        "total_records": 0,
        "mapped_records": 0,
        "unmapped_records": 0,
        "primary_mapped_records": 0,
        "secondary_records": 0,
        "supplementary_records": 0,
        "duplicate_records": 0,
        "paired_records": 0,
        "proper_pair_records": 0,
        "mapq_ge_min_records": 0,
        "mitochondrial_records": 0,
    }
    mapq_hist = {}

    with pysam.AlignmentFile(bam_file, "rb") as bam:
        mito_ids = {bam.get_tid(chrom) for chrom in mito_names if bam.get_tid(chrom) >= 0}
        for read in bam.fetch(until_eof=True):
            counts["total_records"] += 1
            mapq_hist[read.mapping_quality] = mapq_hist.get(read.mapping_quality, 0) + 1
            if read.is_unmapped:
                counts["unmapped_records"] += 1
                continue
            counts["mapped_records"] += 1
            if read.is_secondary:
                counts["secondary_records"] += 1
            if read.is_supplementary:
                counts["supplementary_records"] += 1
            if read.is_duplicate:
                counts["duplicate_records"] += 1
            if read.is_paired:
                counts["paired_records"] += 1
            if read.is_proper_pair:
                counts["proper_pair_records"] += 1
            if read.mapping_quality >= min_mapq:
                counts["mapq_ge_min_records"] += 1
            if read.reference_id in mito_ids:
                counts["mitochondrial_records"] += 1
            if not read.is_secondary and not read.is_supplementary:
                counts["primary_mapped_records"] += 1

    total = max(counts["total_records"], 1)
    mapped = max(counts["mapped_records"], 1)
    counts["mapped_rate"] = counts["mapped_records"] / total
    counts["duplicate_rate_mapped"] = counts["duplicate_records"] / mapped
    counts["proper_pair_rate_paired"] = (
        counts["proper_pair_records"] / max(counts["paired_records"], 1)
    )
    counts["mapq_ge_min_rate_mapped"] = counts["mapq_ge_min_records"] / mapped
    counts["mitochondrial_rate_mapped"] = counts["mitochondrial_records"] / mapped

    qc_df = pd.DataFrame([counts])
    hist_df = pd.DataFrame(
        [{"sample": sample_name, "mapq": k, "records": v} for k, v in sorted(mapq_hist.items())]
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        qc_df.to_csv(os.path.join(output_dir, f"{sample_name}_bam_qc.csv"),
                     index=False, encoding="utf-8-sig")
        hist_df.to_csv(os.path.join(output_dir, f"{sample_name}_mapq_hist.csv"),
                       index=False, encoding="utf-8-sig")
    return qc_df, hist_df


def fragment_length_distribution(bam_file, output_dir=None, sample_name=None,
                                 min_mapq=30, max_fragment=1000,
                                 keep_duplicates=False):
    """
    ATAC paired-end fragment-length distribution from template length.

    Only read1 of proper pairs is counted, so each DNA fragment contributes once.
    """
    pysam = _require_pysam()
    sample_name = sample_name or _sample_name(bam_file)
    lengths = []

    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if not _is_primary_usable(read, min_mapq=min_mapq,
                                      keep_duplicates=keep_duplicates):
                continue
            if not read.is_paired or not read.is_proper_pair or not read.is_read1:
                continue
            tlen = abs(int(read.template_length))
            if 0 < tlen <= max_fragment:
                lengths.append(tlen)

    if lengths:
        hist = np.bincount(np.asarray(lengths, dtype=np.int64), minlength=max_fragment + 1)
        hist_df = pd.DataFrame({
            "sample": sample_name,
            "fragment_length": np.arange(max_fragment + 1),
            "count": hist,
        })
        summary = {
            "sample": sample_name,
            "n_fragments": int(len(lengths)),
            "mean_fragment_length": float(np.mean(lengths)),
            "median_fragment_length": float(np.median(lengths)),
            "mode_fragment_length": int(np.argmax(hist)),
            "short_fragment_fraction_lt100": float(np.mean(np.asarray(lengths) < 100)),
            "mono_nucleosome_fraction_140_220": float(
                np.mean((np.asarray(lengths) >= 140) & (np.asarray(lengths) <= 220))
            ),
        }
    else:
        hist_df = pd.DataFrame(columns=["sample", "fragment_length", "count"])
        summary = {
            "sample": sample_name,
            "n_fragments": 0,
            "mean_fragment_length": np.nan,
            "median_fragment_length": np.nan,
            "mode_fragment_length": np.nan,
            "short_fragment_fraction_lt100": np.nan,
            "mono_nucleosome_fraction_140_220": np.nan,
        }

    summary_df = pd.DataFrame([summary])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        hist_df.to_csv(os.path.join(output_dir, f"{sample_name}_fragment_lengths.csv"),
                       index=False, encoding="utf-8-sig")
        summary_df.to_csv(os.path.join(output_dir, f"{sample_name}_fragment_summary.csv"),
                          index=False, encoding="utf-8-sig")
        plot_fragment_lengths(hist_df, os.path.join(output_dir, f"{sample_name}_fragment_lengths.png"))
    return summary_df, hist_df


def plot_fragment_lengths(hist_df, output_png):
    import matplotlib.pyplot as plt

    if hist_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_df["fragment_length"], hist_df["count"], linewidth=1.2)
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel("Fragments")
    ax.set_title("ATAC fragment length distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


def _load_bed_regions(bed_file):
    rows = []
    with open(bed_file, "rt", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            chrom, start, end = cols[0], int(cols[1]), int(cols[2])
            name = cols[3] if len(cols) >= 4 and cols[3] else f"{chrom}:{start}-{end}"
            rows.append({"region_id": name, "chrom": chrom, "start": start, "end": end})
    return pd.DataFrame(rows)


def count_reads_in_regions(bam_file, regions, sample_name=None, min_mapq=30,
                           count_fragments=True, keep_duplicates=False):
    """
    Count reads/fragments in BED-like regions.

    regions can be a BED path or a DataFrame with columns chrom/start/end and
    optional region_id.
    """
    pysam = _require_pysam()
    sample_name = sample_name or _sample_name(bam_file)
    if isinstance(regions, (str, os.PathLike)):
        regions_df = _load_bed_regions(regions)
    else:
        regions_df = regions.copy()
    if "region_id" not in regions_df.columns:
        regions_df["region_id"] = (
            regions_df["chrom"].astype(str) + ":" +
            regions_df["start"].astype(str) + "-" +
            regions_df["end"].astype(str)
        )

    counts = []
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for _, region in regions_df.iterrows():
            chrom, start, end = region["chrom"], int(region["start"]), int(region["end"])
            n = 0
            try:
                iterator = bam.fetch(chrom, start, end)
            except ValueError:
                iterator = []
            for read in iterator:
                if not _is_primary_usable(read, min_mapq=min_mapq,
                                          keep_duplicates=keep_duplicates):
                    continue
                if count_fragments and read.is_paired:
                    if not read.is_proper_pair or not read.is_read1:
                        continue
                n += 1
            counts.append({
                "region_id": region["region_id"],
                "chrom": chrom,
                "start": start,
                "end": end,
                sample_name: int(n),
            })
    return pd.DataFrame(counts)


def count_matrix_from_bams(bam_files, regions, sample_names=None, output_csv=None,
                           min_mapq=30, count_fragments=True,
                           keep_duplicates=False):
    sample_names = sample_names or [_sample_name(p) for p in bam_files]
    merged = None
    for bam_file, sample in zip(bam_files, sample_names):
        df = count_reads_in_regions(
            bam_file,
            regions,
            sample_name=sample,
            min_mapq=min_mapq,
            count_fragments=count_fragments,
            keep_duplicates=keep_duplicates,
        )
        key_cols = ["region_id", "chrom", "start", "end"]
        merged = df if merged is None else merged.merge(df, on=key_cols, how="outer")
    merged = merged.fillna(0)
    for sample in sample_names:
        if sample in merged.columns:
            merged[sample] = merged[sample].astype(int)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return merged


def bam_to_bigwig(bam_file, output_bw, bin_size=50, min_mapq=30,
                  normalize="CPM", count_fragments=True,
                  keep_duplicates=False):
    """
    Convert BAM to a binned bigWig coverage track.

    This is a lightweight Python implementation intended for visualization.
    For publication-grade coverage tracks, deepTools bamCoverage is still a
    stronger backend, but this keeps the package usable without R.
    """
    pysam = _require_pysam()
    pyBigWig = _require_pybigwig()
    bin_size = max(1, int(bin_size))

    usable_units = 0
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        chrom_sizes = list(zip(bam.references, bam.lengths))
        for read in bam.fetch(until_eof=True):
            if not _is_primary_usable(read, min_mapq=min_mapq,
                                      keep_duplicates=keep_duplicates):
                continue
            if count_fragments and read.is_paired:
                if not read.is_proper_pair or not read.is_read1:
                    continue
            usable_units += 1

    scale = 1.0
    if normalize.upper() == "CPM" and usable_units > 0:
        scale = 1_000_000.0 / usable_units

    with pysam.AlignmentFile(bam_file, "rb") as bam, pyBigWig.open(output_bw, "w") as bw:
        bw.addHeader(chrom_sizes)
        for chrom, chrom_len in chrom_sizes:
            n_bins = int(np.ceil(chrom_len / bin_size))
            cov = np.zeros(n_bins, dtype=np.float64)
            try:
                iterator = bam.fetch(chrom)
            except ValueError:
                iterator = []
            for read in iterator:
                if not _is_primary_usable(read, min_mapq=min_mapq,
                                          keep_duplicates=keep_duplicates):
                    continue
                if count_fragments and read.is_paired:
                    if not read.is_proper_pair or not read.is_read1:
                        continue
                    start = min(read.reference_start, read.next_reference_start)
                    end = start + abs(int(read.template_length))
                    if end <= start:
                        start, end = read.reference_start, read.reference_end
                else:
                    start, end = read.reference_start, read.reference_end
                start = max(0, int(start))
                end = min(chrom_len, int(end))
                if end <= start:
                    continue
                cov[start // bin_size:(end - 1) // bin_size + 1] += 1

            starts = np.arange(n_bins, dtype=np.int64) * bin_size
            ends = np.minimum(starts + bin_size, chrom_len)
            values = (cov * scale).astype(float)
            bw.addEntries(
                [chrom] * n_bins,
                starts.tolist(),
                ends=ends.tolist(),
                values=values.tolist(),
            )
    return output_bw


def pydeseq2_differential(count_matrix, metadata, condition_col="condition",
                          reference_level=None, output_csv=None):
    """
    Python-only differential test using PyDESeq2.

    count_matrix: DataFrame with region_id plus sample columns.
    metadata: DataFrame or CSV with columns sample and condition_col.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:
        raise ImportError("需要安装 PyDESeq2: pip install pydeseq2") from exc

    counts = count_matrix.copy()
    if "region_id" not in counts.columns:
        raise ValueError("count_matrix 必须包含 region_id 列")
    meta = pd.read_csv(metadata) if isinstance(metadata, (str, os.PathLike)) else metadata.copy()
    if "sample" not in meta.columns:
        raise ValueError("metadata 必须包含 sample 列")
    if condition_col not in meta.columns:
        raise ValueError(f"metadata 缺少列: {condition_col}")

    sample_cols = [s for s in meta["sample"].astype(str).tolist() if s in counts.columns]
    if len(sample_cols) < 2:
        raise ValueError("count_matrix 与 metadata 中匹配到的样本少于 2 个")

    count_df = counts.set_index("region_id")[sample_cols].T.astype(int)
    clinical = meta.set_index("sample").loc[sample_cols]

    design = f"~{condition_col}"
    try:
        dds = DeseqDataSet(counts=count_df, metadata=clinical, design=design)
    except TypeError:
        dds = DeseqDataSet(counts=count_df, clinical=clinical, design_factors=condition_col)
    dds.deseq2()

    if reference_level is not None:
        stats = DeseqStats(dds, contrast=[condition_col, reference_level[1], reference_level[0]])
    else:
        stats = DeseqStats(dds)
    stats.summary()
    result = stats.results_df.reset_index().rename(columns={"index": "region_id"})
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result


def run_bam_downstream(bam_files, regions_bed, output_dir,
                       sample_names=None, min_mapq=30,
                       make_bigwig=True, bigwig_bin_size=50):
    """
    Convenience wrapper: BAM QC + fragment length + count matrix + optional bigWig.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_names = sample_names or [_sample_name(p) for p in bam_files]
    qc_tables = []
    frag_tables = []
    bigwigs = []
    for bam_file, sample in zip(bam_files, sample_names):
        qc_df, _ = bam_qc(bam_file, output_dir=output_dir,
                          sample_name=sample, min_mapq=min_mapq)
        frag_df, _ = fragment_length_distribution(
            bam_file,
            output_dir=output_dir,
            sample_name=sample,
            min_mapq=min_mapq,
        )
        qc_tables.append(qc_df)
        frag_tables.append(frag_df)
        if make_bigwig:
            bw = os.path.join(output_dir, f"{sample}.bin{bigwig_bin_size}.cpm.bw")
            bigwigs.append(bam_to_bigwig(
                bam_file,
                bw,
                bin_size=bigwig_bin_size,
                min_mapq=min_mapq,
                normalize="CPM",
            ))

    qc_csv = os.path.join(output_dir, "bam_qc_summary.csv")
    frag_csv = os.path.join(output_dir, "fragment_summary.csv")
    count_csv = os.path.join(output_dir, "count_matrix.csv")
    pd.concat(qc_tables, ignore_index=True).to_csv(qc_csv, index=False, encoding="utf-8-sig")
    pd.concat(frag_tables, ignore_index=True).to_csv(frag_csv, index=False, encoding="utf-8-sig")
    count_df = count_matrix_from_bams(
        bam_files,
        regions_bed,
        sample_names=sample_names,
        output_csv=count_csv,
        min_mapq=min_mapq,
    )
    return {
        "qc_csv": qc_csv,
        "fragment_summary_csv": frag_csv,
        "count_matrix_csv": count_csv,
        "bigwig_files": bigwigs,
        "count_matrix": count_df,
    }
