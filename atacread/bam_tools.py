"""BAM-level ATAC/RNA utilities.

This module starts from aligned BAM files. It keeps the project Python-first:
QC and count matrices are implemented with pysam, bigWig output uses pyBigWig,
and differential testing can optionally use PyDESeq2.
"""
import bisect
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


def _validate_bam_inputs(bam_files, sample_names=None):
    bam_files = [str(p) for p in (bam_files or [])]
    if not bam_files:
        raise ValueError("没有提供 BAM 文件")
    missing = [p for p in bam_files if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"BAM 文件不存在: {missing}")

    if sample_names is None:
        sample_names = [_sample_name(p) for p in bam_files]
    else:
        sample_names = [str(s) for s in sample_names]
        if len(sample_names) != len(bam_files):
            raise ValueError(
                f"sample_names 数量 ({len(sample_names)}) 与 BAM 数量 "
                f"({len(bam_files)}) 不一致"
            )
    if len(set(sample_names)) != len(sample_names):
        raise ValueError("sample_names 不能重复")
    return bam_files, sample_names


def ensure_bam_index(bam_file, auto_index=True):
    """Require a coordinate-sorted BAM index, creating it when possible."""
    pysam = _require_pysam()
    bam_file = str(bam_file)
    if not os.path.isfile(bam_file):
        raise FileNotFoundError(f"BAM 文件不存在: {bam_file}")

    with pysam.AlignmentFile(bam_file, "rb") as bam:
        try:
            if bam.has_index():
                return bam_file
        except (ValueError, OSError):
            pass
        sort_order = bam.header.to_dict().get("HD", {}).get("SO", "unknown")

    if not auto_index:
        raise ValueError(
            f"BAM 缺少索引: {bam_file}。请运行 samtools index，"
            "或不要使用 --no-auto-index。"
        )
    try:
        pysam.index(bam_file)
    except Exception as exc:
        raise ValueError(
            f"无法为 BAM 建立索引 (SO={sort_order}): {bam_file}。"
            "文件可能没有按坐标排序，请先使用 samtools sort 或 pysam.sort。"
        ) from exc
    return bam_file


def _fragment_span(read):
    """Return one genomic span per paired fragment or single-end read."""
    if read.is_paired:
        if (
            not read.is_proper_pair
            or not read.is_read1
            or read.next_reference_id != read.reference_id
        ):
            return None
        start = min(int(read.reference_start), int(read.next_reference_start))
        end = start + abs(int(read.template_length))
        if end <= start:
            return None
        return start, end
    if read.reference_start is None or read.reference_end is None:
        return None
    return int(read.reference_start), int(read.reference_end)


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
        "primary_nonduplicate_mapped_records": 0,
        "usable_units": 0,
        "nonduplicate_usable_units": 0,
        "duplicate_usable_units": 0,
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
                if not read.is_duplicate:
                    counts["primary_nonduplicate_mapped_records"] += 1

                is_unit = (
                    (read.is_paired and read.is_proper_pair and read.is_read1)
                    or not read.is_paired
                )
                if is_unit:
                    counts["usable_units"] += 1
                    if read.is_duplicate:
                        counts["duplicate_usable_units"] += 1
                    else:
                        counts["nonduplicate_usable_units"] += 1

    total = max(counts["total_records"], 1)
    mapped = max(counts["mapped_records"], 1)
    counts["mapped_rate"] = counts["mapped_records"] / total
    counts["duplicate_rate_mapped"] = counts["duplicate_records"] / mapped
    counts["proper_pair_rate_paired"] = (
        counts["proper_pair_records"] / max(counts["paired_records"], 1)
    )
    counts["mapq_ge_min_rate_mapped"] = counts["mapq_ge_min_records"] / mapped
    counts["mitochondrial_rate_mapped"] = counts["mitochondrial_records"] / mapped
    counts["nonduplicate_rate_primary"] = (
        counts["primary_nonduplicate_mapped_records"]
        / max(counts["primary_mapped_records"], 1)
    )
    counts["nonduplicate_unit_fraction"] = (
        counts["nonduplicate_usable_units"] / max(counts["usable_units"], 1)
    )

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


def _merge_intervals(regions_df):
    merged = {}
    for chrom, group in regions_df.groupby("chrom", sort=False):
        intervals = sorted(
            (max(0, int(r.start)), int(r.end))
            for r in group.itertuples()
            if int(r.end) > int(r.start)
        )
        chrom_intervals = []
        for start, end in intervals:
            if chrom_intervals and start <= chrom_intervals[-1][1]:
                chrom_intervals[-1][1] = max(chrom_intervals[-1][1], end)
            else:
                chrom_intervals.append([start, end])
        merged[str(chrom)] = {
            "starts": [x[0] for x in chrom_intervals],
            "ends": [x[1] for x in chrom_intervals],
        }
    return merged


def _overlaps_merged(merged, chrom, start, end):
    data = merged.get(str(chrom))
    if not data or end <= start:
        return False
    i = bisect.bisect_right(data["starts"], end - 1) - 1
    return i >= 0 and data["ends"][i] > start


def frip_score(bam_file, peak_regions, sample_name=None, min_mapq=30,
               keep_duplicates=False):
    """Calculate the fraction of usable fragments/reads overlapping peaks."""
    pysam = _require_pysam()
    sample_name = sample_name or _sample_name(bam_file)
    regions_df = (
        _load_bed_regions(peak_regions)
        if isinstance(peak_regions, (str, os.PathLike))
        else peak_regions.copy()
    )
    if regions_df.empty:
        raise ValueError("peak_regions 中没有有效区域，无法计算 FRiP")
    merged = _merge_intervals(regions_df)

    usable_units = 0
    in_peak_units = 0
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        references = bam.references
        reference_set = set(references)
        normalized = {}
        for chrom, data in merged.items():
            resolved = _chrom_alias(chrom, reference_set)
            if resolved is not None:
                normalized[resolved] = data
        if not normalized:
            raise ValueError("peak_regions 与 BAM 没有匹配的染色体")
        for read in bam.fetch(until_eof=True):
            if not _is_primary_usable(
                read, min_mapq=min_mapq, keep_duplicates=keep_duplicates
            ):
                continue
            span = _fragment_span(read)
            if span is None:
                continue
            usable_units += 1
            chrom = references[read.reference_id]
            if _overlaps_merged(normalized, chrom, span[0], span[1]):
                in_peak_units += 1

    return pd.DataFrame([{
        "sample": sample_name,
        "usable_units": usable_units,
        "in_peak_units": in_peak_units,
        "frip": in_peak_units / usable_units if usable_units else np.nan,
        "peak_regions": int(len(regions_df)),
    }])


def _load_tss_sites(source):
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = str(source)
        if path.lower().endswith((".gtf", ".gtf.gz")):
            import gzip

            opener = gzip.open if path.lower().endswith(".gz") else open
            rows = []
            with opener(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line or line.startswith("#"):
                        continue
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) < 9 or cols[2] != "gene":
                        continue
                    start, end, strand = int(cols[3]), int(cols[4]), cols[6]
                    tss = start - 1 if strand != "-" else end - 1
                    rows.append({"chrom": cols[0], "tss": tss, "strand": strand})
            df = pd.DataFrame(rows)
        else:
            rows = []
            with open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip() or line.startswith("#"):
                        continue
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) < 3:
                        continue
                    start, end = int(cols[1]), int(cols[2])
                    strand = cols[5] if len(cols) >= 6 and cols[5] in ("+", "-") else "+"
                    tss = start if end - start <= 1 else (end - 1 if strand == "-" else start)
                    rows.append({"chrom": cols[0], "tss": tss, "strand": strand})
            df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=["chrom", "tss", "strand"])
    if "tss" not in df.columns:
        if not {"start", "end"}.issubset(df.columns):
            raise ValueError("TSS 表必须包含 tss，或 start/end 列")
        strand = df.get("strand", pd.Series("+", index=df.index)).astype(str)
        df["tss"] = np.where(strand.eq("-"), df["end"].astype(int) - 1,
                             df["start"].astype(int))
    if "strand" not in df.columns:
        df["strand"] = "+"
    return df[["chrom", "tss", "strand"]].dropna().assign(
        chrom=lambda x: x["chrom"].astype(str),
        tss=lambda x: x["tss"].astype(int),
        strand=lambda x: x["strand"].astype(str),
    )


def _chrom_alias(chrom, references):
    if chrom in references:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in references:
        return chrom[3:]
    candidate = f"chr{chrom}"
    return candidate if candidate in references else None


def tss_enrichment(bam_file, tss_regions, output_dir=None, sample_name=None,
                   min_mapq=30, flank=2000, center_window=100,
                   edge_window=100, keep_duplicates=False,
                   apply_atac_shift=True):
    """Aggregate ATAC insertion sites around TSS and report enrichment."""
    pysam = _require_pysam()
    sample_name = sample_name or _sample_name(bam_file)
    flank = max(1, int(flank))
    center_window = min(flank, max(0, int(center_window)))
    edge_window = min(flank, max(1, int(edge_window)))
    sites = _load_tss_sites(tss_regions)
    if sites.empty:
        raise ValueError("没有读到有效 TSS")

    profile = np.zeros(2 * flank + 1, dtype=np.float64)
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        references = set(bam.references)
        by_chrom = {}
        for chrom, group in sites.groupby("chrom", sort=False):
            resolved = _chrom_alias(str(chrom), references)
            if resolved is None:
                continue
            entries = sorted(
                (int(r.tss), str(r.strand)) for r in group.itertuples()
            )
            by_chrom.setdefault(resolved, []).extend(entries)
        for chrom in by_chrom:
            by_chrom[chrom].sort()

        positions = {c: [x[0] for x in entries] for c, entries in by_chrom.items()}
        strands = {c: [x[1] for x in entries] for c, entries in by_chrom.items()}
        matched_tss = sum(len(v) for v in positions.values())

        for read in bam.fetch(until_eof=True):
            if not _is_primary_usable(
                read, min_mapq=min_mapq, keep_duplicates=keep_duplicates
            ):
                continue
            chrom = bam.get_reference_name(read.reference_id)
            chrom_positions = positions.get(chrom)
            if not chrom_positions:
                continue
            if read.is_reverse:
                insertion = int(read.reference_end) - (5 if apply_atac_shift else 1)
            else:
                insertion = int(read.reference_start) + (4 if apply_atac_shift else 0)

            left = bisect.bisect_left(chrom_positions, insertion - flank)
            right = bisect.bisect_right(chrom_positions, insertion + flank)
            for i in range(left, right):
                relative = insertion - chrom_positions[i]
                if strands[chrom][i] == "-":
                    relative = -relative
                profile[relative + flank] += 1

    per_tss = profile / max(matched_tss, 1)
    edge_values = np.concatenate([per_tss[:edge_window], per_tss[-edge_window:]])
    background = float(np.mean(edge_values)) if edge_values.size else np.nan
    center = per_tss[flank - center_window:flank + center_window + 1]
    score = float(np.max(center) / background) if background > 0 else np.nan

    summary_df = pd.DataFrame([{
        "sample": sample_name,
        "tss_sites_input": int(len(sites)),
        "tss_sites_matched": int(matched_tss),
        "flank": flank,
        "background": background,
        "center_max": float(np.max(center)) if center.size else np.nan,
        "tss_enrichment": score,
        "atac_shift_applied": bool(apply_atac_shift),
    }])
    profile_df = pd.DataFrame({
        "sample": sample_name,
        "position": np.arange(-flank, flank + 1),
        "insertions_per_tss": per_tss,
        "normalized_signal": per_tss / background if background > 0 else np.nan,
    })

    if output_dir:
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        profile_csv = os.path.join(output_dir, f"{sample_name}_tss_profile.csv")
        profile_png = os.path.join(output_dir, f"{sample_name}_tss_enrichment.png")
        profile_df.to_csv(profile_csv, index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(profile_df["position"], profile_df["normalized_signal"], linewidth=1.2)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Position relative to TSS (bp)")
        ax.set_ylabel("Normalized insertion signal")
        ax.set_title(f"{sample_name} TSS enrichment")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(profile_png, dpi=150)
        plt.close(fig)
    return summary_df, profile_df


def count_reads_in_regions(bam_file, regions, sample_name=None, min_mapq=30,
                           count_fragments=True, keep_duplicates=False,
                           auto_index=True, max_fragment=2000):
    """
    Count reads/fragments in BED-like regions.

    regions can be a BED path or a DataFrame with columns chrom/start/end and
    optional region_id.
    """
    pysam = _require_pysam()
    ensure_bam_index(bam_file, auto_index=auto_index)
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
            if chrom not in bam.references:
                alias = _chrom_alias(str(chrom), set(bam.references))
                if alias is None:
                    raise ValueError(f"区域染色体 {chrom} 不在 BAM 中")
                chrom = alias
            n = 0
            seen_fragments = set()
            fetch_start = max(0, start - max_fragment) if count_fragments else start
            fetch_end = end + max_fragment if count_fragments else end
            iterator = bam.fetch(chrom, fetch_start, fetch_end)
            for read in iterator:
                if not _is_primary_usable(read, min_mapq=min_mapq,
                                          keep_duplicates=keep_duplicates):
                    continue
                if count_fragments and read.is_paired:
                    if (
                        not read.is_proper_pair
                        or read.next_reference_id != read.reference_id
                        or read.query_name in seen_fragments
                    ):
                        continue
                    fragment_start = min(read.reference_start, read.next_reference_start)
                    fragment_end = fragment_start + abs(int(read.template_length))
                    if fragment_end <= fragment_start:
                        continue
                    seen_fragments.add(read.query_name)
                    if fragment_start < end and fragment_end > start:
                        n += 1
                elif read.reference_start < end and read.reference_end > start:
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
                           keep_duplicates=False, auto_index=True):
    bam_files, sample_names = _validate_bam_inputs(bam_files, sample_names)
    merged = None
    for bam_file, sample in zip(bam_files, sample_names):
        df = count_reads_in_regions(
            bam_file,
            regions,
            sample_name=sample,
            min_mapq=min_mapq,
            count_fragments=count_fragments,
            keep_duplicates=keep_duplicates,
            auto_index=auto_index,
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
                  keep_duplicates=False, auto_index=True):
    """
    Convert BAM to a binned bigWig coverage track.

    This is a lightweight Python implementation intended for visualization.
    For publication-grade coverage tracks, deepTools bamCoverage is still a
    stronger backend, but this keeps the package usable without R.
    """
    pysam = _require_pysam()
    pyBigWig = _require_pybigwig()
    ensure_bam_index(bam_file, auto_index=auto_index)
    bin_size = max(1, int(bin_size))
    os.makedirs(os.path.dirname(str(output_bw)) or ".", exist_ok=True)

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
            covered_bases = np.zeros(n_bins, dtype=np.float64)
            iterator = bam.fetch(chrom)
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
                first_bin = start // bin_size
                last_bin = (end - 1) // bin_size
                if first_bin == last_bin:
                    covered_bases[first_bin] += end - start
                else:
                    covered_bases[first_bin] += (first_bin + 1) * bin_size - start
                    covered_bases[last_bin] += end - last_bin * bin_size
                    if last_bin > first_bin + 1:
                        covered_bases[first_bin + 1:last_bin] += bin_size

            starts = np.arange(n_bins, dtype=np.int64) * bin_size
            ends = np.minimum(starts + bin_size, chrom_len)
            widths = np.maximum(ends - starts, 1)
            values = (covered_bases / widths * scale).astype(float)
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

    counts = (
        pd.read_csv(count_matrix)
        if isinstance(count_matrix, (str, os.PathLike))
        else count_matrix.copy()
    )
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

    numeric = counts.set_index("region_id")[sample_cols].apply(pd.to_numeric, errors="raise")
    if (numeric < 0).any().any():
        raise ValueError("count_matrix 不能包含负数")
    if not np.allclose(numeric.to_numpy(), np.rint(numeric.to_numpy())):
        raise ValueError("PyDESeq2 需要原始整数 count，不能使用 bigWig 连续信号")
    count_df = numeric.round().astype(int).T
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


def run_bam_downstream(bam_files, regions_bed=None, output_dir="output_bam",
                       sample_names=None, min_mapq=30,
                       make_bigwig=True, bigwig_bin_size=50,
                       tss_regions=None, auto_index=True,
                       keep_duplicates=False):
    """
    BAM QC + fragment length + optional FRiP/TSS/count matrix/bigWig.
    """
    os.makedirs(output_dir, exist_ok=True)
    bam_files, sample_names = _validate_bam_inputs(bam_files, sample_names)
    qc_tables = []
    frag_tables = []
    frip_tables = []
    tss_tables = []
    bigwigs = []
    for bam_file, sample in zip(bam_files, sample_names):
        qc_df, _ = bam_qc(bam_file, output_dir=output_dir,
                          sample_name=sample, min_mapq=min_mapq)
        frag_df, _ = fragment_length_distribution(
            bam_file,
            output_dir=output_dir,
            sample_name=sample,
            min_mapq=min_mapq,
            keep_duplicates=keep_duplicates,
        )
        qc_tables.append(qc_df)
        frag_tables.append(frag_df)
        if regions_bed:
            frip_tables.append(frip_score(
                bam_file,
                regions_bed,
                sample_name=sample,
                min_mapq=min_mapq,
                keep_duplicates=keep_duplicates,
            ))
        if tss_regions:
            tss_df, _ = tss_enrichment(
                bam_file,
                tss_regions,
                output_dir=output_dir,
                sample_name=sample,
                min_mapq=min_mapq,
                keep_duplicates=keep_duplicates,
            )
            tss_tables.append(tss_df)
        if make_bigwig:
            bw = os.path.join(output_dir, f"{sample}.bin{bigwig_bin_size}.cpm.bw")
            bigwigs.append(bam_to_bigwig(
                bam_file,
                bw,
                bin_size=bigwig_bin_size,
                min_mapq=min_mapq,
                normalize="CPM",
                keep_duplicates=keep_duplicates,
                auto_index=auto_index,
            ))

    qc_csv = os.path.join(output_dir, "bam_qc_summary.csv")
    frag_csv = os.path.join(output_dir, "fragment_summary.csv")
    frip_csv = os.path.join(output_dir, "frip_summary.csv") if frip_tables else None
    tss_csv = os.path.join(output_dir, "tss_enrichment_summary.csv") if tss_tables else None
    count_csv = os.path.join(output_dir, "count_matrix.csv") if regions_bed else None
    pd.concat(qc_tables, ignore_index=True).to_csv(qc_csv, index=False, encoding="utf-8-sig")
    pd.concat(frag_tables, ignore_index=True).to_csv(frag_csv, index=False, encoding="utf-8-sig")
    if frip_tables:
        pd.concat(frip_tables, ignore_index=True).to_csv(
            frip_csv, index=False, encoding="utf-8-sig"
        )
    if tss_tables:
        pd.concat(tss_tables, ignore_index=True).to_csv(
            tss_csv, index=False, encoding="utf-8-sig"
        )
    count_df = None
    if regions_bed:
        count_df = count_matrix_from_bams(
            bam_files,
            regions_bed,
            sample_names=sample_names,
            output_csv=count_csv,
            min_mapq=min_mapq,
            keep_duplicates=keep_duplicates,
            auto_index=auto_index,
        )
    return {
        "qc_csv": qc_csv,
        "fragment_summary_csv": frag_csv,
        "frip_summary_csv": frip_csv,
        "tss_enrichment_summary_csv": tss_csv,
        "count_matrix_csv": count_csv,
        "bigwig_files": bigwigs,
        "count_matrix": count_df,
    }
