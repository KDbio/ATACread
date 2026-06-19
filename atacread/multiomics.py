"""三个主模式: catalog / profile / paired"""
import os
import numpy as np
import pandas as pd

from .signal_utils import (
    to_array, summarize,
    detect_outliers, classify_gene_state, classify_direction,
    compute_global_thresholds, plot_gene_signals, plot_pca_2d,
    resolve_genes, pairwise_binned_permutation_tests,
)


# ============================================================
# 内部
# ============================================================

def _build_features(gtf_file, fasta_file, atac_files, rna_files,
                    genes=None, gene_file=None,
                    promoter_upstream=200, promoter_downstream=200,
                    atac_names=None, rna_names=None):
    from .read import fasta_read, ATACReader, RNAReader, assemble_gene_features
    
    gene_df = resolve_genes(gtf_file, genes, gene_file,
                             promoter_upstream, promoter_downstream)
    if gene_df.empty:
        raise ValueError("没有找到可分析的基因")
    
    target_chroms = sorted(set(gene_df["chrom"].dropna().astype(str)))
    print(f"[info] 加载 FASTA: {len(target_chroms)} 条目标染色体...")
    fasta_dict = fasta_read(fasta_file, keep_chroms=target_chroms)
    
    atac = ATACReader(atac_files, sample_names=atac_names) if atac_files else None
    rna = RNAReader(rna_files, sample_names=rna_names) if rna_files else None
    
    print(f"[info] 提取 {len(gene_df)} 个基因的特征...")
    features = assemble_gene_features(
        genes=gene_df, fasta_dict=fasta_dict,
        atac_reader=atac, rna_reader=rna,
        promoter_upstream=promoter_upstream,
        promoter_downstream=promoter_downstream,
        verbose=False,
    )
    return features, (atac.sample_names if atac else []), (rna.sample_names if rna else [])


def _collect_region_values(features, sample_names, prefix, region, metric="mean"):
    """收集每个基因每个样本的某个统计量。"""
    out = []
    for _, r in features.iterrows():
        row = {}
        for s in sample_names:
            key = f"{prefix}_{s}_{region}_signal"
            if key in r and r[key] is not None:
                row[s] = summarize(r[key])[metric]
        if row:
            out.append(row)
    return out


def _two_group_permutation_p(values_a, values_b, n_permutations=200, random_state=0):
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) == 0 or len(values_b) == 0:
        return np.nan
    observed = float(np.mean(values_b) - np.mean(values_a))
    pooled = np.concatenate([values_a, values_b])
    rng = np.random.default_rng(random_state)
    null = np.empty(int(n_permutations), dtype=np.float64)
    for i in range(int(n_permutations)):
        perm = rng.permutation(pooled)
        null[i] = np.mean(perm[len(values_a):]) - np.mean(perm[:len(values_a)])
    return float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))


# ============================================================
# 模式 1: catalog
# ============================================================

def run_catalog(gtf_file, fasta_file, atac_files=None, rna_files=None,
                output_dir="output_catalog",
                promoter_upstream=200, promoter_downstream=200,
                atac_names=None, rna_names=None,
                classify_state=True):
    """
    全基因特征表 + 可选基因状态分类。
    """
    os.makedirs(output_dir, exist_ok=True)
    features, atac_names, rna_names = _build_features(
        gtf_file, fasta_file, atac_files, rna_files,
        promoter_upstream=promoter_upstream,
        promoter_downstream=promoter_downstream,
        atac_names=atac_names, rna_names=rna_names,
    )
    
    base_cols = ["gene_id", "gene_name", "gene_type", "chrom", "strand",
                 "start", "end", "length", "tss", "promoter_start", "promoter_end"]
    
    rows = []
    for _, r in features.iterrows():
        row = {c: r.get(c) for c in base_cols}
        for s in atac_names:
            for region in ("promoter", "gene_body"):
                key = f"atac_{s}_{region}_signal"
                if key in r:
                    for k, v in summarize(r[key]).items():
                        row[f"atac_{s}_{region}_{k}"] = v
        for s in rna_names:
            key = f"rna_{s}_gene_body_signal"
            if key in r:
                for k, v in summarize(r[key]).items():
                    row[f"rna_{s}_gene_body_{k}"] = v
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 基因状态分类
    if classify_state and atac_names and rna_names:
        thresholds = compute_global_thresholds(df, atac_names, rna_names)
        states = []
        for _, r in df.iterrows():
            p = np.nanmean([r.get(f"atac_{s}_promoter_mean", np.nan) for s in atac_names])
            g = np.nanmean([r.get(f"atac_{s}_gene_body_mean", np.nan) for s in atac_names])
            rna = np.nanmean([r.get(f"rna_{s}_gene_body_mean", np.nan) for s in rna_names])
            states.append(classify_gene_state(p, g, rna, thresholds))
        df["gene_state"] = states
        
        threshold_csv = os.path.join(output_dir, "classification_thresholds.csv")
        pd.DataFrame([thresholds]).to_csv(threshold_csv, index=False, encoding="utf-8-sig")
    
    out_csv = os.path.join(output_dir, "gene_catalog.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[catalog] {out_csv}  ({len(df)} 基因)")
    return {"catalog_csv": out_csv}


# ============================================================
# 模式 2: profile
# ============================================================

def run_profile(gtf_file, fasta_file, atac_files=None, rna_files=None,
                genes=None, gene_file=None,
                output_dir="output_profile",
                promoter_upstream=200, promoter_downstream=200,
                atac_names=None, rna_names=None,
                outlier_z=2.5,
                bin_size="auto",
                n_permutations=200):
    """
    单基因 raw 画图 + 离群检测 + 分箱置换检验。
    """
    if not genes and not gene_file:
        raise ValueError("profile 模式必须指定 genes 或 gene_file")
    
    os.makedirs(output_dir, exist_ok=True)
    features, atac_names, rna_names = _build_features(
        gtf_file, fasta_file, atac_files, rna_files,
        genes=genes, gene_file=gene_file,
        promoter_upstream=promoter_upstream,
        promoter_downstream=promoter_downstream,
        atac_names=atac_names, rna_names=rna_names,
    )

    outlier_rows = []
    test_rows = []
    plots = []
    
    for _, r in features.iterrows():
        gname = r.get("gene_name", "")
        gid = r.get("gene_id", "")
        
        def collect(prefix, samples, region):
            return {s: to_array(r[f"{prefix}_{s}_{region}_signal"])
                    for s in samples if f"{prefix}_{s}_{region}_signal" in r}
        
        atac_p_raw = collect("atac", atac_names, "promoter") if atac_names else {}
        atac_g_raw = collect("atac", atac_names, "gene_body") if atac_names else {}
        rna_raw = collect("rna", rna_names, "gene_body") if rna_names else {}
        
        for label, signals in [
            ("ATAC_promoter", atac_p_raw),
            ("ATAC_genebody", atac_g_raw),
            ("RNA_genebody", rna_raw),
        ]:
            if len(signals) >= 3:
                odf = detect_outliers(signals, z_threshold=outlier_z)
                odf["gene_id"] = gid
                odf["gene_name"] = gname
                odf["region"] = label
                odf["scale"] = "raw"
                outlier_rows.append(odf)
            if len(signals) >= 2:
                tdf = pairwise_binned_permutation_tests(
                    signals,
                    bin_size=bin_size,
                    n_permutations=n_permutations,
                    region_name=label,
                )
                if not tdf.empty:
                    tdf["gene_id"] = gid
                    tdf["gene_name"] = gname
                    tdf["region"] = label
                    tdf["scale"] = "raw"
                    test_rows.append(tdf)
        
        # 绘图
        if atac_p_raw or rna_raw:
            png = os.path.join(output_dir, f"{gname}_signals.png")
            plot_gene_signals(
                gname,
                atac_p_raw,
                atac_g_raw,
                rna_raw,
                png,
                title_suffix=f"({r.get('chrom')}, {r.get('strand')})",
                promoter_seq=r.get("promoter_seq"),
            )
            plots.append(png)
    
    outlier_csv = os.path.join(output_dir, "outliers.csv")
    if outlier_rows:
        pd.concat(outlier_rows, ignore_index=True).to_csv(outlier_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(outlier_csv, index=False)
    
    test_csv = os.path.join(output_dir, "raw_permutation_tests.csv")
    if test_rows:
        pd.concat(test_rows, ignore_index=True).to_csv(test_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(test_csv, index=False, encoding="utf-8-sig")
    
    print(f"[profile] {len(plots)} 张图 | outliers: {outlier_csv} | tests: {test_csv}")
    return {"plots": plots, "outlier_csv": outlier_csv, "test_csv": test_csv}


# ============================================================
# 模式 3: paired
# ============================================================

def run_paired(gtf_file, fasta_file, atac_files, rna_files, metadata_file,
               genes=None, gene_file=None,
               output_dir="output_paired",
               promoter_upstream=200, promoter_downstream=200,
               atac_names=None, rna_names=None,
               lfc_threshold=0.5, make_pca=False,
               n_permutations=200):
    """
    配对样本下 ATAC promoter + ATAC gene_body + RNA 三维方向一致性分析。
    """
    if not genes and not gene_file:
        raise ValueError("paired 模式必须指定 genes 或 gene_file")
    
    os.makedirs(output_dir, exist_ok=True)
    meta = pd.read_csv(metadata_file)
    required = {"sample", "assay", "group"}
    if not required.issubset(meta.columns):
        raise ValueError(f"metadata 缺少列: {required - set(meta.columns)}")
    
    # 先做 profile 复用绘图
    profile_out = run_profile(
        gtf_file, fasta_file, atac_files, rna_files,
        genes=genes, gene_file=gene_file, output_dir=output_dir,
        promoter_upstream=promoter_upstream,
        promoter_downstream=promoter_downstream,
        atac_names=atac_names, rna_names=rna_names,
        n_permutations=n_permutations,
    )
    
    features, atac_names, rna_names = _build_features(
        gtf_file, fasta_file, atac_files, rna_files,
        genes=genes, gene_file=gene_file,
        promoter_upstream=promoter_upstream,
        promoter_downstream=promoter_downstream,
        atac_names=atac_names, rna_names=rna_names,
    )
    
    meta_a = meta[meta["assay"].str.upper() == "ATAC"].set_index("sample")
    meta_r = meta[meta["assay"].str.upper() == "RNA"].set_index("sample")
    
    def group_lfc(values_dict, meta_df, random_state=0):
        df = pd.DataFrame([{"sample": s, "value": v} for s, v in values_dict.items()])
        df = df.merge(meta_df.reset_index()[["sample", "group"]], on="sample")
        groups = sorted(df["group"].unique())
        if len(groups) != 2:
            return np.nan, np.nan, ""
        a = df[df["group"] == groups[0]]["value"].to_numpy()
        b = df[df["group"] == groups[1]]["value"].to_numpy()
        lfc = float(np.log2((b.mean() + 1e-6) / (a.mean() + 1e-6)))
        p = _two_group_permutation_p(
            a,
            b,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        return lfc, p, f"{groups[1]}_vs_{groups[0]}"
    
    direction_rows = []
    rna_matrix = []
    
    for _, r in features.iterrows():
        gname, gid = r.get("gene_name"), r.get("gene_id")
        
        # 三个指标
        p_atac = {s: float(np.mean(to_array(r.get(f"atac_{s}_promoter_signal", []))))
                  for s in atac_names if f"atac_{s}_promoter_signal" in r}
        g_atac = {s: float(np.mean(to_array(r.get(f"atac_{s}_gene_body_signal", []))))
                  for s in atac_names if f"atac_{s}_gene_body_signal" in r}
        rna = {s: float(np.mean(to_array(r.get(f"rna_{s}_gene_body_signal", []))))
               for s in rna_names if f"rna_{s}_gene_body_signal" in r}
        
        rna_matrix.append({"gene_id": gid, "gene_name": gname, **rna})
        
        p_lfc, p_p, contrast = group_lfc(p_atac, meta_a, random_state=1)
        g_lfc, g_p, _ = group_lfc(g_atac, meta_a, random_state=2)
        r_lfc, r_p, _ = group_lfc(rna, meta_r, random_state=3)
        
        direction = classify_direction(p_lfc, g_lfc, r_lfc, lfc_threshold=lfc_threshold)
        
        direction_rows.append({
            "gene_id": gid, "gene_name": gname, "contrast": contrast,
            "promoter_ATAC_log2fc": p_lfc, "promoter_ATAC_p": p_p,
            "genebody_ATAC_log2fc": g_lfc, "genebody_ATAC_p": g_p,
            "RNA_log2fc": r_lfc, "RNA_p": r_p,
            "direction": direction,
        })
    
    direction_csv = os.path.join(output_dir, "direction_analysis.csv")
    pd.DataFrame(direction_rows).to_csv(direction_csv, index=False, encoding="utf-8-sig")
    
    matrix_df = pd.DataFrame(rna_matrix)
    matrix_csv = os.path.join(output_dir, "rna_expression_matrix.csv")
    matrix_df.to_csv(matrix_csv, index=False, encoding="utf-8-sig")
    
    out = {**profile_out, "direction_csv": direction_csv, "rna_matrix_csv": matrix_csv}
    
    if make_pca and len(matrix_df) >= 2:
        pca_png = os.path.join(output_dir, "rna_pca.png")
        score_df, _ = plot_pca_2d(matrix_df, pca_png, title="RNA PCA")
        score_df.to_csv(os.path.join(output_dir, "rna_pca_scores.csv"), index=False, encoding="utf-8-sig")
        out["pca_png"] = pca_png
    
    print(f"[paired] direction: {direction_csv}")
    return out


# ============================================================
# 兼容 demo 的高层接口
# ============================================================

def extract_multiomics_features(gtf_file, fasta_file, atac_files=None, rna_files=None,
                                atac_sample_names=None, rna_sample_names=None,
                                genes=None, gene_file=None,
                                promoter_upstream=200, promoter_downstream=200,
                                output_dir="demo_multiomics_output",
                                save_per_base=False,
                                make_plots=True,
                                plot_first_n=10,
                                outlier_z=2.5,
                                bin_size="auto",
                                n_permutations=200):
    """
    兼容旧 demo 的一站式接口。

    重要说明
    ----
    该接口只保留 raw 信号统计和分箱置换检验，不再做基线对齐或归一化。
    """
    if not genes and not gene_file:
        raise ValueError("extract_multiomics_features 必须指定 genes 或 gene_file")

    os.makedirs(output_dir, exist_ok=True)

    features, atac_names, rna_names = _build_features(
        gtf_file=gtf_file,
        fasta_file=fasta_file,
        atac_files=atac_files,
        rna_files=rna_files,
        genes=genes,
        gene_file=gene_file,
        promoter_upstream=promoter_upstream,
        promoter_downstream=promoter_downstream,
        atac_names=atac_sample_names,
        rna_names=rna_sample_names,
    )

    summary_rows = []
    outlier_rows = []
    test_rows = []
    per_base_rows = []
    plot_files = []

    for i, (_, r) in enumerate(features.iterrows()):
        gene_name = r.get("gene_name", "")
        gene_id = r.get("gene_id", "")

        row = {
            "gene_id": gene_id,
            "gene_name": gene_name,
            "gene_type": r.get("gene_type", ""),
            "chrom": r.get("chrom", ""),
            "strand": r.get("strand", ""),
            "start": r.get("start", ""),
            "end": r.get("end", ""),
            "length": r.get("length", ""),
            "tss": r.get("tss", ""),
            "promoter_start": r.get("promoter_start", ""),
            "promoter_end": r.get("promoter_end", ""),
        }

        def collect(prefix, samples, region):
            raw = {}
            for sample in samples:
                key = f"{prefix}_{sample}_{region}_signal"
                if key not in r or r[key] is None:
                    continue
                raw_signal = to_array(r[key])
                raw[sample] = raw_signal

                for stat_name, stat_value in summarize(raw_signal).items():
                    row[f"{prefix}_{sample}_{region}_raw_{stat_name}"] = stat_value

                if save_per_base:
                    for pos, value in enumerate(raw_signal):
                        per_base_rows.append({
                            "gene_id": gene_id,
                            "gene_name": gene_name,
                            "assay": prefix,
                            "sample": sample,
                            "region": region,
                            "position": pos,
                            "raw_value": float(value),
                        })
            return raw

        atac_p_raw = collect("atac", atac_names, "promoter") if atac_names else {}
        atac_g_raw = collect("atac", atac_names, "gene_body") if atac_names else {}
        rna_raw = collect("rna", rna_names, "gene_body") if rna_names else {}

        for label, signals in [
            ("ATAC_promoter", atac_p_raw),
            ("ATAC_genebody", atac_g_raw),
            ("RNA_genebody", rna_raw),
        ]:
            if len(signals) >= 3:
                odf = detect_outliers(signals, z_threshold=outlier_z)
                odf["gene_id"] = gene_id
                odf["gene_name"] = gene_name
                odf["region"] = label
                odf["scale"] = "raw"
                outlier_rows.append(odf)
            if len(signals) >= 2:
                tdf = pairwise_binned_permutation_tests(
                    signals,
                    bin_size=bin_size,
                    n_permutations=n_permutations,
                    region_name=label,
                )
                if not tdf.empty:
                    tdf["gene_id"] = gene_id
                    tdf["gene_name"] = gene_name
                    tdf["region"] = label
                    tdf["scale"] = "raw"
                    test_rows.append(tdf)

        if make_plots and i < plot_first_n:
            png = os.path.join(output_dir, f"{gene_name}_signals.png")
            plot_gene_signals(
                gene_name,
                atac_p_raw,
                atac_g_raw,
                rna_raw,
                png,
                title_suffix=f"({r.get('chrom')}, {r.get('strand')})",
                promoter_seq=r.get("promoter_seq"),
            )
            plot_files.append(png)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "multiomics_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    outlier_csv = os.path.join(output_dir, "outliers.csv")
    if outlier_rows:
        pd.concat(outlier_rows, ignore_index=True).to_csv(outlier_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(outlier_csv, index=False, encoding="utf-8-sig")

    test_csv = os.path.join(output_dir, "raw_permutation_tests.csv")
    if test_rows:
        pd.concat(test_rows, ignore_index=True).to_csv(test_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(test_csv, index=False, encoding="utf-8-sig")

    per_base_csv = None
    if save_per_base:
        per_base_csv = os.path.join(output_dir, "per_base_values.csv")
        pd.DataFrame(per_base_rows).to_csv(per_base_csv, index=False, encoding="utf-8-sig")

    return {
        "n_genes": len(summary_df),
        "summary_csv": summary_csv,
        "outlier_csv": outlier_csv,
        "test_csv": test_csv,
        "per_base_csv": per_base_csv,
        "plot_files": plot_files,
        "features": features,
    }
