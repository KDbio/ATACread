"""信号处理通用工具。"""
import os
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 基础
# ============================================================

def to_array(signal):
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def robust_z(values):
    values = np.asarray(values, dtype=np.float64)
    center = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - center))
    if mad == 0 or np.isnan(mad):
        sd = np.nanstd(values)
        return np.zeros_like(values) if sd == 0 else (values - center) / sd
    return 0.6745 * (values - center) / mad


def summarize(signal):
    arr = to_array(signal)
    if len(arr) == 0:
        return dict.fromkeys(["n_bp", "mean", "median", "peak", "peak_pos", "auc"], np.nan)
    return {
        "n_bp": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "peak": float(np.max(arr)),
        "peak_pos": int(np.argmax(arr)),
        "auc": float(np.sum(arr)),
    }


def auto_bin_size(signal_length, region_name=None,
                  promoter_target_bins=120, genebody_target_bins=150,
                  min_bin_size=20, max_bin_size=150):
    """
    根据区域长度自动选择 bin 大小。

    promoter 保留更高分辨率；gene body 更长，目标是控制 bin 数量，
    避免长区域因为切出过多 bin 而使检验过于敏感。
    """
    signal_length = max(1, int(signal_length))
    region = (region_name or "").lower()
    target_bins = promoter_target_bins if "promoter" in region else genebody_target_bins
    raw = int(np.ceil(signal_length / max(1, target_bins)))
    return int(np.clip(raw, min_bin_size, max_bin_size))


def bin_signal(signal, bin_size=20, agg="mean"):
    """
    把连续碱基信号压成较粗的 bin，避免把相邻碱基当作独立观测。
    """
    arr = to_array(signal)
    if len(arr) == 0:
        return np.array([], dtype=np.float64)
    if isinstance(bin_size, str) and bin_size.lower() == "auto":
        bin_size = auto_bin_size(len(arr))
    bin_size = max(1, int(bin_size))
    bins = []
    for start in range(0, len(arr), bin_size):
        chunk = arr[start:start + bin_size]
        if agg == "sum":
            bins.append(float(np.sum(chunk)))
        else:
            bins.append(float(np.mean(chunk)))
    return np.asarray(bins, dtype=np.float64)


def binned_permutation_test(signal_a, signal_b, bin_size="auto",
                            n_permutations=200, agg="mean",
                            random_state=0, region_name=None):
    """
    比较两条 raw accessibility 曲线的观察差异。

    统计单位是 bin，不是单个碱基；p 值来自标签随机置换。
    这个检验回答的是“观察信号是否不同”，不区分批次差异和生物学差异。
    """
    arr_a = to_array(signal_a)
    arr_b = to_array(signal_b)
    resolved_bin_size = (
        auto_bin_size(max(len(arr_a), len(arr_b)), region_name=region_name)
        if isinstance(bin_size, str) and bin_size.lower() == "auto"
        else max(1, int(bin_size))
    )
    a = bin_signal(arr_a, bin_size=resolved_bin_size, agg=agg)
    b = bin_signal(arr_b, bin_size=resolved_bin_size, agg=agg)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {
            "test": "binned_permutation",
            "bin_size": int(resolved_bin_size),
            "bin_size_mode": "auto" if isinstance(bin_size, str) else "fixed",
            "n_permutations": int(n_permutations),
            "n_bins_a": int(len(a)),
            "n_bins_b": int(len(b)),
            "mean_a": np.nan,
            "mean_b": np.nan,
            "mean_diff": np.nan,
            "auc_a": np.nan,
            "auc_b": np.nan,
            "auc_ratio": np.nan,
            "log2_auc_ratio": np.nan,
            "effect_size": np.nan,
            "p_value": np.nan,
            "direction": "unknown",
            "interpretation": "raw_observed_difference",
        }

    eps = 1e-9
    observed = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(random_state)
    null = np.empty(int(n_permutations), dtype=np.float64)
    for i in range(int(n_permutations)):
        perm = rng.permutation(pooled)
        null[i] = np.mean(perm[:n_a]) - np.mean(perm[n_a:])
    p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))

    auc_a = float(np.sum(arr_a))
    auc_b = float(np.sum(arr_b))
    sd = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
    effect = float(observed / sd) if sd > 0 else np.nan
    log2_ratio = float(np.log2((auc_a + eps) / (auc_b + eps)))
    direction = "higher_a" if observed > 0 else "higher_b" if observed < 0 else "flat"

    return {
        "test": "binned_permutation",
        "bin_size": int(resolved_bin_size),
        "bin_size_mode": "auto" if isinstance(bin_size, str) else "fixed",
        "n_permutations": int(n_permutations),
        "n_bins_a": int(len(a)),
        "n_bins_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": observed,
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_ratio": float((auc_a + eps) / (auc_b + eps)),
        "log2_auc_ratio": log2_ratio,
        "effect_size": effect,
        "p_value": p_value,
        "direction": direction,
        "interpretation": "raw_observed_difference",
    }


def pairwise_binned_permutation_tests(signals_by_sample, bin_size="auto",
                                      n_permutations=200, random_state=0,
                                      region_name=None):
    rows = []
    for i, (sample_a, sample_b) in enumerate(combinations(signals_by_sample.keys(), 2)):
        result = binned_permutation_test(
            signals_by_sample[sample_a],
            signals_by_sample[sample_b],
            bin_size=bin_size,
            n_permutations=n_permutations,
            random_state=random_state + i,
            region_name=region_name,
        )
        result["sample_a"] = sample_a
        result["sample_b"] = sample_b
        rows.append(result)
    return pd.DataFrame(rows)


# ============================================================
# 离群检测
# ============================================================

def detect_outliers(signals_by_sample, z_threshold=2.5):
    rows = []
    for s, sig in signals_by_sample.items():
        rows.append({"sample": s, **summarize(sig)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["auc_z"] = robust_z(df["auc"].to_numpy())
    df["peak_z"] = robust_z(df["peak"].to_numpy())
    df["is_outlier"] = (df["auc_z"].abs() >= z_threshold) | (df["peak_z"].abs() >= z_threshold)
    return df


# ============================================================
# 三维基因状态分类
# ============================================================

def classify_gene_state(promoter_atac, genebody_atac, rna,
                        thresholds=None):
    """
    根据三个信号水平分类基因状态。
    
    thresholds: dict 或 None
        {"promoter_atac_high": ..., "genebody_atac_high": ..., "rna_high": ...,
         "promoter_atac_low": ..., "genebody_atac_low": ..., "rna_low": ...}
        如果 None, 调用方需自行预先决定。
    """
    if thresholds is None:
        return "unknown"
    p_high = promoter_atac >= thresholds["promoter_atac_high"]
    p_low = promoter_atac < thresholds["promoter_atac_low"]
    g_high = genebody_atac >= thresholds["genebody_atac_high"]
    g_low = genebody_atac < thresholds["genebody_atac_low"]
    r_high = rna >= thresholds["rna_high"]
    r_low = rna < thresholds["rna_low"]
    
    # 五种典型状态
    if p_high and g_high and r_high:
        return "active"               # 完全激活
    if p_high and g_low and r_low:
        return "promoter_paused"      # 启动子开但暂停（双价/Paused）
    if p_low and g_low and r_low:
        return "silenced"             # 完全沉默
    if p_low and r_high:
        return "post_transcriptional" # RNA 高但启动子关，可能 mRNA 稳定
    if p_high and g_high and r_low:
        return "transcribed_unstable" # 转录但 RNA 不稳定（快速降解）
    return "intermediate"


def compute_global_thresholds(catalog_df, sample_names_atac, sample_names_rna,
                              high_q=0.75, low_q=0.25):
    """
    从 catalog 表中用全部基因估计阈值。
    用各样本均值的样本间均值作为基因水平指标。
    """
    def gene_level(df, samples, region, metric="mean"):
        cols = [f"atac_{s}_{region}_{metric}" if region != "gene_body_rna"
                else f"rna_{s}_gene_body_{metric}" for s in samples]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return np.array([])
        return df[cols].mean(axis=1).to_numpy()
    
    p_atac = gene_level(catalog_df, sample_names_atac, "promoter")
    g_atac = gene_level(catalog_df, sample_names_atac, "gene_body")
    rna = gene_level(catalog_df, sample_names_rna, "gene_body_rna")
    
    thresholds = {}
    if len(p_atac):
        thresholds["promoter_atac_high"] = float(np.quantile(p_atac, high_q))
        thresholds["promoter_atac_low"] = float(np.quantile(p_atac, low_q))
    if len(g_atac):
        thresholds["genebody_atac_high"] = float(np.quantile(g_atac, high_q))
        thresholds["genebody_atac_low"] = float(np.quantile(g_atac, low_q))
    if len(rna):
        thresholds["rna_high"] = float(np.quantile(rna, high_q))
        thresholds["rna_low"] = float(np.quantile(rna, low_q))
    return thresholds


# ============================================================
# 方向判断
# ============================================================

def classify_direction(promoter_atac_lfc, genebody_atac_lfc, rna_lfc,
                       lfc_threshold=0.5):
    """
    基于三个 log2FC 综合判断 ATAC/RNA 变化方向。
    
    lfc_threshold: 绝对值小于这个视为「无变化」。
    """
    def sign(x):
        if pd.isna(x):
            return None
        if x > lfc_threshold:
            return "up"
        if x < -lfc_threshold:
            return "down"
        return "flat"
    
    p, g, r = sign(promoter_atac_lfc), sign(genebody_atac_lfc), sign(rna_lfc)
    
    if None in (p, g, r):
        return "unknown"
    
    # 完全一致
    if p == g == r == "up":
        return "fully_activated"
    if p == g == r == "down":
        return "fully_repressed"
    if p == g == r == "flat":
        return "stable"
    
    # 部分一致 (ATAC 一致, RNA 跟随)
    if p == g and p != "flat":
        if r == p:
            return f"ATAC_RNA_concordant_{p}"
        if r == "flat":
            return f"ATAC_{p}_RNA_unchanged"
        return f"ATAC_{p}_RNA_opposite"
    
    # ATAC 内部不一致
    if p != g:
        return f"ATAC_decoupled_promoter_{p}_genebody_{g}_RNA_{r}"
    
    return "complex"


# ============================================================
# 绘图 —— 只展示 raw 信号
# ============================================================

def _promoter_sequence_step(seq_len):
    if seq_len <= 80:
        return 1, 7
    if seq_len <= 160:
        return 2, 6
    if seq_len <= 240:
        return 3, 5
    return 5, 4


def _add_promoter_sequence_axis(ax, promoter_seq, max_len=400):
    if not promoter_seq:
        return
    seq = str(promoter_seq).upper()
    seq_len = len(seq)
    if seq_len == 0 or seq_len > max_len:
        return

    step, fontsize = _promoter_sequence_step(seq_len)
    ticks = list(range(0, seq_len, step))
    if ticks[-1] != seq_len - 1:
        ticks.append(seq_len - 1)

    ax.set_xlim(0, max(seq_len - 1, 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [seq[i] for i in ticks],
        fontsize=fontsize,
        fontfamily="monospace",
        rotation=0,
    )
    ax.tick_params(axis="x", which="major", length=2, pad=2, labelbottom=True)
    ax.set_xlabel("Promoter sequence (gene-strand, 5' to 3')", fontsize=8)


def plot_gene_signals(gene_name, atac_promoter_raw,
                      atac_genebody_raw,
                      rna_raw,
                      output_png, title_suffix="", promoter_seq=None,
                      promoter_seq_max_len=400):
    """
    每个基因整合图：
        Left:  promoter ATAC raw
        Right: gene_body ATAC raw；有 RNA 时再增加 gene_body RNA raw
    """
    show_promoter_seq = (
        promoter_seq is not None
        and 0 < len(str(promoter_seq)) <= promoter_seq_max_len
    )
    fig_height = 7.2 if show_promoter_seq else 6.8
    fig = plt.figure(figsize=(15, fig_height))
    grid = fig.add_gridspec(
        2, 2,
        width_ratios=[1.05, 1.35],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.34 if show_promoter_seq else 0.26,
    )
    has_rna = bool(rna_raw)
    ax_promoter = fig.add_subplot(grid[:, 0])
    ax_atac_body = fig.add_subplot(grid[0, 1] if has_rna else grid[:, 1])
    ax_rna_body = fig.add_subplot(grid[1, 1]) if has_rna else None
    
    panels = [
        (ax_promoter, atac_promoter_raw, "Promoter ATAC (raw)", "raw ATAC"),
        (ax_atac_body, atac_genebody_raw, "Gene body ATAC (raw)", "raw ATAC"),
    ]
    if has_rna:
        panels.append((ax_rna_body, rna_raw, "Gene body RNA (raw)", "raw RNA"))
    
    for ax, data, title, ylabel in panels:
        if data:
            for sample, sig in data.items():
                ax.plot(to_array(sig), label=sample, linewidth=1.0, alpha=0.8)
            ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)

    if show_promoter_seq:
        _add_promoter_sequence_axis(ax_promoter, promoter_seq, promoter_seq_max_len)
    else:
        ax_promoter.set_xlabel("Position in promoter (bp)")
    
    ax_atac_body.set_xlabel("Position in gene body (bp)")
    if ax_rna_body is not None:
        ax_rna_body.set_xlabel("Position in gene body (bp)")
    
    fig.suptitle(f"{gene_name}  {title_suffix}", fontsize=12, y=1.005)
    fig.savefig(output_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_pca_2d(matrix_df, output_png, title="PCA"):
    values = matrix_df.drop(columns=["gene_id", "gene_name"], errors="ignore")
    samples = list(values.columns)
    x = np.log2(values.to_numpy(dtype=np.float64).T + 1.0)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1, x.std(axis=0))
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    scores = u[:, :2] * s[:2]
    explained = s ** 2 / (s ** 2).sum() if s.sum() > 0 else np.zeros_like(s)
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(scores[:, 0], scores[:, 1], s=50)
    for i, name in enumerate(samples):
        ax.text(scores[i, 0], scores[i, 1], name, fontsize=8)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return pd.DataFrame({"sample": samples, "PC1": scores[:, 0], "PC2": scores[:, 1]}), explained


# ============================================================
# 基因解析
# ============================================================

def resolve_genes(gtf_file, genes=None, gene_file=None,
                  promoter_upstream=200, promoter_downstream=200):
    from .read import GTFFullReader, GTFQueryReader
    
    items = []
    if gene_file:
        with open(gene_file) as f:
            items += [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if genes:
        items += [genes] if isinstance(genes, (str, int)) else list(genes)
    
    if not items:
        return GTFFullReader(gtf_file, promoter_upstream=promoter_upstream,
                              promoter_downstream=promoter_downstream).read()
    
    names, indices = [], []
    for it in items:
        if isinstance(it, int) or (isinstance(it, str) and it.isdigit()):
            indices.append(int(it))
        else:
            names.append(str(it).strip())
    
    parts = []
    if names:
        parts.append(GTFQueryReader(gtf_file, queries=names,
                                     promoter_upstream=promoter_upstream,
                                     promoter_downstream=promoter_downstream).read())
    if indices:
        all_df = GTFFullReader(gtf_file, promoter_upstream=promoter_upstream,
                                promoter_downstream=promoter_downstream).read()
        parts.append(all_df.iloc[indices])
    
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["gene_id"])
