import re
import os
import pandas as pd
import numpy as np
import scipy

try:
    import pyBigWig
except ImportError:
    pyBigWig = None

try:
    import matplotlib
except ImportError:
    matplotlib = None

try:
    import seaborn
except ImportError:
    seaborn = None

try:
    import plotly
except ImportError:
    plotly = None

try:
    import dash
except ImportError:
    dash = None

try:
    import jinja2
except ImportError:
    jinja2 = None

try:
    import pygenometracks
except ImportError:
    pygenometracks = None


# ============================================================
# GTF Reader 父类与子类
# ============================================================

class GTFReader:
    """GTF 读取父类，封装通用解析逻辑。"""

    # 用正则匹配 key "value" 这种模式
    ATTR_PATTERN = re.compile(r'(\S+)\s+"([^"]+)"')

    def __init__(
        self,
        gtf_file,
        feature_type="gene",
        promoter_upstream=200,
        promoter_downstream=200,
        only_chr_prefix=True,
    ):
        self.gtf_file = gtf_file
        self.feature_type = feature_type
        self.promoter_upstream = promoter_upstream
        self.promoter_downstream = promoter_downstream
        self.only_chr_prefix = only_chr_prefix

    @classmethod
    def parse_attributes(cls, attribute_string):
        """解析 GTF 第 9 列 attributes 字符串。"""
        attr = {}
        for match in cls.ATTR_PATTERN.finditer(attribute_string):
            key = match.group(1)
            value = match.group(2)
            attr[key] = value
        return attr

    def _parse_line(self, line):
        """
        解析单行 GTF。
        如果该行不需要处理 (注释 / 空行 / 列数不对 / feature 不匹配 / 染色体不符合)，
        返回 None。
        否则返回解析后的 record dict。
        """
        line = line.strip()

        if line == "" or line.startswith("#"):
            return None

        cols = line.split("\t")
        if len(cols) != 9:
            return None

        chrom = cols[0]
        source = cols[1]
        feature = cols[2]
        start_gtf = int(cols[3])
        end_gtf = int(cols[4])
        score = cols[5]
        strand = cols[6]
        frame = cols[7]
        attribute_string = cols[8]

        if feature != self.feature_type:
            return None

        if self.only_chr_prefix and not chrom.startswith("chr"):
            return None

        attr = self.parse_attributes(attribute_string)
        gene_id = attr.get("gene_id")
        if gene_id is None:
            return None

        gene_name = attr.get("gene_name", gene_id)
        gene_type = attr.get("gene_type", attr.get("gene_biotype"))
        gene_id_base = gene_id.split(".")[0]

        # GTF 1-based inclusive -> 0-based half-open
        start = start_gtf - 1
        end = end_gtf
        length = end - start

        # 计算 promoter
        if strand == "+":
            tss = start
            promoter_start = tss - self.promoter_upstream
            promoter_end = tss + self.promoter_downstream
        elif strand == "-":
            tss = end
            promoter_start = tss - self.promoter_downstream
            promoter_end = tss + self.promoter_upstream
        else:
            tss = start
            promoter_start = tss - self.promoter_upstream
            promoter_end = tss + self.promoter_downstream

        if promoter_start < 0:
            promoter_start = 0

        record = {
            "gene_id": gene_id,
            "gene_id_base": gene_id_base,
            "gene_name": gene_name,
            "gene_type": gene_type,

            "chrom": chrom,
            "start": start,
            "end": end,
            "strand": strand,
            "length": length,

            "tss": tss,
            "promoter_start": promoter_start,
            "promoter_end": promoter_end,

            "source": source,
            "score": score,
            "frame": frame,

            "start_gtf": start_gtf,
            "end_gtf": end_gtf,

            "attributes": attribute_string,
        }
        return record

    def _accept_record(self, record):
        """子类可重写：判断一条 record 是否要保留。默认全部接受。"""
        return True

    def _should_stop(self, found_count):
        """子类可重写：是否提前结束遍历。默认不提前结束。"""
        return False

    def read(self, return_dataframe=True):
        """遍历 GTF 文件，返回 DataFrame 或 list[dict]。"""
        records = []

        with open(self.gtf_file, "rt", encoding="utf-8") as gtf:
            for line in gtf:
                record = self._parse_line(line)
                if record is None:
                    continue

                if not self._accept_record(record):
                    continue

                records.append(record)

                if self._should_stop(len(records)):
                    break

        if return_dataframe:
            return pd.DataFrame(records) if records else pd.DataFrame()
        return records


class GTFFullReader(GTFReader):
    """全量读取 GTF 中所有 gene 行。"""
    pass


class GTFQueryReader(GTFReader):
    """按基因名 / gene_id 列表查询 GTF。"""

    def __init__(self, gtf_file, queries, verbose=True, **kwargs):
        super().__init__(gtf_file, **kwargs)

        # 处理 queries 输入
        if isinstance(queries, str):
            queries = [queries]
        queries = [q.strip() for q in queries if q.strip() != ""]
        self.queries = queries
        self.verbose = verbose

        # 分类：基因名 / 带版本号 ID / 不带版本号 ID
        self.name_set = set()
        self.id_full_set = set()
        self.id_base_set = set()

        for q in queries:
            if q.startswith("ENSG") or q.startswith("ENSMUSG"):
                self.id_full_set.add(q)
                self.id_base_set.add(q.split(".")[0])
            else:
                self.name_set.add(q)

        # 用于记录命中的 query key
        self._found_keys = set()

    def _accept_record(self, record):
        gene_id = record["gene_id"]
        gene_name = record["gene_name"]
        gene_id_base = record["gene_id_base"]

        matched_key = None
        if gene_name in self.name_set:
            matched_key = gene_name
        elif gene_id in self.id_full_set:
            matched_key = gene_id
        elif gene_id_base in self.id_base_set:
            matched_key = gene_id_base

        if matched_key is None:
            return False

        record["query"] = matched_key
        self._found_keys.add(matched_key)
        return True

    def _should_stop(self, found_count):
        return found_count >= len(self.queries)

    def read(self, return_dataframe=True):
        if len(self.queries) == 0:
            return pd.DataFrame() if return_dataframe else {}

        records = super().read(return_dataframe=False)

        # 报告没找到的
        if self.verbose:
            missing = []
            for q in self.queries:
                q_base = q.split(".")[0]
                if (q in self._found_keys) or (q_base in self._found_keys):
                    continue
                missing.append(q)
            if missing:
                print(f"[warning] 以下基因没有找到 (共 {len(missing)} 个):")
                for m in missing:
                    print(f"  - {m}")

        if return_dataframe:
            return pd.DataFrame(records) if records else pd.DataFrame()
        else:
            return {r["query"]: r for r in records}


# ============================================================
# FASTA 读取与序列提取
# ============================================================

_DNA_COMPLEMENT_TABLE = str.maketrans({
    "A": "T", "C": "G", "G": "C", "T": "A",
    "a": "t", "c": "g", "g": "c", "t": "a",
})


def reverse_complement(seq):
    """
    计算 DNA 序列的反向互补序列，也就是负链序列。
    N不用替换
    最后在""join的方法反向输出
    """
    return seq.translate(_DNA_COMPLEMENT_TABLE)[::-1]


_FASTA_CACHE = {}


def fasta_read(file, keep_chroms=None, store_reverse=False, use_cache=True):
    """
    读取一个 fasta 中的所有染色体的序列数据并分别储存。

    参数
    ----
    keep_chroms : set/list/None
        只读取这些染色体。None 表示读取全部染色体。
    store_reverse : bool
        是否预先保存整条负链。默认 False，因为后续通常只需要对切片
        做 reverse_complement，预先保存全基因组负链会明显增加时间和内存。
    use_cache : bool
        同一进程内缓存读取结果，避免重复加载同一个 FASTA。
    """
    keep_chroms = None if keep_chroms is None else frozenset(str(c) for c in keep_chroms)
    cache_key = (os.path.abspath(file), keep_chroms, bool(store_reverse))
    if use_cache and cache_key in _FASTA_CACHE:
        return _FASTA_CACHE[cache_key]

    sequences = {}
    midchr = None
    keep_current = False

    with open(file, "rt", encoding="utf-8") as fasta:
        for line in fasta:
            if not line.strip():
                continue

            if line.startswith(">"):
                midchr = line[1:].split()[0]
                keep_current = keep_chroms is None or midchr in keep_chroms
                if keep_current and midchr not in sequences:
                    sequences[midchr] = []
                continue

            if keep_current and midchr is not None:
                sequences[midchr].append(line.strip().upper())

    final = {}
    for chrom, seq_list in sequences.items():
        plus_seq = "".join(seq_list)
        final[chrom] = {"+": plus_seq}
        if store_reverse:
            final[chrom]["-"] = reverse_complement(plus_seq)

    if use_cache:
        _FASTA_CACHE[cache_key] = final
    return final


def get_gene_sequence(
    gene_info,
    fasta_dict,
    promoter_upstream=200,
    promoter_downstream=200,
    flank_upstream=2000,
    flank_downstream=2000,
    verbose=True,
    include_gene_body_seq=True,
    include_full_seq=True,
):
    """
    根据基因信息从基因组 FASTA 中切取相关序列。
    返回 promoter / gene_body / full 三段。
    """
    chrom = gene_info["chrom"]
    start = gene_info["start"]
    end = gene_info["end"]
    strand = gene_info["strand"]

    gene_id = gene_info.get("gene_id", "")
    gene_name = gene_info.get("gene_name", "")

    if chrom not in fasta_dict:
        if verbose:
            print(f"[warning] 染色体 {chrom} 不在 fasta_dict 中, 跳过 {gene_name}")
        return None

    chrom_seq = fasta_dict[chrom]["+"]
    chrom_len = len(chrom_seq)

    if strand == "+":
        promoter_start = start - promoter_upstream
        promoter_end = start + promoter_downstream
        full_start = start - flank_upstream
        full_end = end + flank_downstream
    elif strand == "-":
        promoter_start = end - promoter_downstream
        promoter_end = end + promoter_upstream
        full_start = start - flank_downstream
        full_end = end + flank_upstream
    else:
        if verbose:
            print(f"[warning] 未知 strand '{strand}', 跳过 {gene_name}")
        return None

    gene_body_start = start
    gene_body_end = end

    expected_promoter_len = promoter_end - promoter_start
    expected_gene_body_len = gene_body_end - gene_body_start
    expected_full_len = full_end - full_start

    def clip(s, e):
        return max(0, s), min(chrom_len, e)

    promoter_start_c, promoter_end_c = clip(promoter_start, promoter_end)
    gene_body_start_c, gene_body_end_c = clip(gene_body_start, gene_body_end)
    full_start_c, full_end_c = clip(full_start, full_end)

    if verbose:
        if (promoter_start_c != promoter_start) or (promoter_end_c != promoter_end):
            print(f"[info] {gene_name} promoter 区段超出染色体边界, 已截断")
        if (full_start_c != full_start) or (full_end_c != full_end):
            print(f"[info] {gene_name} full 区段超出染色体边界, 已截断")

    promoter_seq_plus = chrom_seq[promoter_start_c:promoter_end_c]

    if strand == "+":
        promoter_seq = promoter_seq_plus
    else:
        promoter_seq = reverse_complement(promoter_seq_plus)

    gene_body_seq = ""
    if include_gene_body_seq:
        gene_body_seq_plus = chrom_seq[gene_body_start_c:gene_body_end_c]
        gene_body_seq = (
            gene_body_seq_plus if strand == "+"
            else reverse_complement(gene_body_seq_plus)
        )

    full_seq = ""
    if include_full_seq:
        full_seq_plus = chrom_seq[full_start_c:full_end_c]
        full_seq = (
            full_seq_plus if strand == "+"
            else reverse_complement(full_seq_plus)
        )

    result = {
        "gene_id": gene_id,
        "gene_name": gene_name,
        "chrom": chrom,
        "strand": strand,

        "promoter_seq": promoter_seq,
        "gene_body_seq": gene_body_seq,
        "full_seq": full_seq,

        "promoter_coord": (chrom, promoter_start_c, promoter_end_c),
        "gene_body_coord": (chrom, gene_body_start_c, gene_body_end_c),
        "full_coord": (chrom, full_start_c, full_end_c),

        "promoter_len": len(promoter_seq),
        "promoter_expected_len": expected_promoter_len,

        "gene_body_len": gene_body_end_c - gene_body_start_c,
        "gene_body_expected_len": expected_gene_body_len,

        "full_len": full_end_c - full_start_c,
        "full_expected_len": expected_full_len,
    }

    return result
def get_genes_sequences_batch(
    genes,
    fasta_dict,
    promoter_upstream=200,
    promoter_downstream=200,
    flank_upstream=2000,
    flank_downstream=2000,
    verbose=True,
    return_dataframe=True,
    save_csv=None,
):
    """
    批量提取多个基因的序列。

    参数
    ----
    genes : pandas.DataFrame 或 str 或 list[dict]
        - DataFrame: 来自 GTFFullReader / GTFQueryReader 的输出
        - str:       CSV 文件路径
        - list[dict]: 基因信息字典列表

    fasta_dict : dict
        由 fasta_read 加载的基因组字典。

    promoter_upstream / promoter_downstream / flank_upstream / flank_downstream : int
        传给 get_gene_sequence 的参数。

    verbose : bool
        是否打印警告信息。

    return_dataframe : bool
        True  -> 返回 pandas.DataFrame
        False -> 返回 list[dict]

    save_csv : str 或 None
        如果给定路径，会把结果保存为 CSV 文件。

    返回
    ----
    results : pandas.DataFrame 或 list[dict]
    """

    # ---- 1. 统一输入格式为 list[dict] ----
    if isinstance(genes, str):
        # CSV 文件路径
        if not os.path.exists(genes):
            raise FileNotFoundError(f"找不到文件: {genes}")
        df = pd.read_csv(genes)
        gene_records = df.to_dict(orient="records")

    elif isinstance(genes, pd.DataFrame):
        gene_records = genes.to_dict(orient="records")

    elif isinstance(genes, list):
        gene_records = genes

    else:
        raise TypeError(
            f"genes 参数类型不支持: {type(genes)}，"
            f"应为 DataFrame / str (CSV路径) / list[dict]"
        )

    if len(gene_records) == 0:
        if verbose:
            print("[warning] 输入基因列表为空")
        return pd.DataFrame() if return_dataframe else []

    # ---- 2. 批量调用 get_gene_sequence ----
    results = []
    success_count = 0
    fail_count = 0

    for gene_info in gene_records:
        result = get_gene_sequence(
            gene_info,
            fasta_dict,
            promoter_upstream=promoter_upstream,
            promoter_downstream=promoter_downstream,
            flank_upstream=flank_upstream,
            flank_downstream=flank_downstream,
            verbose=verbose,
        )

        if result is None:
            fail_count += 1
            continue

        results.append(result)
        success_count += 1

    if verbose:
        print(f"[info] 批量提取完成: 成功 {success_count} 个, 失败 {fail_count} 个")

    # ---- 3. 保存 CSV (可选) ----
    if save_csv is not None:
        if len(results) == 0:
            if verbose:
                print("[warning] 结果为空，不保存 CSV")
        else:
            df_out = pd.DataFrame(results)
            df_out.to_csv(save_csv, index=False)
            if verbose:
                print(f"[info] 已保存到 {save_csv}")

    # ---- 4. 返回结果 ----
    if return_dataframe:
        return pd.DataFrame(results) if results else pd.DataFrame()
    else:
        return results
# ============================================================
# BigWig Reader (ATAC / RNA) —— 只保留原始信号, 不做统计
# ============================================================

class BigWigReader:
    """BigWig 读取父类，封装通用逻辑。返回每个碱基的原始信号。"""

    def __init__(self, bw_files, sample_names=None, label="signal", regions=None):
        """
        参数
        ----
        bw_files : str 或 list[str]
            一个或多个 BigWig 文件路径。

        sample_names : list[str] 或 None
            样本名称列表，长度需与 bw_files 一致。
            None 时自动用文件名 (不含扩展名) 命名。

        label : str
            信号类型标签，例如 "atac" / "rna"，用作输出列名前缀。
        """
        if isinstance(bw_files, str):
            bw_files = [bw_files]

        self.bw_files = bw_files
        self.label = label
        self.regions = regions or ("gene_body",)

        if sample_names is None:
            sample_names = [
                os.path.splitext(os.path.basename(f))[0] for f in bw_files
            ]
        if len(sample_names) != len(bw_files):
            raise ValueError("sample_names 长度必须与 bw_files 一致")
        self.sample_names = sample_names

        self._bw_handles = None
        self._bw_chroms = None

    def _open(self):
        if pyBigWig is None:
            raise ImportError("读取 BigWig 文件需要先安装 pyBigWig")
        if self._bw_handles is None:
            self._bw_handles = [pyBigWig.open(f) for f in self.bw_files]
            self._bw_chroms = [bw.chroms() for bw in self._bw_handles]

    def close(self):
        if self._bw_handles is not None:
            for bw in self._bw_handles:
                bw.close()
            self._bw_handles = None
            self._bw_chroms = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _fetch_one(self, bw, chroms, chrom, start, end):
        """
        从单个 BigWig 读取区间信号，返回长度等于 end-start 的 numpy 数组。
        如果区间不合法或染色体不存在，返回全 0 数组。
        """
        expected_len = max(0, end - start)
        if expected_len == 0:
            return np.zeros(0, dtype=np.float32)

        query_chrom = self._resolve_chrom_name(chroms, chrom)
        if query_chrom is None:
            return np.zeros(expected_len, dtype=np.float32)

        chrom_len = chroms[query_chrom]
        start_c = max(0, start)
        end_c = min(chrom_len, end)

        # 初始化全 0 数组, 边界外位置保持 0
        out = np.zeros(expected_len, dtype=np.float32)

        if start_c >= end_c:
            return out

        values = bw.values(query_chrom, start_c, end_c, numpy=True)
        values = np.nan_to_num(values, nan=0.0)

        # 把读到的信号放到对应位置 (考虑边界截断)
        offset = start_c - start
        out[offset:offset + len(values)] = values
        return out

    @staticmethod
    def _resolve_chrom_name(chroms, chrom):
        """兼容 GTF/FASTA 和 bigWig 之间 chr 前缀不一致的情况。"""
        if chrom in chroms:
            return chrom
        chrom = str(chrom)
        if chrom.startswith("chr"):
            alt = chrom[3:]
            if alt in chroms:
                return alt
        else:
            alt = "chr" + chrom
            if alt in chroms:
                return alt
        return None

    def fetch_region(self, chrom, start, end, strand="+", region_name="region"):
        """
        从所有 BigWig 文件读取同一区段。

        参数
        ----
        chrom, start, end : 基因组正链坐标 (0-based half-open)
        strand : '+' 或 '-'，负链会把信号反转使其与 mRNA 方向一致
        region_name : 区段名，用于命名输出列

        返回
        ----
        dict, 形如:
        {
            "{label}_{sample}_{region_name}_signal": np.ndarray (1D, 长度 = end-start)
        }
        """
        self._open()
        result = {}
        for bw, chroms, name in zip(self._bw_handles, self._bw_chroms, self.sample_names):
            values = self._fetch_one(bw, chroms, chrom, start, end)
            if strand == "-":
                values = values[::-1]  # 与序列方向一致
            key = f"{self.label}_{name}_{region_name}_signal"
            result[key] = values
        return result

    def _get_region_bounds(self, gene_info, region_name):
        if region_name == "promoter":
            if "promoter_start" not in gene_info or "promoter_end" not in gene_info:
                return None
            return int(gene_info["promoter_start"]), int(gene_info["promoter_end"])

        if region_name == "gene_body":
            return int(gene_info["start"]), int(gene_info["end"])

        raise ValueError(f"Unsupported region_name: {region_name}")

    def fetch_gene(self, gene_info):
        """按配置的区段读取单个基因，子类只需声明 label 和 regions。"""
        chrom = gene_info["chrom"]
        strand = gene_info.get("strand", "+")
        result = {}

        for region_name in self.regions:
            bounds = self._get_region_bounds(gene_info, region_name)
            if bounds is None:
                continue

            start, end = bounds
            result.update(self.fetch_region(
                chrom=chrom,
                start=start,
                end=end,
                strand=strand,
                region_name=region_name,
            ))

        return result


class ATACReader(BigWigReader):
    """ATAC-seq BigWig 读取器，读取 promoter 和 gene_body 区段。"""

    def __init__(self, bw_files, sample_names=None):
        super().__init__(
            bw_files,
            sample_names=sample_names,
            label="atac",
            regions=("promoter", "gene_body"),
        )


class RNAReader(BigWigReader):
    """RNA-seq BigWig 读取器，读取 gene_body 区段。"""

    def __init__(self, bw_files, sample_names=None):
        super().__init__(
            bw_files,
            sample_names=sample_names,
            label="rna",
            regions=("gene_body",),
        )


# ============================================================
# 总整合函数: 基因 -> 序列 + ATAC + RNA (信号数组)
# ============================================================

def assemble_gene_features(
    genes,
    fasta_dict,
    atac_reader=None,
    rna_reader=None,
    promoter_upstream=200,
    promoter_downstream=200,
    flank_upstream=2000,
    flank_downstream=2000,
    verbose=True,
    save_pickle=None,
    include_gene_body_seq=False,
    include_full_seq=False,
):
    """
    把 GTF / FASTA / ATAC / RNA 全部整合，输出每个基因的完整特征。
    ATAC 和 RNA 的输出是每个碱基位置的原始信号数组 (numpy)。

    参数
    ----
    genes : pandas.DataFrame 或 str 或 list[dict]
        基因信息表 (来自 GTFFullReader / GTFQueryReader 或 CSV)。

    fasta_dict : dict
        由 fasta_read 加载的基因组字典。

    atac_reader : ATACReader 或 None
        ATAC-seq 读取器，None 表示跳过。

    rna_reader : RNAReader 或 None
        RNA-seq 读取器，None 表示跳过。

    promoter_upstream / promoter_downstream / flank_upstream / flank_downstream : int
        传给 get_gene_sequence 的参数。

    verbose : bool
        是否打印警告信息。

    save_pickle : str 或 None
        如果给定，保存结果为 pickle 文件 (CSV 无法直接存 numpy 数组)。

    返回
    ----
    pandas.DataFrame
        每一行一个基因，包含：
        - 基因元信息 (gene_id, gene_name, chrom, strand, ...)
        - promoter / gene_body / full 三段坐标
        - promoter / gene_body / full 三段 FASTA 序列
        - ATAC 各样本的 promoter / gene_body 信号数组
        - RNA  各样本的 gene_body 信号数组
    """

    # ---- 1. 统一输入 ----
    if isinstance(genes, str):
        if not os.path.exists(genes):
            raise FileNotFoundError(f"找不到文件: {genes}")
        gene_records = pd.read_csv(genes).to_dict(orient="records")
    elif isinstance(genes, pd.DataFrame):
        gene_records = genes.to_dict(orient="records")
    elif isinstance(genes, list):
        gene_records = genes
    else:
        raise TypeError(f"genes 参数类型不支持: {type(genes)}")

    if len(gene_records) == 0:
        if verbose:
            print("[warning] 输入基因列表为空")
        return pd.DataFrame()

    # ---- 2. 打开 BigWig ----
    if atac_reader is not None:
        atac_reader._open()
    if rna_reader is not None:
        rna_reader._open()

    # ---- 3. 逐基因整合 ----
    results = []
    success_count = 0
    fail_count = 0

    for gene_info in gene_records:
        seq_result = get_gene_sequence(
            gene_info,
            fasta_dict,
            promoter_upstream=promoter_upstream,
            promoter_downstream=promoter_downstream,
            flank_upstream=flank_upstream,
            flank_downstream=flank_downstream,
            verbose=verbose,
            include_gene_body_seq=include_gene_body_seq,
            include_full_seq=include_full_seq,
        )
        if seq_result is None:
            fail_count += 1
            continue

        record = {
            "gene_id": gene_info.get("gene_id", ""),
            "gene_id_base": gene_info.get("gene_id_base", ""),
            "gene_name": gene_info.get("gene_name", ""),
            "gene_type": gene_info.get("gene_type", ""),
            "chrom": gene_info["chrom"],
            "strand": gene_info["strand"],
            "start": gene_info["start"],
            "end": gene_info["end"],
            "length": gene_info.get("length", gene_info["end"] - gene_info["start"]),
            "tss": gene_info.get("tss"),
            "promoter_start": gene_info.get("promoter_start"),
            "promoter_end": gene_info.get("promoter_end"),
        }

        record["promoter_seq"] = seq_result["promoter_seq"]
        record["gene_body_seq"] = seq_result["gene_body_seq"]
        record["full_seq"] = seq_result["full_seq"]

        record["promoter_coord"] = seq_result["promoter_coord"]
        record["gene_body_coord"] = seq_result["gene_body_coord"]
        record["full_coord"] = seq_result["full_coord"]

        record["promoter_len"] = seq_result["promoter_len"]
        record["gene_body_len"] = seq_result["gene_body_len"]
        record["full_len"] = seq_result["full_len"]

        if atac_reader is not None:
            record.update(atac_reader.fetch_gene(gene_info))

        if rna_reader is not None:
            record.update(rna_reader.fetch_gene(gene_info))

        results.append(record)
        success_count += 1

    # ---- 4. 关闭 BigWig ----
    if atac_reader is not None:
        atac_reader.close()
    if rna_reader is not None:
        rna_reader.close()

    if verbose:
        print(f"[info] 特征整合完成: 成功 {success_count} 个, 失败 {fail_count} 个")

    df_out = pd.DataFrame(results) if results else pd.DataFrame()

    if save_pickle is not None and not df_out.empty:
        df_out.to_pickle(save_pickle)
        if verbose:
            print(f"[info] 已保存到 {save_pickle}")

    return df_out
