import re
import os
import gzip
import json
import hashlib
import sqlite3
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
    GENE_ID_PATTERN = re.compile(r'(?:^|;)\s*gene_id\s+"([^"]+)"')
    TRANSCRIPT_ID_PATTERN = re.compile(r'(?:^|;)\s*transcript_id\s+"([^"]+)"')

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

        with _open_text_auto(self.gtf_file) as gtf:
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


def _open_text_auto(path):
    """Open plain-text or gzip-compressed annotation files."""
    path = str(path)
    opener = gzip.open if path.lower().endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8")


def _merge_intervals(intervals):
    """Merge overlapping exon intervals in genomic coordinate order."""
    merged = []
    for start, end in sorted(intervals):
        start, end = int(start), int(end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def attach_exon_intervals(gtf_file, gene_df):
    """
    Attach the union of all annotated exons for each selected gene.

    Exons from alternative transcripts are merged, so an RNA position is used
    once even when it occurs in several transcript isoforms. Coordinates remain
    0-based half-open genomic intervals.
    """
    if gene_df is None or gene_df.empty:
        return gene_df

    full_ids = set(gene_df["gene_id"].dropna().astype(str))
    base_ids = set(gene_df["gene_id_base"].dropna().astype(str))
    intervals_by_base = {gene_id: [] for gene_id in base_ids}
    completed = set()
    active_target = None

    with _open_text_auto(gtf_file) as gtf:
        for line in gtf:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9:
                continue
            if cols[2] not in {"gene", "exon"}:
                continue
            match = GTFReader.GENE_ID_PATTERN.search(cols[8])
            if match is None:
                continue
            gene_id = match.group(1)
            gene_id_base = gene_id.split(".")[0]

            # GENCODE groups a gene row and its child records together. Once
            # every selected gene block has ended, a profile query can stop
            # without scanning the rest of a multi-gigabyte GTF.
            if cols[2] == "gene":
                if active_target is not None:
                    completed.add(active_target)
                    if completed >= base_ids:
                        break
                active_target = gene_id_base if gene_id_base in base_ids else None
                continue

            if cols[2] != "exon":
                continue
            if gene_id not in full_ids and gene_id_base not in base_ids:
                continue
            intervals_by_base.setdefault(gene_id_base, []).append(
                (int(cols[3]) - 1, int(cols[4]))
            )

    merged_by_base = {
        gene_id: _merge_intervals(intervals)
        for gene_id, intervals in intervals_by_base.items()
    }
    out = gene_df.copy()
    out["exon_intervals"] = [
        merged_by_base.get(str(gene_id), [])
        for gene_id in out["gene_id_base"]
    ]
    out["exon_length"] = [
        int(sum(end - start for start, end in intervals))
        for intervals in out["exon_intervals"]
    ]
    return out


def query_genes_with_exons(gtf_file, queries, promoter_upstream=200,
                            promoter_downstream=200, verbose=True):
    """Find named genes and collect their exon unions in one GTF pass."""
    reader = GTFQueryReader(
        gtf_file,
        queries=queries,
        promoter_upstream=promoter_upstream,
        promoter_downstream=promoter_downstream,
        verbose=False,
    )
    records = []
    active_record = None
    active_intervals = []

    def finish_active():
        nonlocal active_record, active_intervals
        if active_record is None:
            return
        merged = _merge_intervals(active_intervals)
        active_record["exon_intervals"] = merged
        active_record["exon_length"] = int(
            sum(end - start for start, end in merged)
        )
        records.append(active_record)
        active_record = None
        active_intervals = []

    with _open_text_auto(gtf_file) as gtf:
        for line in gtf:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9:
                continue

            if cols[2] == "gene":
                finish_active()
                if len(reader._found_keys) >= len(reader.queries):
                    break
                record = reader._parse_line(line)
                if record is not None and reader._accept_record(record):
                    active_record = record
                continue

            if cols[2] != "exon" or active_record is None:
                continue
            match = GTFReader.GENE_ID_PATTERN.search(cols[8])
            exon_gene = match.group(1).split(".")[0] if match else ""
            if exon_gene == active_record["gene_id_base"]:
                active_intervals.append((int(cols[3]) - 1, int(cols[4])))

    finish_active()
    if verbose:
        found = {
            record.get("query", record["gene_id_base"])
            for record in records
        }
        missing = [
            query for query in reader.queries
            if query not in found and query.split(".")[0] not in found
        ]
        if missing:
            print(f"[warning] 以下基因没有找到 (共 {len(missing)} 个):")
            for query in missing:
                print(f"  - {query}")
    return pd.DataFrame(records) if records else pd.DataFrame()


class GTFAnnotationCache:
    """Persistent SQLite index for gene records and merged exon intervals."""

    SCHEMA_VERSION = "2"

    def __init__(self, gtf_file, cache_file=None):
        self.gtf_file = os.path.abspath(os.fspath(gtf_file))
        self.cache_file = (
            os.path.abspath(os.fspath(cache_file))
            if cache_file else self._default_cache_path()
        )

    def _default_cache_path(self):
        adjacent = self.gtf_file + ".atacread.sqlite"
        parent = os.path.dirname(adjacent) or "."
        if os.access(parent, os.W_OK):
            return adjacent
        stat = os.stat(self.gtf_file)
        identity = f"{self.gtf_file}|{stat.st_size}|{stat.st_mtime_ns}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "atacread")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"gtf-{digest}.sqlite")

    def _source_metadata(self):
        stat = os.stat(self.gtf_file)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "source_size": str(stat.st_size),
            "source_mtime_ns": str(stat.st_mtime_ns),
        }

    def is_valid(self):
        if not os.path.exists(self.cache_file):
            return False
        try:
            with sqlite3.connect(self.cache_file) as conn:
                rows = conn.execute("SELECT key, value FROM metadata").fetchall()
            return dict(rows) == self._source_metadata()
        except (sqlite3.Error, OSError):
            return False

    @staticmethod
    def _record_with_promoter(record, upstream, downstream):
        record = dict(record)
        start, end = int(record["start"]), int(record["end"])
        strand = record.get("strand", "+")
        tss = start if strand != "-" else end
        if strand == "-":
            promoter_start = tss - int(downstream)
            promoter_end = tss + int(upstream)
        else:
            promoter_start = tss - int(upstream)
            promoter_end = tss + int(downstream)
        record["tss"] = tss
        record["promoter_start"] = max(0, promoter_start)
        record["promoter_end"] = promoter_end
        return record

    def build(self, force=False):
        if not force and self.is_valid():
            return self.cache_file

        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        tmp_file = f"{self.cache_file}.tmp-{os.getpid()}"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        print(f"[gtf-cache] 创建索引: {self.cache_file}")
        parser = GTFReader(self.gtf_file)
        conn = sqlite3.connect(tmp_file)
        current_record = None
        exon_intervals = []
        transcript_intervals = {}
        gene_index = 0

        def finish_gene():
            nonlocal current_record, exon_intervals, transcript_intervals, gene_index
            if current_record is None:
                return
            merged = _merge_intervals(exon_intervals)
            merged_transcripts = {
                transcript_id: _merge_intervals(intervals)
                for transcript_id, intervals in transcript_intervals.items()
            }
            exon_length = int(sum(end - start for start, end in merged))
            conn.execute(
                """INSERT INTO genes
                   (gene_index, gene_id, gene_id_base, gene_name,
                    record_json, exon_json, transcript_json, exon_length)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    gene_index,
                    current_record["gene_id"],
                    current_record["gene_id_base"],
                    current_record["gene_name"],
                    json.dumps(current_record, ensure_ascii=False),
                    json.dumps(merged),
                    json.dumps(merged_transcripts),
                    exon_length,
                ),
            )
            gene_index += 1
            if gene_index % 10000 == 0:
                conn.commit()
                print(f"[gtf-cache] 已索引 {gene_index} 个基因")
            current_record = None
            exon_intervals = []
            transcript_intervals = {}

        try:
            conn.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE genes (
                    gene_index INTEGER PRIMARY KEY,
                    gene_id TEXT NOT NULL,
                    gene_id_base TEXT NOT NULL,
                    gene_name TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    exon_json TEXT NOT NULL,
                    transcript_json TEXT NOT NULL,
                    exon_length INTEGER NOT NULL
                );
                """
            )
            with _open_text_auto(self.gtf_file) as gtf:
                for line in gtf:
                    if not line or line.startswith("#"):
                        continue
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) != 9:
                        continue
                    if cols[2] == "gene":
                        finish_gene()
                        current_record = parser._parse_line(line)
                    elif cols[2] == "exon" and current_record is not None:
                        match = GTFReader.GENE_ID_PATTERN.search(cols[8])
                        exon_gene = match.group(1).split(".")[0] if match else ""
                        if exon_gene == current_record["gene_id_base"]:
                            interval = (int(cols[3]) - 1, int(cols[4]))
                            exon_intervals.append(interval)
                            transcript_match = GTFReader.TRANSCRIPT_ID_PATTERN.search(cols[8])
                            if transcript_match:
                                transcript_intervals.setdefault(
                                    transcript_match.group(1), []
                                ).append(interval)
            finish_gene()
            conn.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                self._source_metadata().items(),
            )
            conn.executescript(
                """
                CREATE INDEX genes_gene_id_idx ON genes(gene_id);
                CREATE INDEX genes_gene_id_base_idx ON genes(gene_id_base);
                CREATE INDEX genes_gene_name_idx ON genes(gene_name);
                """
            )
            conn.commit()
        except Exception:
            conn.close()
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            raise
        else:
            conn.close()

        os.replace(tmp_file, self.cache_file)
        print(f"[gtf-cache] 完成: {gene_index} 个基因")
        return self.cache_file

    def _deserialize(self, row, upstream, downstream, query=None):
        record = json.loads(row[0])
        record["exon_intervals"] = [tuple(interval) for interval in json.loads(row[1])]
        record["transcript_intervals"] = {
            transcript_id: [tuple(interval) for interval in intervals]
            for transcript_id, intervals in json.loads(row[2]).items()
        }
        record["exon_length"] = int(row[3])
        record["gene_index"] = int(row[4])
        if query is not None:
            record["query"] = query
        return self._record_with_promoter(record, upstream, downstream)

    def read(self, queries=None, indices=None, promoter_upstream=200,
             promoter_downstream=200):
        self.build()
        records = []
        with sqlite3.connect(self.cache_file) as conn:
            if queries:
                for query in queries:
                    query = str(query).strip()
                    query_base = query.split(".")[0]
                    rows = conn.execute(
                        """SELECT record_json, exon_json, transcript_json,
                                  exon_length, gene_index
                           FROM genes
                           WHERE gene_name = ? OR gene_id = ? OR gene_id_base = ?
                           ORDER BY gene_index""",
                        (query, query, query_base),
                    ).fetchall()
                    records.extend(
                        self._deserialize(
                            row, promoter_upstream, promoter_downstream, query=query
                        )
                        for row in rows
                    )
            elif indices:
                for index in indices:
                    row = conn.execute(
                        """SELECT record_json, exon_json, transcript_json,
                                  exon_length, gene_index
                           FROM genes WHERE gene_index = ?""",
                        (int(index),),
                    ).fetchone()
                    if row is not None:
                        records.append(self._deserialize(
                            row, promoter_upstream, promoter_downstream
                        ))
            else:
                rows = conn.execute(
                    """SELECT record_json, exon_json, transcript_json,
                              exon_length, gene_index
                       FROM genes ORDER BY gene_index"""
                )
                records.extend(
                    self._deserialize(row, promoter_upstream, promoter_downstream)
                    for row in rows
                )
        return pd.DataFrame(records) if records else pd.DataFrame()


def configure_rna_regions(gene_df, mode="exon_union", transcript_ids=None):
    """Choose merged gene exons or one explicitly requested transcript."""
    mode = str(mode).lower().replace("-", "_")
    if mode not in {"exon_union", "transcript"}:
        raise ValueError("rna_region_mode 必须是 exon_union 或 transcript")
    if gene_df is None or gene_df.empty:
        return gene_df

    out = gene_df.copy()
    if mode == "exon_union":
        intervals = [list(value or []) for value in out["exon_intervals"]]
        out["rna_intervals"] = intervals
        out["rna_region"] = "exon_union"
        out["rna_transcript_id"] = None
        out["rna_length"] = [
            int(sum(end - start for start, end in value)) for value in intervals
        ]
        return out

    if isinstance(transcript_ids, str):
        transcript_ids = [
            value.strip() for value in transcript_ids.split(",") if value.strip()
        ]
    requested = {str(value).strip() for value in (transcript_ids or []) if str(value).strip()}
    requested_base = {value.split(".")[0] for value in requested}
    if not requested:
        raise ValueError("transcript 模式必须通过 --transcripts 指定 transcript_id")

    selected_intervals = []
    selected_ids = []
    missing = []
    ambiguous = []
    for _, row in out.iterrows():
        transcript_map = row.get("transcript_intervals")
        transcript_map = transcript_map if isinstance(transcript_map, dict) else {}
        exact_matches = [
            transcript_id for transcript_id in transcript_map
            if transcript_id in requested
        ]
        matches = exact_matches or [
            transcript_id for transcript_id in transcript_map
            if transcript_id.split(".")[0] in requested_base
        ]
        if not matches:
            missing.append(str(row.get("gene_name", row.get("gene_id", "unknown"))))
            selected_intervals.append([])
            selected_ids.append(None)
            continue
        if len(matches) > 1:
            ambiguous.append(
                f"{row.get('gene_name', row.get('gene_id', 'unknown'))}: {matches}"
            )
            selected_intervals.append([])
            selected_ids.append(None)
            continue
        transcript_id = matches[0]
        selected_ids.append(transcript_id)
        selected_intervals.append(list(transcript_map[transcript_id]))

    if missing:
        raise ValueError("以下基因没有匹配到指定转录本: " + ", ".join(missing))
    if ambiguous:
        raise ValueError("每个基因只能指定一个转录本: " + "; ".join(ambiguous))

    out["rna_intervals"] = selected_intervals
    out["rna_region"] = "transcript"
    out["rna_transcript_id"] = selected_ids
    out["rna_length"] = [
        int(sum(end - start for start, end in value)) for value in selected_intervals
    ]
    return out


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


class FastaIndex:
    """Build and use a standard five-column FASTA ``.fai`` index."""

    def __init__(self, fasta_file, index_file=None):
        self.fasta_file = os.path.abspath(os.fspath(fasta_file))
        self.index_file = os.path.abspath(
            os.fspath(index_file) if index_file else self.fasta_file + ".fai"
        )

    def is_valid(self):
        if not os.path.exists(self.index_file):
            return False
        try:
            if os.stat(self.index_file).st_mtime_ns < os.stat(self.fasta_file).st_mtime_ns:
                return False
            return bool(self.read_entries())
        except (OSError, ValueError):
            return False

    def build(self, force=False):
        if not force and self.is_valid():
            return self.index_file
        if self.fasta_file.lower().endswith(".gz"):
            raise ValueError("当前内置 FAI 索引只支持未压缩 FASTA")

        os.makedirs(os.path.dirname(self.index_file) or ".", exist_ok=True)
        tmp_file = f"{self.index_file}.tmp-{os.getpid()}"
        entries = []
        current = None

        def finish_current():
            if current is not None:
                entries.append((
                    current["name"],
                    current["length"],
                    current["offset"],
                    current["line_bases"],
                    current["line_width"],
                ))

        print(f"[fasta-index] 创建索引: {self.index_file}")
        with open(self.fasta_file, "rb") as fasta:
            while True:
                line = fasta.readline()
                if not line:
                    break
                if line.startswith(b">"):
                    finish_current()
                    name = line[1:].split(None, 1)[0].decode("utf-8")
                    current = {
                        "name": name,
                        "length": 0,
                        "offset": fasta.tell(),
                        "line_bases": 0,
                        "line_width": 0,
                        "last_full_bases": None,
                    }
                    continue
                if current is None:
                    continue
                sequence_line = line.rstrip(b"\r\n")
                if not sequence_line:
                    continue
                bases = len(sequence_line)
                width = len(line)
                if current["line_bases"] == 0:
                    current["line_bases"] = bases
                    current["line_width"] = width
                elif current["last_full_bases"] is not None:
                    raise ValueError(
                        f"FASTA {current['name']} has sequence lines after a short final line"
                    )
                elif bases != current["line_bases"]:
                    if bases > current["line_bases"]:
                        raise ValueError(
                            f"FASTA {current['name']} has inconsistent line lengths"
                        )
                    current["last_full_bases"] = bases
                current["length"] += bases
        finish_current()

        try:
            with open(tmp_file, "wt", encoding="utf-8", newline="\n") as out:
                for entry in entries:
                    out.write("\t".join(str(value) for value in entry) + "\n")
            os.replace(tmp_file, self.index_file)
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        print(f"[fasta-index] 完成: {len(entries)} 条序列")
        return self.index_file

    def read_entries(self):
        entries = {}
        with open(self.index_file, "rt", encoding="utf-8") as index:
            for line in index:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 5:
                    raise ValueError(f"无效 FAI 行: {line.rstrip()}")
                entries[cols[0]] = {
                    "length": int(cols[1]),
                    "offset": int(cols[2]),
                    "line_bases": int(cols[3]),
                    "line_width": int(cols[4]),
                }
        return entries

    def fetch_chromosome(self, chrom):
        self.build()
        entries = self.read_entries()
        if chrom not in entries:
            raise KeyError(chrom)
        entry = entries[chrom]
        length = entry["length"]
        line_bases = entry["line_bases"]
        line_width = entry["line_width"]
        if length == 0:
            return ""
        full_lines, remainder = divmod(length, line_bases)
        byte_count = full_lines * line_width
        if remainder == 0:
            byte_count -= line_width - line_bases
        else:
            byte_count += remainder
        with open(self.fasta_file, "rb") as fasta:
            fasta.seek(entry["offset"])
            raw = fasta.read(byte_count)
        sequence = raw.replace(b"\n", b"").replace(b"\r", b"")[:length]
        if len(sequence) != length:
            raise ValueError(f"FAI 无法完整读取 {chrom}: {len(sequence)}/{length}")
        return sequence.decode("ascii").upper()


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
    source_stat = os.stat(file)
    cache_key = (
        os.path.abspath(file),
        source_stat.st_size,
        source_stat.st_mtime_ns,
        keep_chroms,
        bool(store_reverse),
    )
    if use_cache and cache_key in _FASTA_CACHE:
        return _FASTA_CACHE[cache_key]

    index = FastaIndex(file)
    index.build()
    available = index.read_entries()
    selected = list(available) if keep_chroms is None else [
        chrom for chrom in keep_chroms if chrom in available
    ]
    final = {}
    for chrom in selected:
        plus_seq = index.fetch_chromosome(chrom)
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
        if isinstance(bw_files, (str, os.PathLike)):
            bw_files = [bw_files]

        self.bw_files = [str(f) for f in bw_files]
        self.label = label
        self.regions = regions or ("gene_body",)

        if sample_names is None:
            sample_names = [
                os.path.splitext(os.path.basename(f))[0] for f in self.bw_files
            ]
        if len(sample_names) != len(self.bw_files):
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
    """RNA-seq BigWig reader; annotated exons are concatenated when available."""

    def __init__(self, bw_files, sample_names=None):
        super().__init__(
            bw_files,
            sample_names=sample_names,
            label="rna",
            regions=("gene_body",),
        )

    def fetch_gene(self, gene_info):
        """
        Read the union of annotated exons in transcript orientation.

        Falling back to the whole gene body keeps compatibility with custom
        gene records that do not contain ``exon_intervals``.
        """
        intervals = gene_info.get("rna_intervals", gene_info.get("exon_intervals"))
        if not isinstance(intervals, (list, tuple)) or not intervals:
            return super().fetch_gene(gene_info)

        self._open()
        chrom = gene_info["chrom"]
        strand = gene_info.get("strand", "+")
        gene_start = int(gene_info["start"])
        gene_end = int(gene_info["end"])
        clipped = _merge_intervals([
            (max(gene_start, int(start)), min(gene_end, int(end)))
            for start, end in intervals
        ])

        result = {}
        for bw, chroms, name in zip(self._bw_handles, self._bw_chroms, self.sample_names):
            gene_values = self._fetch_one(
                bw, chroms, chrom, gene_start, gene_end
            )
            chunks = [
                gene_values[start - gene_start:end - gene_start]
                for start, end in clipped
                if end > start
            ]
            values = (
                np.concatenate(chunks).astype(np.float32, copy=False)
                if chunks else np.zeros(0, dtype=np.float32)
            )
            if strand == "-":
                values = values[::-1]
            result[f"rna_{name}_gene_body_signal"] = values
        return result


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
            "exon_length": gene_info.get("exon_length"),
            "rna_region": gene_info.get(
                "rna_region",
                "exon_union" if gene_info.get("exon_intervals") else "gene_body",
            ),
            "rna_transcript_id": gene_info.get("rna_transcript_id"),
            "rna_length": gene_info.get("rna_length", gene_info.get("exon_length")),
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
