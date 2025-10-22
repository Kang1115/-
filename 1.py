# -*- coding: utf-8 -*-
"""
LEVEL 4 · 工业级板块联动网络框架（军工板块示例）
=====================================================
特性亮点：
- 模块化/可复用：Config + Builder + IO + Metrics + Pipeline
- 大文件友好：CSV 分块读取 / Parquet / Feather；自动编码/列名识别
- 统计稳健：去极值；覆盖阈值；小缺口填补；有效重叠样本；负相关截断
- 资源友好：DTW 采用两行滚动数组（O(T) 内存）；Sakoe–Chiba 约束
- 可解释输出：分步诊断矩阵 + 排名 + 社区 +（可选）Granger 有向边
- CLI 一把梭：支持所有关键超参数；日志清晰

快速开始（两种方式）：
1) 直接修改下方“【输入文件路径】”与“【输出目录】”，然后 `python industrial_linkage_framework_level4.py`
2) 使用命令行参数（推荐）：
   python industrial_linkage_framework_level4.py \
     --input /ABS/PATH/TO/stock_code_Military_Industry_Chinese.csv \
     --outdir ./outputs \
     --window 52 --max-stocks 300 \
     --dtw-band-ratio 0.10 --alpha 0.45 --beta 0.35 --gamma 0.20 \
     --qu 0.90 --ql 0.10 --run-granger --granger-topk 2000 --granger-maxlag 2

依赖：
  pandas numpy networkx tqdm (可选) statsmodels(可选) python-louvain(可选)
安装：
  pip install pandas numpy networkx tqdm statsmodels python-louvain
"""
from __future__ import annotations
import os
import sys
import gc
import math
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# 进度条（可选）
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# 图与社区
try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    import community as community_louvain  # python-louvain
except Exception:  # pragma: no cover
    community_louvain = None

# Granger（可选）
try:
    from statsmodels.tsa.stattools import grangercausalitytests
except Exception:  # pragma: no cover
    grangercausalitytests = None


# ========================【输入文件路径】与【输出目录】========================
# 方式一（快捷）：直接编辑下面两行，运行本脚本（CLI 方式会覆盖这两项）
INPUT_FILE_PATH = "data/军工板块2020_01_01——2025_10_17周线数据数据（未清洗）.csv"  # ←←←★ 在此处填写你的“输入数据文件”绝对或相对路径
OUTPUT_DIR_PATH = "./output"  # ←←←★ 在此处填写你的“输出目录”（不存在会自动创建）
# ===========================================================================


# =============================== 配置模型 ==================================
@dataclass
class LinkageConfig:
    # IO
    input_path: str = INPUT_FILE_PATH
    output_dir: str = OUTPUT_DIR_PATH
    csv_chunksize: Optional[int] = None   # 大 CSV 可设置为如 2_000_000 行

    # 窗口与筛选
    window_weeks: int = 52
    fallback_windows: Tuple[int, int, int] = (104, 208, -1)  # -1 表示全样本
    max_stocks: int = 300

    # DTW 与相似度
    dtw_band_ratio: float = 0.10

    # 融合权重
    alpha: float = 0.45
    beta: float = 0.35
    gamma: float = 0.20

    # 尾部分位
    q_upper: float = 0.90
    q_lower: float = 0.10

    # Granger（可选）
    run_granger: bool = False
    granger_topK: int = 2000
    granger_maxlag: int = 2
    granger_alpha: float = 0.05

    # 相关/覆盖
    min_overlap_corr: int = 8


# =============================== 工具函数 ==================================
class IOUtils:
    @staticmethod
    def _standardize_code(code: str) -> str:
        if pd.isna(code):
            return code
        s = str(code).strip().upper()
        # 常见格式：600000, 600000.SH, SH600000, 600000-SH 等
        for sep in ['.', '-', '_', '/']:
            if sep in s:
                parts = s.split(sep)
                for p in reversed(parts):
                    if p.isdigit() and len(p) == 6:
                        return p
                for p in parts:
                    if p.isdigit() and len(p) == 6:
                        return p
        digits = ''.join(ch for ch in s if ch.isdigit())
        if len(digits) >= 6:
            return digits[-6:]
        return s

    @staticmethod
    def _infer_columns(df: pd.DataFrame) -> Dict[str, str]:
        cols = {c.lower(): c for c in df.columns}
        def find(*cands):
            for x in cands:
                if x in cols:
                    return cols[x]
            # 宽松包含匹配
            for k in cols:
                if any(x in k for x in cands):
                    return cols[k]
            return None
        mapping = dict(
            stock_code=find('stock_code', 'code', 'sec_code', 'ts_code', 'ticker'),
            trade_date=find('trade_date', 'date', 'week', 'trade_week'),
            open=find('open', 'open_price'),
            close=find('close', 'close_price', 'last', 'last_price'),
            high=find('high', 'high_price'),
            low=find('low', 'low_price'),
            volume=find('volume', 'vol'),
            amount=find('amount', 'amt', 'turnover'),
            change_pct=find('change_pct', 'pct_chg', 'pct_change', 'return', 'ret'),
            change=find('change', 'chg'),
            turnover_ratio=find('turnover_ratio', 'turnoverrate', 'turn_rate'),
            pre_close=find('pre_close', 'preclose', 'prev_close', 'previous_close'),
        )
        return mapping

    @staticmethod
    def _parse_dates_safe(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, errors='coerce').dt.date

    @staticmethod
    def _winsorize_series(x: pd.Series, lower_q=0.005, upper_q=0.995) -> pd.Series:
        if x.notna().sum() <= 100:
            return x
        lo = np.nanquantile(x, lower_q)
        hi = np.nanquantile(x, upper_q)
        return x.clip(lower=lo, upper=hi)

    @staticmethod
    def sniff_csv_encoding(path: str, nbytes: int = 4096) -> str:
        # 简易探测：utf-8 / gbk 优先
        encs = ["utf-8", "utf-8-sig", "gbk", "cp936"]
        for enc in encs:
            try:
                with open(path, 'r', encoding=enc) as f:
                    f.read(nbytes)
                return enc
            except Exception:
                continue
        return 'utf-8'

    @staticmethod
    def read_table(input_path: str, usecols: Optional[List[str]] = None, chunksize: Optional[int] = None) -> pd.DataFrame:
        ext = os.path.splitext(input_path)[1].lower()
        if ext in [".parquet", ".pq"]:
            return pd.read_parquet(input_path, columns=usecols)
        if ext in [".feather", ".ft"]:
            return pd.read_feather(input_path, columns=usecols)
        # CSV：可分块
        enc = IOUtils.sniff_csv_encoding(input_path)
        if chunksize:
            parts = []
            for ch in pd.read_csv(input_path, usecols=usecols, chunksize=chunksize, encoding=enc):
                parts.append(ch)
            return pd.concat(parts, ignore_index=True)
        return pd.read_csv(input_path, usecols=usecols, encoding=enc)


# =============================== 指标计算 ==================================
class Metrics:
    @staticmethod
    def corr_nonneg(R: pd.DataFrame, min_overlap: int = 8) -> pd.DataFrame:
        cols = R.columns.tolist()
        N = len(cols)
        C = np.zeros((N, N), dtype=float)
        C[:] = np.nan
        X = R.values
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        Z = (X - mu) / sd
        Z[:, (sd == 0) | np.isnan(sd)] = np.nan
        for i in range(N):
            C[i, i] = 1.0
            zi = Z[:, i]
            for j in range(i + 1, N):
                zj = Z[:, j]
                mask = np.isfinite(zi) & np.isfinite(zj)
                n = int(mask.sum())
                if n >= min_overlap:
                    xi = zi[mask]
                    xj = zj[mask]
                    xi = xi - xi.mean()
                    xj = xj - xj.mean()
                    denom = xi.std(ddof=1) * xj.std(ddof=1)
                    if denom > 0:
                        rho = float((xi * xj).sum() / ((n - 1) * xi.std(ddof=1) * xj.std(ddof=1)))
                        C[i, j] = rho
                        C[j, i] = rho
        C = np.nan_to_num(C, nan=0.0)
        C = np.maximum(C, 0.0)  # 负相关截断
        return pd.DataFrame(C, index=cols, columns=cols)

    @staticmethod
    def _dtw_distance_constrained(a: np.ndarray, b: np.ndarray, band: int) -> float:
        T = min(len(a), len(b))
        if T == 0:
            return float('inf')
        a = a[:T]
        b = b[:T]
        band = max(1, int(band))
        INF = 1e18
        prev = np.full(T + 1, INF)
        curr = np.full(T + 1, INF)
        prev[0] = 0.0
        for i in range(1, T + 1):
            j_start = max(1, i - band)
            j_end = min(T, i + band)
            curr[:] = INF
            ai = a[i - 1]
            for j in range(j_start, j_end + 1):
                cost = abs(ai - b[j - 1])
                curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, prev
        return float(prev[T])

    @staticmethod
    def dtw_similarity(R: pd.DataFrame, band_ratio: float = 0.10) -> pd.DataFrame:
        cols = R.columns.tolist()
        X = R.values.astype(float)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        Z = (X - mu) / sd
        Z[:, (sd == 0) | np.isnan(sd)] = 0.0
        Z = np.nan_to_num(Z, nan=0.0)
        T = Z.shape[0]
        band = int(math.floor(band_ratio * T))
        N = len(cols)
        D = np.zeros((N, N), dtype=float)
        for i in tqdm(range(N), desc="DTW distance"):
            D[i, i] = 0.0
            ai = Z[:, i]
            for j in range(i + 1, N):
                bj = Z[:, j]
                d = Metrics._dtw_distance_constrained(ai, bj, band)
                D[i, j] = D[j, i] = d
        # 线性归一为相似度
        mask = ~np.eye(N, dtype=bool)
        off = D[mask]
        dmin = float(off.min()) if off.size else 0.0
        dmax = float(off.max()) if off.size else 1.0
        if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
            S = np.eye(N, dtype=float)
        else:
            S = np.ones((N, N), dtype=float)
            S[mask] = 1.0 - (off - dmin) / (dmax - dmin)
        return pd.DataFrame(S, index=cols, columns=cols)

    @staticmethod
    def tail_dependence(R: pd.DataFrame, q_u: float = 0.90, q_l: float = 0.10) -> pd.DataFrame:
        cols = R.columns.tolist()
        X = R.values
        qu = np.nanquantile(X, q_u, axis=0)
        ql = np.nanquantile(X, q_l, axis=0)
        N = len(cols)
        S = np.zeros((N, N), dtype=float)
        for i in range(N):
            ri = R.iloc[:, i].values
            for j in range(i, N):
                rj = R.iloc[:, j].values
                valid = (~np.isnan(ri)) & (~np.isnan(rj))
                n = int(valid.sum())
                if n == 0:
                    lam = 0.0
                else:
                    up = (ri[valid] > qu[i]) & (rj[valid] > qu[j])
                    dn = (ri[valid] < ql[i]) & (rj[valid] < ql[j])
                    lam = 0.5 * (up.mean() + dn.mean())
                S[i, j] = S[j, i] = float(lam)
        return pd.DataFrame(S, index=cols, columns=cols)

    @staticmethod
    def fuse(S_corr: pd.DataFrame, S_dtw: pd.DataFrame, S_tail: pd.DataFrame,
             alpha: float, beta: float, gamma: float) -> pd.DataFrame:
        idx = S_corr.index
        S_dtw = S_dtw.reindex(index=idx, columns=idx, fill_value=0.0)
        S_tail = S_tail.reindex(index=idx, columns=idx, fill_value=0.0)
        W = alpha * S_corr.values + beta * S_dtw.values + gamma * S_tail.values
        np.fill_diagonal(W, 0.0)
        return pd.DataFrame(W, index=idx, columns=idx)

    @staticmethod
    def linkage_scores(W: pd.DataFrame) -> pd.Series:
        return W.sum(axis=1)

    @staticmethod
    def detect_communities(W: pd.DataFrame) -> pd.Series:
        if nx is None:
            return pd.Series(0, index=W.index)
        G = nx.from_pandas_adjacency(W, create_using=nx.Graph)
        if community_louvain is not None:
            part = community_louvain.best_partition(G, weight='weight', resolution=1.0, random_state=42)
            s = pd.Series(part).reindex(W.index)
            # 重新编码为 0..k-1
            u = {c: i for i, c in enumerate(sorted(s.dropna().unique()))}
            return s.map(u)
        else:
            comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
            lab = {}
            for cid, nodes in enumerate(comms):
                for n in nodes:
                    lab[n] = cid
            return pd.Series(lab).reindex(W.index)

    @staticmethod
    def top_k_pairs_by_corr(C: pd.DataFrame, K: int) -> List[Tuple[str, str]]:
        tri = []
        cols = C.columns.tolist()
        N = len(cols)
        for i in range(N):
            for j in range(i + 1, N):
                tri.append((cols[i], cols[j], C.iat[i, j]))
        tri.sort(key=lambda x: x[2], reverse=True)
        return [(a, b) for a, b, r in tri if np.isfinite(r) and r > 0][:K]

    @staticmethod
    def granger_edges(R: pd.DataFrame, pairs: List[Tuple[str, str]], maxlag: int, alpha: float) -> pd.DataFrame:
        if grangercausalitytests is None or not pairs:
            return pd.DataFrame(columns=["src", "dst", "pvalue", "lag"])
        rows = []
        for (i, j) in tqdm(pairs, desc="Granger tests"):
            for src, dst in [(i, j), (j, i)]:
                data = pd.concat([R[dst], R[src]], axis=1).dropna()
                if len(data) < max(16, maxlag * 4):
                    continue
                data.columns = ["y", "x"]
                try:
                    res = grangercausalitytests(data[["y", "x"]], maxlag=maxlag, verbose=False)
                except Exception:
                    continue
                best_p = 1.0
                best_lag = None
                for lag, out in res.items():
                    p = out[0].get('ssr_ftest', (np.nan,))[1]
                    if p is not None and np.isfinite(p) and p < best_p:
                        best_p, best_lag = float(p), int(lag)
                if best_lag is not None and best_p < alpha:
                    rows.append((src, dst, best_p, best_lag))
        if rows:
            df = pd.DataFrame(rows, columns=["src", "dst", "pvalue", "lag"]).sort_values('pvalue')
        else:
            df = pd.DataFrame(columns=["src", "dst", "pvalue", "lag"])
        return df


# =============================== 主构建器 ==================================
class LinkageNetworkBuilder:
    def __init__(self, cfg: LinkageConfig):
        self.cfg = cfg

    # --- 读入 & 预处理 ---
    def load_and_prepare(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        # 先读小样本推断列
        ext = os.path.splitext(self.cfg.input_path)[1].lower()
        if ext in [".parquet", ".pq", ".feather", ".ft"]:
            head = IOUtils.read_table(self.cfg.input_path)
            if len(head) > 10000:
                head = head.head(10000).copy()
        else:
            enc = IOUtils.sniff_csv_encoding(self.cfg.input_path)
            head = pd.read_csv(self.cfg.input_path, nrows=10000, encoding=enc)
        colmap = IOUtils._infer_columns(head)

        if colmap.get('stock_code') is None or colmap.get('trade_date') is None:
            raise ValueError("无法识别 'stock_code' 或 'trade_date' 字段")
        need = [c for c in colmap.values() if c is not None]
        if colmap.get('change_pct') is None:
            for k in ['close', 'pre_close']:
                if colmap.get(k) is not None and colmap[k] not in need:
                    need.append(colmap[k])
        # 正式读
        df = IOUtils.read_table(self.cfg.input_path, usecols=need, chunksize=self.cfg.csv_chunksize)
        # 标准化
        code_col = colmap['stock_code']
        date_col = colmap['trade_date']
        df[code_col] = df[code_col].map(IOUtils._standardize_code)
        df[date_col] = IOUtils._parse_dates_safe(df[date_col])
        df = df.dropna(subset=[code_col, date_col]).sort_values([date_col, code_col])
        return df, colmap

    def prepare_returns_matrix(self, df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
        code_col = colmap['stock_code']
        date_col = colmap['trade_date']
        # 收益率
        cp = colmap.get('change_pct')
        cl = colmap.get('close')
        pcl = colmap.get('pre_close')
        if cp is None and (cl is None or pcl is None):
            raise ValueError("缺少 change_pct 或 close/pre_close 字段")
        if cp is None:
            df['__ret'] = 100.0 * (df[cl] / df[pcl] - 1.0)
        else:
            df['__ret'] = df[cp].astype(float)
            if cl is not None and pcl is not None:
                m = df['__ret'].isna()
                df.loc[m, '__ret'] = 100.0 * (df.loc[m, cl] / df.loc[m, pcl] - 1.0)
        # 去极值
        df['__ret'] = IOUtils._winsorize_series(df['__ret'])
        # 同一周重复记为均值
        g = df.groupby([date_col, code_col], as_index=False)['__ret'].mean()
        g = g.sort_values([date_col, code_col])
        # 全局周序列（保证对齐）
        weeks = np.array(sorted(g[date_col].dropna().unique()))
        def make_R_for_window(wlen: int) -> pd.DataFrame:
            if wlen == -1 or len(weeks) <= wlen:
                sel = weeks
            else:
                sel = weeks[-wlen:]
            sub = g[g[date_col].isin(sel)]
            R = sub.pivot(index=date_col, columns=code_col, values='__ret').sort_index()
            return R
        # 自动回退
        for w in [self.cfg.window_weeks] + list(self.cfg.fallback_windows):
            R = make_R_for_window(w)
            if R.shape[1] >= 2:
                break
        # 覆盖筛选 + 小缺口填补
        T = R.shape[0]
        thresh1 = max(10, int(math.floor(0.3 * T)))
        cov = R.notna().sum(axis=0)
        keep = cov[cov >= thresh1].index.tolist()
        if len(keep) == 0:
            N = min(150, self.cfg.max_stocks, R.shape[1])
            keep = cov.sort_values(ascending=False).head(N).index.tolist()
        R = R[keep]
        # 小缺口填补
        R = R.ffill(limit=2).bfill(limit=2)
        # 二次覆盖
        thresh2 = max(8, int(math.floor(0.3 * T)))
        R = R.loc[:, R.notna().sum(axis=0) >= thresh2]
        # 上限截断
        if R.shape[1] > self.cfg.max_stocks:
            keep2 = R.notna().sum().sort_values(ascending=False).head(self.cfg.max_stocks).index
            R = R[keep2]
        return R

    # --- 计算三视角、融合、排名等 ---
    def build_all(self, R: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        S_corr = Metrics.corr_nonneg(R, min_overlap=self.cfg.min_overlap_corr)
        S_dtw  = Metrics.dtw_similarity(R, band_ratio=self.cfg.dtw_band_ratio)
        S_tail = Metrics.tail_dependence(R, q_u=self.cfg.q_upper, q_l=self.cfg.q_lower)
        W = Metrics.fuse(S_corr, S_dtw, S_tail, self.cfg.alpha, self.cfg.beta, self.cfg.gamma)
        ranking = Metrics.linkage_scores(W).sort_values(ascending=False).rename("net_linkage_score").reset_index().rename(columns={'index': 'stock_code'})
        comm = Metrics.detect_communities(W).rename('community').reset_index().rename(columns={'index': 'stock_code'})
        out = {
            'Corr': S_corr,
            'DTW_Sim': S_dtw,
            'Tail': S_tail,
            'W': W,
            'Ranking': ranking,
            'Community': comm
        }
        # 可选：Granger
        if self.cfg.run_granger:
            pairs = Metrics.top_k_pairs_by_corr(S_corr, self.cfg.granger_topK)
            G = Metrics.granger_edges(R, pairs, self.cfg.granger_maxlag, self.cfg.granger_alpha)
            out['Granger'] = G
        return out

    # --- 保存 ---
    def save_outputs(self, results: Dict[str, pd.DataFrame]):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        p = lambda name: os.path.join(self.cfg.output_dir, name)
        # 诊断矩阵
        results['Corr'].to_csv(p("军工_价格相关矩阵.csv"), float_format="%.6f", encoding="utf-8-sig")
        # 将 1-S 作为“归一化距离”输出，便于与“DTW距离矩阵”口径对应
        D_norm = 1.0 - results['DTW_Sim'].values
        pd.DataFrame(D_norm, index=results['DTW_Sim'].index, columns=results['DTW_Sim'].columns)\
            .to_csv(p("军工_DTW距离矩阵.csv"), float_format="%.6f", encoding="utf-8-sig")
        results['Tail'].to_csv(p("军工_尾部依赖矩阵.csv"), float_format="%.6f", encoding="utf-8-sig")
        results['W'].to_csv(p("军工_综合权重矩阵.csv"), float_format="%.6f", encoding="utf-8-sig")
        results['Ranking'].to_csv(p("军工_综合联动系数_排名.csv"), index=False, float_format="%.6f", encoding="utf-8-sig")
        results['Community'].to_csv(p("军工_社区划分.csv"), index=False, encoding="utf-8-sig")
        if 'Granger' in results:
            results['Granger'].to_csv(p("军工_Granger_有向边.csv"), index=False, float_format="%.6f", encoding="utf-8-sig")
        # 简要报告
        summary = {
            'config': asdict(self.cfg),
            'shape': {k: (int(v.shape[0]), int(v.shape[1])) if isinstance(v, pd.DataFrame) else None for k, v in results.items()},
        }
        with open(p("summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        # 控制台预览
        print("\n[Top 15 综合联动系数]")
        print(results['Ranking'].head(15).to_string(index=False))
        print("\n[社区统计]")
        comm = results['Community']
        if not comm.empty:
            print(comm['community'].value_counts().to_string())
        else:
            print("（空）")
        if 'Granger' in results:
            G = results['Granger']
            print(f"\n[Granger 有向边数] {len(G)} (alpha={self.cfg.granger_alpha})")
            print(G.head(10).to_string(index=False))
        print("\n[输出目录]", os.path.abspath(self.cfg.output_dir))


# =============================== 命令行入口 =================================

def build_argparser():
    p = argparse.ArgumentParser(description="LEVEL 4 · 工业级板块联动网络（军工示例）")
    p.add_argument('--input', type=str, help='输入文件路径（CSV/Parquet/Feather）')
    p.add_argument('--outdir', type=str, help='输出目录')
    p.add_argument('--window', type=int, default=52, help='目标窗口长度（周），默认52；自动回退至104/208/全样本')
    p.add_argument('--max-stocks', type=int, default=300, help='最大股票数上限，默认300')
    p.add_argument('--dtw-band-ratio', type=float, default=0.10, help='DTW Sakoe–Chiba 带宽比，默认0.10')
    p.add_argument('--alpha', type=float, default=0.45, help='融合权重 α（相关）')
    p.add_argument('--beta', type=float, default=0.35, help='融合权重 β（DTW）')
    p.add_argument('--gamma', type=float, default=0.20, help='融合权重 γ（尾部）')
    p.add_argument('--qu', type=float, default=0.90, help='上尾分位 Q_U，默认0.90')
    p.add_argument('--ql', type=float, default=0.10, help='下尾分位 Q_L，默认0.10')
    p.add_argument('--run-granger', action='store_true', help='是否执行 Granger 方向性分析')
    p.add_argument('--granger-topk', type=int, default=2000, help='Granger 仅在相关 Top-K 对上执行，默认2000')
    p.add_argument('--granger-maxlag', type=int, default=2, help='Granger 最大滞后阶，默认2')
    p.add_argument('--granger-alpha', type=float, default=0.05, help='Granger 显著性阈，默认0.05')
    p.add_argument('--csv-chunksize', type=int, default=None, help='CSV 分块读取行数（建议超大 CSV 使用，如 2_000_000）')
    return p


def main_cli():
    if len(sys.argv) == 1:
        # 直接运行脚本时采用顶部的【输入文件路径】与【输出目录】
        cfg = LinkageConfig(
            input_path=INPUT_FILE_PATH,
            output_dir=OUTPUT_DIR_PATH,
        )
    else:
        args = build_argparser().parse_args()
        cfg = LinkageConfig(
            input_path=args.input or INPUT_FILE_PATH,
            output_dir=args.outdir or OUTPUT_DIR_PATH,
            window_weeks=args.window,
            max_stocks=args.max_stocks,
            dtw_band_ratio=args.dtw_band_ratio,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            q_upper=args.qu,
            q_lower=args.ql,
            run_granger=args.run_granger,
            granger_topK=args.granger_topk,
            granger_maxlag=args.granger_maxlag,
            granger_alpha=args.granger_alpha,
            csv_chunksize=args.csv_chunksize,
        )
    print("[Config]", json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    builder = LinkageNetworkBuilder(cfg)
    print("[读取与识别列]", cfg.input_path)
    df, colmap = builder.load_and_prepare()
    print("[构建收益矩阵 R]")
    R = builder.prepare_returns_matrix(df, colmap)
    del df
    gc.collect()
    print(f"[样本维度] weeks={R.shape[0]} stocks={R.shape[1]}")
    print("[计算三视角 & 融合 & 排名 & 社区]")
    results = builder.build_all(R)
    print("[写出结果]")
    builder.save_outputs(results)


if __name__ == '__main__':
    main_cli()
