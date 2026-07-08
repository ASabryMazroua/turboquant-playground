"""Retrieval chapter — does TurboQuant (rotate-before-quantize) + the QJL sign
sketch help *retrieval*, the regime Finding 7 said is QJL's real home? Now across
**multiple datasets**, scored on the three axes that decide an ANN index:

* **quality** — recall@10 vs an exact ``IndexFlatIP`` ground truth (raw code +
  after a 2-stage exact rerank), and
* **latency** — queries/sec of the exhaustive scan (smaller codes ⇒ less memory
  traffic ⇒ faster), and
* **RAM** — bytes/vector (the code footprint).

Finding 7 (M16) showed the QJL unbiased inner-product estimator is fatal for
**softmax attention** (unbiased but high-variance; attention over thousands of
keys is exquisitely sensitive to per-logit variance). Retrieval is different:
top-k + a cheap exact **rerank** only needs the true neighbours to land in the
candidate pool, so it tolerates variance. This benchmark tests that across four
dataset regimes.

Datasets (synthetic, reproducible, no downloads; ``mnist`` optional/real):

* ``aniso`` — low-rank + heavy-tailed **outlier coordinates** (the KV structure).
* ``iso``   — isotropic Gaussian (rotation should not help — a control).
* ``blobs`` — clustered / low intrinsic dimension (embedding-like).
* ``unit``  — ``aniso`` normalised to the unit sphere (cosine retrieval).

Methods (all vs exact ``IndexFlatIP``):

* ``sq8`` / ``sq4`` — FAISS per-dimension scalar quantizers.
* ``rrot+sq4`` — random-rotation ``IndexPreTransform`` then SQ4: TurboQuant's
  rotate-before-quantize running *inside* FAISS (no C++ fork).
* ``pq`` / ``opq+pq`` — product quantization, and OPQ (a **learned** rotation
  before PQ) — the field's canonical rotate-before-quantize win.
* ``qjl-m`` — the *exact* estimator from ``turbo_kv/qjl.py`` (sign(S·x) + ‖x‖).
* ``simhash-m`` — the SAME sign bits, decoded by Hamming agreement (sign-LSH):
  the control that isolates QJL's unbiased-IP decoder from the raw sign code.

CPU-only. Run::

    python retrieval/bench_faiss_turbo.py --datasets aniso,iso,blobs,unit \
        --n-db 100000 --n-q 1000 --dim 128 --k 10 --rerank 100 \
        --qjl-ms 512,1024 --out-dir results
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbo_kv import reporting  # noqa: E402


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def _aniso(n, d, rng):
    rank = max(4, d // 4)
    W = rng.standard_normal((rank, d)).astype("float32")
    x = rng.standard_normal((n, rank)).astype("float32") @ W
    od = rng.choice(d, size=min(8, d), replace=False)
    x[:, od] += rng.standard_normal((n, od.size)).astype("float32") * 6.0
    x += 0.1 * rng.standard_normal(x.shape).astype("float32")
    return x


def _iso(n, d, rng):
    return rng.standard_normal((n, d)).astype("float32")


def _blobs(n, d, rng, n_clusters=64):
    centers = rng.standard_normal((n_clusters, d)).astype("float32") * 4.0
    lab = rng.integers(0, n_clusters, size=n)
    return centers[lab] + rng.standard_normal((n, d)).astype("float32")


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)


def load_dataset(name, n_db, n_q, d, seed):
    """Return (xb [n_db,d], xq [n_q,d]) float32 for a named dataset."""
    rng = np.random.default_rng(seed)
    n = n_db + n_q
    if name == "aniso":
        x = _aniso(n, d, rng)
    elif name == "iso":
        x = _iso(n, d, rng)
    elif name == "blobs":
        x = _blobs(n, d, rng)
    elif name == "unit":
        x = _unit(_aniso(n, d, rng))
    elif name == "mnist":
        return _load_mnist(n_db, n_q, seed)
    else:
        raise ValueError(f"unknown dataset: {name}")
    x = np.ascontiguousarray(x.astype("float32"))
    return x[:n_db], x[n_db:]


def _load_mnist(n_db, n_q, seed):
    """Optional real dataset: MNIST-784 via sklearn (downloads ~15MB once)."""
    from sklearn.datasets import fetch_openml

    X = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto").data
    X = np.ascontiguousarray(X.astype("float32"))
    rng = np.random.default_rng(seed)
    rng.shuffle(X)
    n_db = min(n_db, X.shape[0] - n_q)
    return X[:n_db], X[n_db:n_db + n_q]


# --------------------------------------------------------------------------- #
# quality helpers
# --------------------------------------------------------------------------- #
def recall_at_k(gt, got, k):
    hits = 0
    for i in range(gt.shape[0]):
        hits += len(set(gt[i, :k].tolist()) & set(got[i, :k].tolist()))
    return hits / (gt.shape[0] * k)


def rerank_exact(xb, xq, cand, k):
    out = np.full((xq.shape[0], k), -1, dtype=np.int64)
    for i in range(xq.shape[0]):
        c = cand[i][cand[i] >= 0]
        if c.size == 0:
            continue
        top = c[np.argsort(-(xb[c] @ xq[i]))[:k]]
        out[i, :top.size] = top
    return out


# --------------------------------------------------------------------------- #
# searchers — FAISS indexes and the QJL / SimHash sign sketches
# --------------------------------------------------------------------------- #
def _pq_m(d):
    target = max(1, d // 8)
    cands = [m for m in range(1, min(64, d) + 1) if d % m == 0]
    return min(cands, key=lambda m: abs(m - target))


class _FaissSearcher:
    def __init__(self, index, bytes_per_vec):
        self.index, self.bytes_per_vec = index, bytes_per_vec

    def search(self, xq, R):
        return self.index.search(xq, R)[1]


def build_faiss(kind, xb, seed):
    import faiss

    d = xb.shape[1]
    IP = faiss.METRIC_INNER_PRODUCT
    SQ = faiss.ScalarQuantizer
    t0 = time.perf_counter()
    if kind == "exact":
        index, bpv = faiss.IndexFlatIP(d), 4.0 * d
    elif kind == "sq8":
        index, bpv = faiss.IndexScalarQuantizer(d, SQ.QT_8bit, IP), 1.0 * d
    elif kind == "sq4":
        index, bpv = faiss.IndexScalarQuantizer(d, SQ.QT_4bit, IP), 0.5 * d
    elif kind == "rrot+sq4":
        rr = faiss.RandomRotationMatrix(d, d)
        rr.init(seed)
        sq = faiss.IndexScalarQuantizer(d, SQ.QT_4bit, IP)
        index, bpv = faiss.IndexPreTransform(rr, sq), 0.5 * d
    elif kind == "pq":
        m = _pq_m(d)
        index, bpv = faiss.IndexPQ(d, m, 8, IP), float(m)
    elif kind == "opq+pq":
        m = _pq_m(d)
        opq = faiss.OPQMatrix(d, m)
        index, bpv = faiss.IndexPreTransform(opq, faiss.IndexPQ(d, m, 8, IP)), float(m)
    else:
        raise ValueError(kind)
    if not index.is_trained:
        index.train(xb)
    index.add(xb)
    return _FaissSearcher(index, bpv), time.perf_counter() - t0


class _SketchSearcher:
    """QJL (unbiased IP) or SimHash (Hamming) over a shared sign sketch."""

    def __init__(self, mode, sk, signs, norm, bytes_per_vec):
        import torch

        self.mode, self.sk, self.signs, self.norm = mode, sk, signs, norm
        self.bytes_per_vec = bytes_per_vec
        if mode == "simhash":
            self.S = sk._ensure(signs.device, torch.float32)
            self.db_pm = signs.to(torch.float32) * 2.0 - 1.0

    def search(self, xq, R):
        import torch

        tq = torch.from_numpy(xq)
        R = min(R, self.signs.shape[0])
        out = np.empty((xq.shape[0], R), dtype=np.int64)
        bs = 256
        for s in range(0, xq.shape[0], bs):
            if self.mode == "qjl":
                est = self.sk.estimate_matrix(tq[s:s + bs], self.signs, self.norm)
            else:
                q_pm = ((tq[s:s + bs] @ self.S.t()) >= 0).to(torch.float32) * 2.0 - 1.0
                est = q_pm @ self.db_pm.t()
            out[s:s + bs] = torch.topk(est, R, dim=1).indices.cpu().numpy()
        return out


def build_sketch(mode, m, xb, seed):
    import torch

    from turbo_kv.qjl import QJLSketch

    t0 = time.perf_counter()
    sk = QJLSketch(xb.shape[1], m=m, seed=seed)
    signs, norm = sk.sketch(torch.from_numpy(xb))       # signs = sign(S·xb)
    bpv = m / 8.0 + (2.0 if mode == "qjl" else 0.0)     # + fp16 ‖x‖ for QJL only
    return _SketchSearcher(mode, sk, signs, norm, bpv), time.perf_counter() - t0


def timed_search(searcher, xq, R, reps=3):
    searcher.search(xq[:min(64, xq.shape[0])], R)        # warmup
    best = float("inf")
    cand = None
    for _ in range(reps):
        t0 = time.perf_counter()
        cand = searcher.search(xq, R)
        best = min(best, time.perf_counter() - t0)
    best = max(best, 1e-9)
    return cand, best * 1000.0 / xq.shape[0], xq.shape[0] / best   # ms/query, qps


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="aniso,iso,blobs,unit")
    ap.add_argument("--n-db", type=int, default=100_000)
    ap.add_argument("--n-q", type=int, default=1000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank", type=int, default=100)
    ap.add_argument("--qjl-ms", default="512,1024")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    import faiss

    k, R = args.k, args.rerank
    faiss_kinds = ["exact", "sq8", "sq4", "rrot+sq4", "pq", "opq+pq"]
    ms = [int(x) for x in args.qjl_ms.split(",")]
    sketch_specs = [("qjl", m) for m in ms] + [("simhash", m) for m in ms]
    out_csv = os.path.join(args.out_dir, "retrieval_recall.csv")
    rows = []

    for ds in [d.strip() for d in args.datasets.split(",")]:
        try:
            xb, xq = load_dataset(ds, args.n_db, args.n_q, args.dim, args.seed)
        except Exception as e:  # optional/real datasets may be unavailable
            print(f"[skip dataset {ds}] {type(e).__name__}: {e}")
            continue
        n_db, d = xb.shape
        gt_index = faiss.IndexFlatIP(d)
        gt_index.add(xb)
        gt = gt_index.search(xq, k)[1]
        print(f"\n=== {ds}: db={xb.shape} q={xq.shape}  (exact = {4*d} B/vec) ===")

        def evaluate(name, build_fn):
            try:
                searcher, enc_s = build_fn()
            except Exception as e:
                print(f"  {name:12s} [skip] {type(e).__name__}: {e}")
                return
            cand, ms_q, qps = timed_search(searcher, xq, R)
            raw = recall_at_k(gt, cand[:, :k], k)
            t0 = time.perf_counter()
            rr_cand = rerank_exact(xb, xq, cand, k)
            rr_ms = (time.perf_counter() - t0) * 1000.0 / xq.shape[0]
            rr = recall_at_k(gt, rr_cand, k)
            bpv = searcher.bytes_per_vec
            rows.append(dict(
                dataset=ds, method=name, bytes_per_vec=round(bpv, 1),
                index_mb=round(bpv * n_db / 1e6, 1),
                compression=round(4.0 * d / bpv, 1),
                recall_at_k=round(raw, 4), recall_rerank=round(rr, 4),
                ms_per_query=round(ms_q, 4), qps=round(qps, 1),
                rerank_ms_per_query=round(rr_ms, 4), encode_s=round(enc_s, 2)))
            print(f"  {name:12s} recall@{k} {rr:.3f} (raw {raw:.3f}) · "
                  f"{ms_q:6.3f} ms/q ({qps:7.0f} qps) · {bpv:6.1f} B/vec "
                  f"({bpv * n_db / 1e6:5.1f} MB)")

        for kind in faiss_kinds:
            evaluate(kind, lambda kind=kind: build_faiss(kind, xb, args.seed))
        for mode, m in sketch_specs:
            evaluate(f"{mode}-m{m}", lambda mode=mode, m=m: build_sketch(mode, m, xb, args.seed))

    for r in rows:
        reporting.append_row(out_csv, r)
    print(f"\nwrote {out_csv}")
    _plot(rows, k)
    return 0


def _plot(rows, k) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    datasets = sorted({r["dataset"] for r in rows})
    n = len(datasets)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.4 * nrow), squeeze=False)
    for ax, ds in zip(axes.flat, datasets):
        dr = [r for r in rows if r["dataset"] == ds]
        qmax = max(r["qps"] for r in dr) or 1.0
        for r in dr:
            is_ours = r["method"].startswith("qjl")
            is_sign = r["method"].startswith(("qjl", "simhash"))
            size = 40 + 260 * (r["qps"] / qmax)          # marker size ∝ latency (qps)
            ax.scatter(r["bytes_per_vec"], r["recall_rerank"], s=size,
                       marker=("o" if is_sign else "s"),
                       edgecolor="k", linewidth=0.5,
                       color=("#1f77b4" if is_ours else "#ff7f0e" if r["method"].startswith("simhash") else "#7f7f7f"),
                       alpha=0.8)
            ax.annotate(r["method"], (r["bytes_per_vec"], r["recall_rerank"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("RAM: bytes / vector (log)")
        ax.set_ylabel(f"quality: recall@{k} (reranked)")
        ax.set_title(f"{ds}  (marker size ∝ latency/QPS)")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Retrieval tradeoff: quality vs RAM vs latency — QJL (blue) vs SimHash (orange) vs FAISS (grey)")
    reporting.save_fig(fig, "retrieval_tradeoff")


if __name__ == "__main__":
    raise SystemExit(main())
