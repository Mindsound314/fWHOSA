"""Transform -> row scales -> block moments -> sums over the averaging domain.

Four terms keep four jobs apart:

* segments (``cores``) manage memory and CWT context; a core gets guard samples
  on both sides so its coefficients equal the whole-recording ones;
* blocks of ``gcd(window, step)`` samples are the accumulation units;
* windows are sums of consecutive blocks (temporal smoothing);
* trials are summed sample by sample (pooled weighting) before any ratio.

Row scales are the float64 RMS of each frequency row over all analysed samples
of a channel, shared across trials, so every pooled moment is in common units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import get_block_moments, row_powers
from .plan import TriadPlan, WindowPlan
from .transform import Wavelet, cwt, guard_sd


@dataclass
class Sums:
    S: np.ndarray            # (n_win, T) complex128, row-scaled units
    Q12: np.ndarray          # (n_win, T)
    P1: np.ndarray           # (n_win, T) power of W1 at f1, row-scaled units
    P2: np.ndarray
    P3: np.ndarray
    n: np.ndarray            # (n_win, T) weight totals (samples)
    scale_prod: np.ndarray   # (T,) r1[f1] * r2[f2] * r3[f3]
    starts: np.ndarray       # (n_win,) window start sample, -1 for "all"
    boundary: np.ndarray     # (n_win, T) True where the window touches an edge guard
    duration_s: np.ndarray   # (T,) analysed seconds behind each estimate
    info: dict


def _unique_channels(signals):
    uniq, slots = [], []
    for s in signals:
        for idx, u in enumerate(uniq):
            if s is u or (s.shape == u.shape and np.array_equal(s, u)):
                slots.append(idx)
                break
        else:
            uniq.append(s)
            slots.append(len(uniq) - 1)
    return uniq, slots


def _window_sums(arr, starts_b, per_window):
    """Sum ``per_window`` consecutive block columns starting at each of ``starts_b``."""
    out = arr[:, starts_b].copy()
    for m in range(1, per_window):
        out += arr[:, starts_b + m]
    return out


def run(signals, fs: float, plan: TriadPlan, wavelet: Wavelet, expectation: str, wplan: WindowPlan,
        backend: str, memory_limit: int, nthreads: int, guard_tol: float,
        transform: str = "exact") -> Sums:
    uniq, slots = _unique_channels(signals)
    if len(slots) == 1:
        slots = slots * 3
    n_trials, N = signals[0].shape
    block = wplan.block
    n_blocks = N // block
    if n_blocks == 0:
        raise ValueError("signal is shorter than one block")
    n_win = 1 if expectation == "all" else wplan.n_windows(n_blocks)
    if n_win == 0:
        raise ValueError("signal is shorter than one window")

    k_guard = guard_sd(guard_tol)
    freqs = plan.grid.freqs
    n_f, T, n_u = plan.grid.n, plan.n, len(uniq)
    f_low = np.minimum(plan.f1, plan.f2)
    g_t = np.ceil(k_guard * wavelet.fb / f_low * fs).astype(np.int64)        # per-triad edge guard
    g_seg = int(math.ceil(k_guard * wavelet.fb / freqs[0] * fs))             # segment context

    # ---- memory plan --------------------------------------------------------
    coef_b = n_f * 8 * n_u                     # bytes per sample of coefficients
    mom_b = T * 24 + n_f * 8 * n_u             # bytes per block of moments
    if N * coef_b + n_blocks * mom_b <= memory_limit:
        core_blocks = n_blocks
    else:
        avail = memory_limit - 2 * g_seg * coef_b
        core_blocks = int(avail // (block * coef_b + mom_b)) if avail > 0 else 0
        if core_blocks < 1:
            raise MemoryError("memory_limit is too small for one block plus its guard context")
    segmented = core_blocks < n_blocks
    if segmented and transform == "fcwt":
        import warnings

        warnings.warn("transform='fcwt' with segmented processing: fCWT's coefficients depend on the "
                      "transformed length by up to ~0.6%, so segment boundaries leave small steps; "
                      "use transform='exact' or a larger memory_limit", stacklevel=4)
    cores = [(b0, min(b0 + core_blocks, n_blocks)) for b0 in range(0, n_blocks, core_blocks)]
    cache = {} if (not segmented and n_trials * N * coef_b <= memory_limit) else None

    def coefs(tr, u, b0, b1):
        if cache is not None and (tr, u) in cache:
            return cache[(tr, u)][:, b0 * block:b1 * block]
        s0, s1 = b0 * block, b1 * block
        a, e = (max(0, s0 - g_seg), min(N, s1 + g_seg)) if segmented else (0, N)
        W = cwt(uniq[u][tr, a:e], fs, plan.grid, wavelet, nthreads, method=transform, pad=g_seg)
        if cache is not None:
            cache[(tr, u)] = W
            return W[:, s0:s1]
        return W[:, s0 - a:s1 - a]

    # ---- pass 1: row scales (recording-wide RMS per channel, shared by trials)
    sumsq = [np.zeros(n_f) for _ in uniq]
    for tr in range(n_trials):
        for b0, b1 in cores:
            for u in range(n_u):
                W = coefs(tr, u, b0, b1)
                sumsq[u] += (W.real.astype(np.float64) ** 2 + W.imag.astype(np.float64) ** 2).sum(axis=1)
    count = n_trials * n_blocks * block
    rms = []
    for s in sumsq:
        r = np.sqrt(s / count)
        rms.append(np.where(r > 0, r, 1.0))

    # ---- pass 2: block moments and reductions -------------------------------
    engine = get_block_moments(backend)
    I, J, K = plan.i, plan.j, plan.k
    out_S = np.zeros((n_win, T), np.complex128)
    out_Q = np.zeros((n_win, T))
    out_P = [np.zeros((n_win, n_f)) for _ in uniq] if expectation != "all" else None
    tri_P = [np.zeros((n_win, T)) for _ in range(3)] if expectation == "all" else None
    n_all = np.zeros(T) if expectation == "all" else None
    acc = None
    if expectation == "trials":
        acc = dict(S=np.zeros((T, n_blocks), np.complex128), Q=np.zeros((T, n_blocks)),
                   P=[np.zeros((n_f, n_blocks)) for _ in uniq])
    buf = dict(S=None, Q=None, P=None, first=0, next_w=0)

    for tr in range(n_trials):
        for b0, b1 in cores:
            Ws = [np.ascontiguousarray(coefs(tr, u, b0, b1) / rms[u][:, None].astype(np.float32),
                                       dtype=np.complex64) for u in range(n_u)]
            W1, W2, W3 = Ws[slots[0]], Ws[slots[1]], Ws[slots[2]]
            nb = b1 - b0
            S, Q = engine(W1, W2, W3, I, J, K, block, nb)
            Pu = [row_powers(Wu, block, nb) for Wu in Ws]

            if expectation == "trials":
                acc["S"][:, b0:b1] += S
                acc["Q"][:, b0:b1] += Q
                for u in range(n_u):
                    acc["P"][u][:, b0:b1] += Pu[u]
            elif expectation == "all":
                s_b = (b0 + np.arange(nb)) * block
                ok = (s_b[None, :] >= g_t[:, None]) & (s_b[None, :] + block <= N - g_t[:, None])
                out_S[0] += (S * ok).sum(axis=1)
                out_Q[0] += (Q * ok).sum(axis=1)
                for m, rows in enumerate((I, J, K)):
                    tri_P[m][0] += (Pu[slots[m]][rows] * ok).sum(axis=1)
                n_all += ok.sum(axis=1) * block
            else:  # "time": stream windows, carrying the blocks later windows still need
                if buf["S"] is None:
                    buf["S"], buf["Q"], buf["P"] = S, Q, Pu
                else:
                    buf["S"] = np.concatenate([buf["S"], S], axis=1)
                    buf["Q"] = np.concatenate([buf["Q"], Q], axis=1)
                    buf["P"] = [np.concatenate([p, q], axis=1) for p, q in zip(buf["P"], Pu)]
                last = min((b1 - wplan.per_window) // wplan.stride, n_win - 1) if b1 >= wplan.per_window else -1
                if last >= buf["next_w"]:
                    ws = np.arange(buf["next_w"], last + 1)
                    sb = ws * wplan.stride - buf["first"]
                    out_S[ws] = _window_sums(buf["S"], sb, wplan.per_window).T
                    out_Q[ws] = _window_sums(buf["Q"], sb, wplan.per_window).T
                    for u in range(n_u):
                        out_P[u][ws] = _window_sums(buf["P"][u], sb, wplan.per_window).T
                    buf["next_w"] = last + 1
                drop = buf["next_w"] * wplan.stride - buf["first"]
                if drop > 0:
                    buf["S"], buf["Q"] = buf["S"][:, drop:], buf["Q"][:, drop:]
                    buf["P"] = [p[:, drop:] for p in buf["P"]]
                    buf["first"] += drop

    if expectation == "trials":
        sb = np.arange(n_win) * wplan.stride
        out_S[:] = _window_sums(acc["S"], sb, wplan.per_window).T
        out_Q[:] = _window_sums(acc["Q"], sb, wplan.per_window).T
        for u in range(n_u):
            out_P[u][:] = _window_sums(acc["P"][u], sb, wplan.per_window).T

    # ---- assemble -----------------------------------------------------------
    if expectation == "all":
        P1, P2, P3 = tri_P
        n = n_all[None, :].copy()
        starts = np.array([-1])
        boundary = (n_all == 0)[None, :]
        duration = n_all / fs
    else:
        P1 = out_P[slots[0]][:, I]
        P2 = out_P[slots[1]][:, J]
        P3 = out_P[slots[2]][:, K]
        n = np.full((n_win, T), float(wplan.window_len * n_trials))
        starts = np.arange(n_win) * wplan.step_len
        stops = starts + wplan.window_len
        boundary = (starts[:, None] < g_t[None, :]) | (stops[:, None] > N - g_t[None, :])
        duration = np.full(T, wplan.window_len * n_trials / fs)

    scale_prod = rms[slots[0]][I] * rms[slots[1]][J] * rms[slots[2]][K]
    info = dict(transform=transform, segmented=segmented, core_blocks=int(core_blocks),
                segment_guard_samples=g_seg,
                guard_sd=float(k_guard), n_trials=int(n_trials), n_samples=int(N), block=int(block),
                n_unique_channels=n_u, row_scales=[r.tolist() for r in rms], slots=list(slots))
    return Sums(S=out_S, Q12=out_Q, P1=P1, P2=P2, P3=P3, n=n, scale_prod=scale_prod, starts=starts,
                boundary=boundary, duration_s=duration, info=info)
