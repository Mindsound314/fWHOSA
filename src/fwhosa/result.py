"""The single result type every estimator returns."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HOSResult:
    """Estimates on a triad plan, with validity flags and full provenance.

    ``values`` has shape ``(n_windows, n_triads)``: one row per averaging window
    (a single row when ``expectation="all"``) and one column per triad
    ``(f1[t], f2[t], f3[t] = f1[t] + f2[t])``. Values are never overwritten by
    masks or by inference; ``valid`` says which ones to trust and ``flags``
    says why the others were flagged (True = problem).
    """

    kind: str
    values: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray
    times: np.ndarray
    valid: np.ndarray
    flags: dict
    dof: np.ndarray
    pair_freqs: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_triads(self) -> int:
        return int(self.values.shape[1])

    @property
    def is_cross(self) -> bool:
        return self.kind.startswith("cross")

    def _pair_index(self, f: np.ndarray) -> np.ndarray:
        df = float(self.params.get("df", 1.0))
        keys = np.round(self.pair_freqs / df).astype(np.int64)
        q = np.round(np.asarray(f) / df).astype(np.int64)
        return np.searchsorted(keys, q)

    def matrix(self, window: int = 0, masked: bool = False) -> np.ndarray:
        """Square map ``M[a, b]`` for ``f1 = pair_freqs[a]``, ``f2 = pair_freqs[b]``; NaN off-plan.

        Auto maps are mirrored across the diagonal. With ``masked=True`` invalid
        cells are NaN as well.
        """
        n = self.pair_freqs.size
        dtype = np.complex128 if np.iscomplexobj(self.values) else np.float64
        M = np.full((n, n), np.nan, dtype=dtype)
        v = np.asarray(self.values[window], dtype=dtype).copy()
        if masked:
            v[~self.valid[window]] = np.nan
        a = self._pair_index(self.f1)
        b = self._pair_index(self.f2)
        M[a, b] = v
        if not self.is_cross:
            M[b, a] = v
        return M

    def plot(self, window: int = 0, ax=None, masked: bool = True, **imshow_kw):
        """Plot one window with f1 on the horizontal axis and f2 on the vertical axis."""
        import matplotlib.pyplot as plt

        M = self.matrix(window, masked=masked)
        if np.iscomplexobj(M):
            M = np.abs(M)
        ax = ax or plt.gca()
        lo, hi = float(self.pair_freqs[0]), float(self.pair_freqs[-1])
        half = 0.5 * float(self.params.get("df", 0.0))
        im = ax.imshow(M.T, origin="lower", aspect="auto",            # rows of M.T are f2
                       extent=[lo - half, hi + half, lo - half, hi + half], **imshow_kw)
        suffix = (" (x)", " (y)") if self.is_cross else ("", "")
        ax.set_xlabel(f"f₁{suffix[0]} (Hz)")
        ax.set_ylabel(f"f₂{suffix[1]} (Hz)")
        t = self.times[window]
        ax.set_title(self.kind + ("" if not np.isfinite(t) else f", t = {t:.2f} s"))
        plt.colorbar(im, ax=ax)
        return ax

    def save(self, path) -> None:
        arrays = dict(values=self.values, f1=self.f1, f2=self.f2, f3=self.f3, times=self.times,
                      valid=self.valid, dof=self.dof, pair_freqs=self.pair_freqs)
        for name, flag in self.flags.items():
            arrays[f"flag_{name}"] = flag
        np.savez_compressed(path, kind=np.array(self.kind),
                            params=np.array(json.dumps(self.params, default=_jsonable)), **arrays)

    @classmethod
    def load(cls, path) -> "HOSResult":
        with np.load(path, allow_pickle=False) as z:
            flags = {k[5:]: z[k] for k in z.files if k.startswith("flag_")}
            return cls(kind=str(z["kind"]), values=z["values"], f1=z["f1"], f2=z["f2"], f3=z["f3"],
                       times=z["times"], valid=z["valid"], flags=flags, dof=z["dof"],
                       pair_freqs=z["pair_freqs"], params=json.loads(str(z["params"])))


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)
