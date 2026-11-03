import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util as tu
from jax.flatten_util import ravel_pytree

__all__ = [
    "Recorder", "Result",
    "hessian_spectrum", "mlp_singularity",
    "pca3", "detect",
    "run_recorded", "reverse_recorded",
    "plot3d", "plot_metrics",
]

# ヘッセ表列のスペクトル
class _SpectrumFns:

    def __init__(self, loss_fn, params):
        _, unravel = ravel_pytree(params)
        f = lambda w, batch: loss_fn(unravel(w), batch)
        g = jax.grad(f)

        @jax.jit
        def eig_exact(w, batch):
            H = jax.hessian(f)(w, batch)
            return jnp.linalg.eigvalsh(0.5 * (H + H.T))

        self.eig_exact = eig_exact
        self.hvp = jax.jit(lambda w, v, batch: jax.jvp(lambda u: g(u, batch), (w,), (v,))[1])

    def spectrum(self, params, batch, exact_max, iters, key):
        flat = ravel_pytree(params)[0]
        if flat.size <= exact_max:
            return np.asarray(self.eig_exact(flat, batch)), True
        hvp = lambda v: self.hvp(flat, v, batch)
        return np.asarray(_lanczos(hvp, flat.size, iters, key, flat.dtype)), False

# ランチョス法
def _lanczos(hvp, n, iters, key, dtype):
    iters = min(iters, n)
    v = jax.random.normal(key, (n,), dtype)
    v = v / jnp.linalg.norm(v)
    V = [v]
    alphas, betas = [], []
    for i in range(iters):
        w = hvp(V[-1])
        a = jnp.vdot(w, V[-1])
        Q = jnp.stack(V)
        w = w - Q.T @ (Q @ w) # 再直交化(Gram-Schmidt 2回分)
        w = w - Q.T @ (Q @ w)
        b = jnp.linalg.norm(w)
        alphas.append(a)
        if i == iters - 1 or float(b) < 1e-10:
            break
        betas.append(b)
        V.append(w / b)
    T = jnp.diag(jnp.stack(alphas))
    if betas:
        off = jnp.stack(betas)
        T = T + jnp.diag(off, 1) + jnp.diag(off, -1)
    return jnp.linalg.eigvalsh(T)

def hessian_spectrum(loss_fn, params, batch, exact_max=2048, iters=40, key=None):
    key = jax.random.key(0) if key is None else key
    return _SpectrumFns(loss_fn, params).spectrum(params, batch, exact_max, iters, key)

def _spectrum_metrics(eig, exact, flat_tol):
    lam_max = float(eig[-1])
    lam_min = float(eig[0])
    scale = max(abs(lam_max), abs(lam_min), 1e-30)
    out = {
        "lam_max": lam_max,
        "lam_min": lam_min,
        "neg_ratio": abs(min(lam_min, 0.0)) / scale,
    }
    if exact:
        out["n_neg"] = int(np.sum(eig < -flat_tol * scale))
        out["n_flat"] = int(np.sum(np.abs(eig) <= flat_tol * scale))
    return out

def mlp_singularity(params):
    names = sorted((k for k in params if k.startswith("1")), key=lambda k: int(k[1:]))
    out_min, in_min, ov_min = np.inf, np.inf, np.inf
    for a, b in zip(names[:-1], names[1:]):
        W_in = np.asarray(params[a]["W"], np.float64)
        W_out = np.asarray(params[b]["W"], np.float64)
        in_norm = np.linalg.norm(W_in, axis=0)
        out_min = min(out_min, float(np.linalg.norm(W_out, axis=1).min()))
        in_min = min(in_min, float(in_norm.min()))
        if W_in.shape[1] >= 2:
            X = W_in.T
            diff = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
            summ = np.linalg.norm(X[:, None, :] + X[None, :, :], axis=-1)
            D = np.minimum(diff, summ)
            np.fill_diagonal(diff, summ)
            ov_min = min(ov_min, float(D.min() / max(in_norm.mean(), 1e-30)))
    return {"out_norm_min": out_min, "in_norm_min": in_min, "overlap_dist": ov_min}

class Recorder:

    def __init__(self, loss_fn, probe_batch, every=1, hessian_every=1, mlp=True,
                 exact_max=2048, lanczos_iters=40, flat_tol=1e-3, seed=0):
        self.loss_fn = loss_fn
        self.probe = probe_batch
        self.every = every
        self.hessian_every = hessian_every
        self.mlp = mlp
        self.exact_max = exact_max
        self.lanczos_iters = lanczos_iters
        self.flat_tol = flat_tol
        self.key = jax.random.key(seed)
        self._vg = jax.jit(jax.value_and_grad(loss_fn))
        self._spec = None
        self.steps, self.P, self.rows = [], [], []

    def __call__(self, step, params, force=False):
        if not force and step % self.every != 0:
            return
        loss, g = self._vg(params, self.probe)
        flat = np.asarray(ravel_pytree(params)[0], np.float32)
        row = {"loss": float(loss), "grad_norm": float(jnp.sqrt(sum(jnp.vdot(x, x) for x in tu.tree_leaves(g))))}
        k = len(self.steps)
        if self.hessian_every and k % self.hessian_every == 0:
            if self._spec is None:
                self._spec = _SpectrumFns(self.loss_fn, params)
            self.key, sub = jax.random.split(self.key)
            eig, exact = self._spec.spectrum(params, self.probe, self.exact_max,
                                             self.lanczos_iters, sub)
            row.update(_spectrum_metrics(eig, exact, self.flat_tol))
        if self.mlp:
            row.update(mlp_singularity(params))
        self.steps.append(int(step))
        self.P.append(flat)
        self.rows.append(row)

    def arrays(self):
        keys = sorted({k for r in self.rows for k in r})
        m = {k: np.array([r.get(k, np.nan) for r in self.rows], np.float64) for k in keys}
        return np.array(self.steps), np.stack(self.P), m

    def result(self, basis=None, **detect_kw):
        steps, P, m = self.arrays()
        Z, basis = pca3(P, basis)
        return Result(steps, P, m, Z, basis, detect(m, **detect_kw))

    def save(self, path):
        steps, P, m = self.arrays()
        np.savez_compressed(path, steps=steps, P=P, **{f"m_{k}": v for k, v in m.items()})

class Result:
    def __init__(self, steps, P, metrics, Z, basis, events):
        self.steps, self.P, self.metrics = steps, P, metrics
        self.Z, self.basis, self.events = Z, basis, events

    def summary(self):
        evr = self.basis["evr"]
        lines = [f"snapshots: {len(self.steps)}, params: {self.P.shape[1]}",
                 f"PCA explained variance (PC1-3): {evr[0]:.3f}, {evr[1]:.3f}, {evr[2]:.3f} "
                 f"(total {evr.sum():.3f})"]
        for name, idx in self.events.items():
            s = ", ".join(str(self.steps[i]) for i in idx[:10])
            more = f" ... (+{len(idx) - 10})" if len(idx) > 10 else ""
            lines.append(f"{name:14s}: {len(idx):4d} 件  steps [{s}{more}]")
        return "\n".join(lines)

def pca3(P, basis=None):
    """軌跡 P (T, N) を PCA で 3 次元に射影する。

    basis を渡すとその基底で射影する (逆向き軌跡を同じ座標系に載せるときなど)。
    """
    P = np.asarray(P, np.float64)
    if basis is None:
        mean = P.mean(axis=0)
        X = P - mean
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(3, Vt.shape[0])
        comps = np.zeros((3, P.shape[1]))
        comps[:k] = Vt[:k]
        var = S ** 2
        evr = np.zeros(3)
        evr[:k] = var[:k] / max(var.sum(), 1e-300)
        basis = {"mean": mean, "components": comps, "evr": evr}
    Z = (P - basis["mean"]) @ basis["components"].T
    return Z, basis

def _robust_spikes(x, z):
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return np.array([], int)
    d = np.full_like(x, np.nan)
    d[1:] = np.diff(x)
    dd = d[np.isfinite(d)]
    med = np.median(dd)
    mad = np.median(np.abs(dd - med)) * 1.4826 + 1e-12
    return np.where(np.abs(d - med) / mad > z)[0]

def _stable_changes(x, persist):
    """整数値の系列 x で、新しい値が persist 回以上続いた変化点だけを返す。

    ゼロ付近の固有値が閾値をまたいで行き来するだけの揺らぎを除くため。
    """
    x = np.asarray(x, np.float64)
    idx = np.where(np.isfinite(x))[0]
    if len(idx) == 0:
        return np.array([], int)
    v = x[idx]
    out, cur = [], v[0]
    i = 1
    while i < len(v):
        if v[i] != cur:
            run = v[i:i + persist]
            if len(run) == persist and np.all(run == v[i]):
                out.append(idx[i])
                cur = v[i]
                i += persist
                continue
        i += 1
    return np.array(out, int)

def _level_shifts(x, window, delta):
    """ばらつく整数系列 x を移動中央値で均し、前回の変化点の水準から
    delta 以上ずれた点を返す (Morse 指数や退化次元の「水準」の変化を見るため)。"""
    x = np.asarray(x, np.float64)
    idx = np.where(np.isfinite(x))[0]
    if len(idx) < window:
        return np.array([], int)
    v = x[idx]
    h = window // 2
    med = np.array([np.median(v[max(0, i - h):i + h + 1]) for i in range(len(v))])
    out, level = [], med[0]
    for i in range(1, len(med)):
        if abs(med[i] - level) >= delta:
            out.append(idx[i])
            level = med[i]
    return np.array(out, int)

def detect(m, z=6.0, persist=3, window=11, delta=3, overlap_tol=0.05, norm_tol=1e-2, neg_tol=1e-3):
    """指標の時系列から特異点の候補となる記録点のインデックスを返す。

    - saddle       : 負曲率 (lam_min < 0) が現れた/消えた点 (persist 回以上続いた変化のみ)
    - morse_change : 負の固有値の数 (Morse 指数) の水準が delta 以上変わった点 [厳密モードのみ]
    - degeneracy   : 退化方向の数 (n_flat) の水準が delta 以上変わった点 [厳密モードのみ]
    - curv_spike   : log(lam_max) の急変
    - grad_spike   : log(grad_norm) の急変
    - unit_vanish  : 隠れユニットの出力側/入力側重みがほぼ 0
    - unit_overlap : 2 つの隠れユニットの入力側重みがほぼ一致 (または符号反転で一致)

    morse_change / degeneracy は、window 点の移動中央値で均してから判定する
    (ゼロ付近の固有値が閾値をまたぐだけの揺らぎを拾わないため)。
    """
    ev = {}
    if "neg_ratio" in m:
        neg = np.where(np.isfinite(m["neg_ratio"]), (m["neg_ratio"] > neg_tol).astype(float), np.nan)
        ev["saddle"] = _stable_changes(neg, persist)
    if "n_neg" in m:
        ev["morse_change"] = _level_shifts(m["n_neg"], window, delta)
    if "n_flat" in m:
        ev["degeneracy"] = _level_shifts(m["n_flat"], window, delta)
    if "lam_max" in m:
        ev["curv_spike"] = _robust_spikes(np.log(np.abs(m["lam_max"]) + 1e-30), z)
    if "grad_norm" in m:
        ev["grad_spike"] = _robust_spikes(np.log(m["grad_norm"] + 1e-30), z)
    if "out_norm_min" in m:
        ev["unit_vanish"] = np.where((m["out_norm_min"] < norm_tol) | (m["in_norm_min"] < norm_tol))[0]
    if "overlap_dist" in m:
        ev["unit_overlap"] = np.where(m["overlap_dist"] < overlap_tol)[0]
    return {k: np.asarray(v, int) for k, v in ev.items()}

def _chunks(T, every):
    b = list(range(0, T, every)) + [T]
    return list(zip(b[:-1], b[1:]))


def run_recorded(opt, state, xs, data, rec, step0=0):
    """opt.run を rec.every ステップずつに分けて実行し、区切りごとに記録する。

    分けて呼んでも opt のログは区間の和で管理されるので、
    あとで opt.reverse(state, xs, data) をまとめて呼んでも正しく戻る。
    """
    if step0 == 0:
        rec(0, opt.params(state), force=True)
    T = tu.tree_leaves(xs)[0].shape[0]
    for s, e in _chunks(T, rec.every):
        state = opt.run(state, tu.tree_map(lambda a: a[s:e], xs), data)
        rec(step0 + e, opt.params(state), force=True)
    return state


def reverse_recorded(opt, state, xs, data, rec, step0):
    """reverse を run_recorded と同じ区切りで逆順に実行し、区切りごとに記録する。

    step0 は、このエポックの開始ステップ (run_recorded に渡した値と同じ)。
    """
    T = tu.tree_leaves(xs)[0].shape[0]
    for s, e in reversed(_chunks(T, rec.every)):
        state = opt.reverse(state, tu.tree_map(lambda a: a[s:e], xs), data)
        rec(step0 + s, opt.params(state), force=True)
    return state

_EVENT_STYLE = {
    "saddle":       ("diamond", "#d62728"),
    "morse_change": ("diamond-open", "#ff7f0e"),
    "degeneracy":   ("square-open", "#9467bd"),
    "curv_spike":   ("x", "#8c564b"),
    "grad_spike":   ("cross", "#e377c2"),
    "unit_vanish":  ("circle-open", "#17becf"),
    "unit_overlap": ("square", "#bcbd22"),
}

def plot3d(res, path="trajectory.html", color="loss", reverse=None, log_color=True, title=None):
    """3D 軌跡を HTML に書き出す。

    color   : 軌跡の色に使う指標名 ("loss", "lam_min", "grad_norm", "step" など)
    reverse : 逆向きの Result。forward と同じ基底で射影して重ねる
    """
    import plotly.graph_objects as go

    Z = res.Z
    if color == "step":
        c, cname = res.steps.astype(float), "step"
    else:
        c = res.metrics[color].copy()
        cname = color
        if log_color and np.nanmin(c) > 0:
            c, cname = np.log10(c), f"log10 {color}"
    hover = [f"step {s}<br>" + "<br>".join(f"{k}={res.metrics[k][i]:.4g}" for k in ("loss", "grad_norm", "lam_min", "lam_max")
                                           if k in res.metrics and np.isfinite(res.metrics[k][i]))
             for i, s in enumerate(res.steps)]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=Z[:, 0], y=Z[:, 1], z=Z[:, 2], mode="lines+markers", name="forward",
        line=dict(color=c, colorscale="Viridis", width=4),
        marker=dict(size=2.5, color=c, colorscale="Viridis",
                    colorbar=dict(title=cname, x=0.0, xanchor="right", len=0.8)),
        text=hover, hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=Z[:1, 0], y=Z[:1, 1], z=Z[:1, 2], mode="markers", name="start",
                               marker=dict(size=7, color="black", symbol="circle")))
    fig.add_trace(go.Scatter3d(x=Z[-1:, 0], y=Z[-1:, 1], z=Z[-1:, 2], mode="markers", name="end",
                               marker=dict(size=7, color="black", symbol="diamond")))

    for name, idx in res.events.items():
        if len(idx) == 0:
            continue
        sym, col = _EVENT_STYLE.get(name, ("circle", "red"))
        fig.add_trace(go.Scatter3d(
            x=Z[idx, 0], y=Z[idx, 1], z=Z[idx, 2], mode="markers", name=f"{name} ({len(idx)})",
            marker=dict(size=6, symbol=sym, color=col),
            text=[hover[i] for i in idx], hoverinfo="text+name"))

    if reverse is not None:
        Zr, _ = pca3(reverse.P, res.basis)
        fig.add_trace(go.Scatter3d(x=Zr[:, 0], y=Zr[:, 1], z=Zr[:, 2], mode="lines", name="reverse",
                                   line=dict(color="orange", width=2, dash="dash")))

    evr = res.basis["evr"]
    fig.update_layout(
        title=title or f"parameter trajectory (PCA, explained {evr.sum():.1%})",
        scene=dict(xaxis_title=f"PC1 ({evr[0]:.1%})", yaxis_title=f"PC2 ({evr[1]:.1%})",
                   zaxis_title=f"PC3 ({evr[2]:.1%})"),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(path, include_plotlyjs="cdn")
    return fig


def plot_metrics(res, path="metrics.html"):
    """指標の時系列を HTML に書き出す。検出イベントは縦線で示す。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    m, x = res.metrics, res.steps
    panels = [
        ("loss / grad_norm", [("loss", "log"), ("grad_norm", "log")]),
        ("Hessian extremes", [("lam_max", None), ("lam_min", None)]),
        ("Morse index / flat dims", [("n_neg", None), ("n_flat", None)]),
        ("MLP singularity (0 に近いほど特異点に近い)",
         [("out_norm_min", "log"), ("in_norm_min", "log"), ("overlap_dist", "log")]),
    ]
    panels = [(t, [s for s in ss if s[0] in m]) for t, ss in panels]
    panels = [p for p in panels if p[1]]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True,
                        subplot_titles=[p[0] for p in panels], vertical_spacing=0.06)
    for r, (_, series) in enumerate(panels, start=1):
        for name, scale in series:
            fig.add_trace(go.Scatter(x=x, y=m[name], mode="lines", name=name), row=r, col=1)
            if scale == "log":
                fig.update_yaxes(type="log", row=r, col=1)
    # add_vline を 1 本ずつ呼ぶと非常に遅いので、shapes をまとめて渡す
    shapes = []
    for name, idx in res.events.items():
        col = _EVENT_STYLE.get(name, ("", "red"))[1]
        for i in idx[:300]:
            shapes.append(dict(type="line", xref="x", yref="paper", x0=float(x[i]), x1=float(x[i]),
                               y0=0, y1=1, line=dict(color=col, width=1, dash="dot"), opacity=0.5))
    fig.update_layout(shapes=shapes, height=260 * len(panels), margin=dict(t=40))
    fig.update_xaxes(title_text="step", row=len(panels), col=1)
    fig.write_html(path, include_plotlyjs="cdn")
    return fig