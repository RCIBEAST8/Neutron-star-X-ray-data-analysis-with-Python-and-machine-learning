# Aql X-1 X-ray analysis
# can the spectrum of a QUIET window predict an outburst that's about to start?

# code breakdown
# only uses windows with no active bins at all, so nothing in the input already has the outburst in it.
# label = 1 if an outburst onset (start of an active episode that reaches >= 0.1 normalised flux) lands within
# HORIZON days after the window ends. same Flax MLP, same JAX training loop and same split by time as training.py.
# compared with a simple baseline, the mean flux over the last 8 bins (4 days) of the window, which would catch
# a slow rise before the outburst if there is one

# libs
import json

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import pandas as pd

matplotlib.use("Agg")   # no plot window, just saves the png
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402
from neural_network import SpectrumMLP, predict_proba  # noqa: E402
from training import BATCH, LR, WEIGHT_DECAY, weighted_xent  # noqa: E402
from testing_and_plots import metrics  # noqa: E402

HORIZON = 14.0          # how many days ahead it tries to predict
EPOCHS = 200
MIN_PEAK = 0.1          # normalised flux an episode has to reach to count as an outburst


# start time of every outburst. active bins get grouped into episodes (gaps up to 4 bins bridged),
# kept if at least 6 bins long and peaking above MIN_PEAK
def onsets(lc):
    a = lc.active.fillna(0).values
    t, f = lc.mjd.values, lc.flux_norm.values
    idx = np.where(a > 0)[0]
    grp = np.cumsum(np.r_[True, np.diff(idx) > 4])
    out = []
    for g in np.unique(grp):
        ii = idx[grp == g]
        if len(ii) >= 6 and np.nanmax(f[ii[0]:ii[-1] + 1]) >= MIN_PEAK:
            out.append(t[ii[0]])
    return np.array(out)


#                      forecast test                       #
def main():
    d = np.load(C.PROC / "dataset.npz")
    w = np.load(C.PROC / "windows.npz")
    lc = pd.read_csv(C.PROC / "aqlx1_clean_lightcurve_0p5d.csv")
    on = onsets(lc)
    t_end = d["t_start"] + C.WIN * C.BIN_DAYS
    quiet = w["active_frac"] == 0               # only fully quiet windows get used
    # 1 if an onset lands within HORIZON days after the window ends
    y = np.array([np.any((on > te) & (on <= te + HORIZON)) for te in t_end]).astype(np.int32)
    split = np.where(quiet, d["split"], -1)
    X = d["X_shape"]
    last = w["flux"][:, -8:].mean(1)               # last 8 bins (4 days), for the baseline

    tr, va, te = split == 0, split == 1, split == 2
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    cw = jnp.asarray(tr.sum() / (2 * np.bincount(y[tr], minlength=2)), jnp.float32)

    # same model and training setup as training.py
    model = SpectrumMLP()
    key = jax.random.PRNGKey(C.SEED)
    params = model.init(key, jnp.zeros((1, X.shape[1])))["params"]
    tx = optax.chain(optax.add_decayed_weights(WEIGHT_DECAY), optax.adam(LR))
    st = tx.init(params)

    def loss_fn(p, xb, yb, r):
        return weighted_xent(model.apply({"params": p}, xb, train=True, rngs={"dropout": r}), yb, cw)

    @jax.jit
    def step(p, s, xb, yb, r):
        l, g = jax.value_and_grad(loss_fn)(p, xb, yb, r)
        u, s = tx.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    @jax.jit
    def vloss(p, xb, yb):
        return weighted_xent(model.apply({"params": p}, xb, train=False), yb, cw)

    # train, keeping whichever epoch had the lowest val loss
    rng = np.random.default_rng(C.SEED)
    Xt, yt = Xs[tr], y[tr]
    best, hist = (np.inf, params, 0), {"train_loss": [], "val_loss": []}
    for ep in range(EPOCHS):
        perm, ls = rng.permutation(len(yt)), []
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            key, k = jax.random.split(key)
            params, st, l = step(params, st, Xt[b], yt[b], k)
            ls.append(float(l))
        vl = float(vloss(params, Xs[va], y[va]))
        hist["train_loss"].append(float(np.mean(ls))); hist["val_loss"].append(vl)
        if vl < best[0]:
            best = (vl, params, ep)
    params = best[1]
    p = np.asarray(predict_proba(model, params, jnp.asarray(Xs)))[:, 1]

    # baseline: last 4 day mean flux, turned into a smooth score
    thr_cands = np.unique(last[tr])
    bacc = [0.5 * (np.mean(last[tr][y[tr] == 1] >= c) + np.mean(last[tr][y[tr] == 0] < c)) for c in thr_cands]
    thr = thr_cands[int(np.argmax(bacc))]
    pb = 1 / (1 + np.exp(-(last - thr) / (np.std(last[tr]) * 0.05)))

    res = {"horizon_days": HORIZON, "n_onsets_total": int(len(on)),
           "n_onsets_test_period": int(np.sum(on >= C.TEST_START)),
           "counts": {k: {"n": int(m.sum()), "positive": int(y[m].sum())}
                      for k, m in (("train", tr), ("val", va), ("test", te))},
           "best_epoch": int(best[2]),
           "mlp": {"val": metrics(y[va], p[va]), "test": metrics(y[te], p[te])},
           "baseline_last4d_flux": {"val": metrics(y[va], pb[va]), "test": metrics(y[te], pb[te])}}
    (C.REPORTS / "forecast_metrics.json").write_text(json.dumps(res, indent=2))

    #                      figure                      #
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    m = lc.mjd >= C.TEST_START - 10
    axs[0].plot(lc.mjd[m], lc.flux_norm[m], ".", ms=2, color="k")
    for o in on[on >= C.TEST_START - 10]:
        for ax in axs:
            ax.axvline(o, color="#d9480f", lw=1, alpha=0.7)
    axs[0].set_ylabel("normalised flux")
    axs[0].set_title(f"Forecast test: P(outburst onset within {HORIZON:.0f} d), from fully quiet 64-day windows "
                     "(orange lines = onsets)")
    o = np.argsort(t_end[te])
    tt = t_end[te][o]
    brk = np.where(np.diff(tt) > 2 * C.STRIDE * C.BIN_DAYS)[0] + 1
    axs[1].plot(np.insert(tt, brk, np.nan), np.insert(p[te][o], brk, np.nan), "-o", ms=2.5, color="#805ad5",
                label=f"MLP on spectrum (test AUC {res['mlp']['test']['roc_auc']:.2f})")
    axs[1].plot(np.insert(tt, brk, np.nan), np.insert(pb[te][o], brk, np.nan), "-", lw=1, color="#718096",
                label=f"baseline: last-4-day flux (test AUC {res['baseline_last4d_flux']['test']['roc_auc']:.2f})")
    pos = y[te][o] == 1
    axs[1].plot(tt[pos], np.full(pos.sum(), 1.03), "v", color="#d9480f", ms=6, label="true: onset within horizon")
    axs[1].set_ylim(-0.05, 1.08); axs[1].set_xlabel("MJD (window end = moment of prediction)")
    axs[1].set_ylabel("predicted probability"); axs[1].legend(fontsize=8, loc="center left")
    fig.tight_layout(); fig.savefig(C.FIGS / "fig5_outburst_forecast.png", dpi=130); plt.close(fig)
    print(json.dumps({k: res[k] for k in ("n_onsets_total", "n_onsets_test_period", "counts", "best_epoch")}))
    for k in ("mlp", "baseline_last4d_flux"):
        t = res[k]["test"]
        print(f"{k:22s} test AUC {t['roc_auc']:.3f} bal.acc {t['balanced_accuracy']:.3f} "
              f"prec {t['precision']:.3f} rec {t['recall']:.3f} CM {t['confusion']}")


#                        main code                         #
if __name__ == "__main__":
    main()
