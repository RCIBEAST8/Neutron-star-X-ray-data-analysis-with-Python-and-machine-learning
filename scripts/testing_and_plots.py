# Aql X-1 X-ray analysis
# tests the models on data they've never seen and makes the plots

# code breakdown
# metrics on the test set (Dec 2020 onward), compared against a one-line baseline that just thresholds the
# window's mean flux (threshold tuned on the training set). then the figures: light curve vs power spectrum,
# loss curves, confusion matrices, and the test period timeline of predicted outburst probability

# libs
import json

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from flax import serialization

matplotlib.use("Agg")   # no plot window, just saves the pngs
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402
from neural_network import SpectrumMLP, predict_proba  # noqa: E402

# plot style and colours
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
COL = {"q": "#2b6cb0", "a": "#d9480f", "full": "#2f855a", "shape": "#805ad5", "base": "#718096"}


# accuracy, balanced accuracy, precision, recall, F1 and ROC AUC for one set of predictions
def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp, tn = int(np.sum((pred == 1) & (y == 1))), int(np.sum((pred == 0) & (y == 0)))
    fp, fn = int(np.sum((pred == 1) & (y == 0))), int(np.sum((pred == 0) & (y == 1)))
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    order = np.argsort(p)                                  # ROC AUC from the ranks (rank-sum)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": 0.5 * (rec + spec),
            "precision": prec, "recall": rec, "specificity": spec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9), "roc_auc": float(auc),
            "confusion": [[tn, fp], [fn, tp]], "n": int(len(y)), "n_active": int(npos)}


# loads a saved model plus its normalisation
def load_model(feat, nin):
    model = SpectrumMLP()
    tmpl = model.init(__import__("jax").random.PRNGKey(0), jnp.zeros((1, nin)))["params"]
    params = serialization.from_bytes(tmpl, (C.MODELS / f"mlp_{feat}_params.msgpack").read_bytes())
    nrm = np.load(C.MODELS / f"mlp_{feat}_norm.npz")
    return model, params, nrm["mu"], nrm["sd"]


#                    evaluate and plot                     #
def main():
    d = np.load(C.PROC / "dataset.npz")
    w = np.load(C.PROC / "windows.npz")
    lc = pd.read_csv(C.PROC / "aqlx1_clean_lightcurve_0p5d.csv")
    y, split, t0 = d["y"], d["split"], d["t_start"]
    tr, va, te = split == 0, split == 1, split == 2
    tmid = t0 + C.WIN * C.BIN_DAYS / 2

    #                models + baseline                 #
    prob, res = {}, {}
    for feat in ("full", "shape"):
        X = d[f"X_{feat}"]
        model, params, mu, sd = load_model(feat, X.shape[1])
        prob[feat] = np.asarray(predict_proba(model, params, jnp.asarray((X - mu) / sd, jnp.float32)))[:, 1]
        res[feat] = {"val": metrics(y[va], prob[feat][va]), "test": metrics(y[te], prob[feat][te])}

    wm = d["window_mean"]                                   # baseline: just a threshold on each window's mean flux
    cand = np.unique(wm[tr])
    bacc = [0.5 * (np.mean(wm[tr][y[tr] == 1] >= c) + np.mean(wm[tr][y[tr] == 0] < c)) for c in cand]
    thr = cand[int(np.argmax(bacc))]
    sc = 1 / (1 + np.exp(-(wm - thr) / (np.std(wm[tr]) * 0.05)))   # smooth score that's 0.5 right at the threshold
    prob["base"] = sc
    res["baseline_mean_flux"] = {"threshold_norm_flux": float(thr),
                                 "val": metrics(y[va], sc[va]), "test": metrics(y[te], sc[te])}
    (C.REPORTS / "metrics.json").write_text(json.dumps(res, indent=2))

    #       fig 1: light curve vs power spectrum       #
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, :])
    for k, c in (("ASM", "#4a5568"), ("MAXI", "#1a202c")):
        m = lc.instrument == k
        ax.plot(lc.mjd[m], lc.flux_norm[m], ".", ms=1.5, color=c, label=f"RXTE/ASM (scaled)" if k == "ASM" else "MAXI 2-10 keV")
    ia = np.where(tr & (y == 1))[0]
    ia = ia[np.argmax(w["active_frac"][ia] * (w["instrument"][ia] == "MAXI"))]
    iq = np.where(tr & (y == 0) & (w["instrument"] == "MAXI") & (w["mask"].mean(1) > 0.85))[0][40]
    for i, c, lab in ((iq, COL["q"], "quiescent window"), (ia, COL["a"], "active window")):
        ax.axvspan(t0[i], t0[i] + C.WIN * C.BIN_DAYS, color=c, alpha=0.25, label=lab)
    ax.axvspan(C.TEST_START, lc.mjd.max(), color="#f6e05e", alpha=0.15, label="test period")
    ax.set_xlim(lc.mjd.min(), lc.mjd.max()); ax.set_ylim(-0.15, 1.3)
    ax.set_xlabel("MJD"); ax.set_ylabel("normalised flux")
    ax.set_title("Aql X-1 cleaned X-ray light curve, 1996-2026 (0.5 d bins)")
    ax.legend(loc="upper left", ncol=5, fontsize=8, markerscale=6)

    ax = fig.add_subplot(gs[1, 0])
    tt = np.arange(C.WIN) * C.BIN_DAYS
    for i, c, lab in ((iq, COL["q"], "quiescent"), (ia, COL["a"], "active outburst")):
        ax.plot(tt, w["flux"][i], color=c, lw=1.2, label=f"{lab} (MJD {t0[i]:.0f})")
    ax.set_xlabel("days into window"); ax.set_ylabel("normalised flux")
    ax.set_title("Raw light curve: two 128-point windows"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    f = d["freq_full"]
    for i, c, lab in ((iq, COL["q"], "quiescent"), (ia, COL["a"], "active outburst")):
        ax.loglog(f[1:], 10 ** d["X_full"][i, 1:], color=c, lw=1.2, label=lab)
        ax.axhline(C.WIN * np.mean(w["err"][i] ** 2), color=c, ls=":", lw=1)
    ax.set_xlabel("frequency [cycles / day]"); ax.set_ylabel("power |FFT|$^2$")
    ax.set_title("Their power density spectra (dotted = white-noise level)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGS / "fig1_lightcurve_vs_power_spectrum.png"); plt.close(fig)

    #    fig 1b: average spectra per class (train)     #
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, feat, ttl in ((axs[0], "full", "full PDS (DC included)"),
                          (axs[1], "shape", "noise-normalised PDS (DC removed)")):
        X, fr = d[f"X_{feat}"], d[f"freq_{feat}"]
        sl = slice(1, None) if feat == "full" else slice(None)
        for cls, c, lab in ((0, COL["q"], "quiescent"), (1, COL["a"], "active")):
            q = np.percentile(X[tr & (y == cls)][:, sl], [25, 50, 75], axis=0)
            ax.fill_between(fr[sl], 10 ** q[0], 10 ** q[2], color=c, alpha=0.2)
            ax.loglog(fr[sl], 10 ** q[1], color=c, label=f"{lab} (median, IQR)")
        if feat == "shape":
            ax.axhline(1, color="k", ls=":", lw=1, label="pure white noise")
        ax.set_xlabel("frequency [cycles / day]"); ax.set_ylabel("power"); ax.set_title(ttl); ax.legend(fontsize=8)
    fig.suptitle("What the network sees: training-set spectra by class")
    fig.tight_layout(); fig.savefig(C.FIGS / "fig1b_class_average_spectra.png"); plt.close(fig)

    #                fig 2: loss curves                #
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for feat in ("full", "shape"):
        h = json.loads((C.MODELS / f"mlp_{feat}_history.json").read_text())
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axs[0].semilogy(ep, h["train_loss"], color=COL[feat], label=f"{feat}: train")
        axs[0].semilogy(ep, h["val_loss"], color=COL[feat], ls="--", label=f"{feat}: validation")
        axs[0].axvline(h["best_epoch"] + 1, color=COL[feat], lw=0.8, ls=":")
        axs[1].plot(ep, h["val_bacc"], color=COL[feat], label=feat)
        axs[1].axvline(h["best_epoch"] + 1, color=COL[feat], lw=0.8, ls=":")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("weighted cross-entropy")
    axs[0].set_title("Training loss (dotted line = epoch kept, lowest validation loss)")
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("balanced accuracy"); axs[1].set_ylim(0.5, 1)
    axs[1].set_title("Validation balanced accuracy"); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGS / "fig2_training_loss.png"); plt.close(fig)

    #       fig 3: confusion matrices (test set)       #
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, ttl in ((axs[0], "full", "MLP, full PDS"), (axs[1], "shape", "MLP, noise-normalised PDS"),
                         (axs[2], "baseline_mean_flux", "baseline: mean-flux threshold")):
        m = res[key]["test"]
        cm = np.array(m["confusion"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=14,
                    color="white" if v > cm.sum() / 2 else "black")
        ax.set_xticks([0, 1], ["quiescent", "active"]); ax.set_yticks([0, 1], ["quiescent", "active"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
        ax.set_title(f"{ttl}\nacc {m['accuracy']:.3f} | bal. acc {m['balanced_accuracy']:.3f} | AUC {m['roc_auc']:.3f}",
                     fontsize=9)
    fig.suptitle(f"Unseen test set: {res['full']['test']['n']} windows from Dec 2020 onward "
                 f"({res['full']['test']['n_active']} active)")
    fig.tight_layout(); fig.savefig(C.FIGS / "fig3_confusion_matrices.png"); plt.close(fig)

    #          fig 4: test period predictions          #
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, height_ratios=[1, 1])
    m = lc.mjd >= C.TEST_START - 10
    axs[0].plot(lc.mjd[m], lc.flux_norm[m], ".", ms=2, color="k")
    axs[0].set_ylabel("normalised flux"); axs[0].set_title("Test period: light curve and model output per window (gaps = windows dropped for >30% missing data)")
    o = np.argsort(tmid[te])
    tt_ = tmid[te][o]
    brk = np.r_[False, np.diff(tt_) > 2 * C.STRIDE * C.BIN_DAYS]      # doesn't draw lines across dropped windows
    for k, lab in (("full", "MLP full PDS"), ("shape", "MLP noise-normalised PDS")):
        pp = prob[k][te][o].astype(float)
        axs[1].plot(np.insert(tt_, np.where(brk)[0], np.nan), np.insert(pp, np.where(brk)[0], np.nan),
                    "-o", ms=2.5, color=COL[k], label=f"P(active), {lab}")
    ib = np.where(brk)[0]
    axs[1].fill_between(np.insert(tt_, ib, np.nan), 0, np.insert(y[te][o].astype(float), ib, np.nan),
                        step="mid", color=COL["a"], alpha=0.15, label="true label = active")
    axs[1].axhline(0.5, color="k", ls=":", lw=0.8)
    axs[1].set_ylabel("predicted P(active)"); axs[1].set_xlabel("MJD (window centre)"); axs[1].set_ylim(-0.05, 1.05)
    axs[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(C.FIGS / "fig4_test_predictions_timeline.png"); plt.close(fig)

    #      supplementary: ASM -> MAXI calibration      #
    ov = (lc.mjd >= C.SWITCH_MJD) & np.isfinite(lc.asm_scaled) & np.isfinite(lc.maxi)
    p1 = json.loads((C.REPORTS / "importing_data_log.json").read_text())["cross_calibration"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(lc.asm_scaled[ov], lc.maxi[ov], ".", ms=3, alpha=0.5)
    lim = [-0.3, max(lc.maxi[ov].max(), lc.asm_scaled[ov].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("RXTE/ASM x %.4f  [MAXI-scale ph/cm$^2$/s]" % p1["scale_maxi_per_asm_cps"])
    ax.set_ylabel("MAXI 2-10 keV [ph/cm$^2$/s]")
    ax.set_title(f"Cross-calibration on the 2009-2011 overlap\n(all overlap bins shown; fit uses the {p1['n_bins']} detected in both)")
    ax.legend(); fig.tight_layout(); fig.savefig(C.FIGS / "supp_cross_calibration.png"); plt.close(fig)

    # test set summary in the terminal
    for k in ("full", "shape", "baseline_mean_flux"):
        t = res[k]["test"]
        print(f"{k:20s} test acc {t['accuracy']:.3f}  bal.acc {t['balanced_accuracy']:.3f}  "
              f"prec {t['precision']:.3f}  rec {t['recall']:.3f}  AUC {t['roc_auc']:.3f}  CM {t['confusion']}")


#                        main code                         #
if __name__ == "__main__":
    main()
