# multi-source extension, figures
# makes the five figures for the multi-source runs

# libs
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # no plot window, just saves the pngs
import matplotlib.pyplot as plt  # noqa: E402

import ms_config as C  # noqa: E402

# plot style, colours, and the results + predictions from ms_experiments.py
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
SRC = list(C.SOURCES)
COL = {"Aql X-1": "#2b6cb0", "GX 339-4": "#2f855a", "V404 Cyg": "#d9480f", "Sco X-1": "#805ad5"}
FEATS = ("full", "noise", "slope")
res = json.loads((C.REPORTS / "ms_results.json").read_text())
data = {s: dict(np.load(C.PROC / f"ds_{s.replace(' ', '_')}.npz")) for s in SRC}
pred = dict(np.load(C.PROC / "ms_predictions.npz"))

#            fig 1: light curves in Crab units             #
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for ax, s in zip(axs, SRC):
    lc = pd.read_csv(C.PROC / f"lc_{s.replace(' ', '_')}.csv")
    for inst, mk in (("ASM", "#718096"), ("MAXI", "k"), ("BAT", "#d9480f")):
        m = lc.instrument == inst
        if m.any():
            ax.plot(lc.mjd[m], lc.mcrab[m], ".", ms=1.2, color=mk, label=inst)
    act = lc.active.fillna(0).values > 0
    ax.fill_between(lc.mjd, 0, 1, where=act, transform=ax.get_xaxis_transform(), color=COL[s], alpha=0.12, lw=0)
    ax.axhline(C.ACTIVE_MCRAB, color=COL[s], ls="--", lw=0.8)
    ax.set_yscale("symlog", linthresh=30)
    ax.set_ylabel("mCrab")
    ax.set_title(f"{s} ({C.SOURCES[s][3]}), shaded = active, dashed = {C.ACTIVE_MCRAB:.0f} mCrab threshold", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, markerscale=6, ncol=3)
axs[-1].set_xlabel("MJD")
fig.tight_layout(); fig.savefig(C.FIGS / "ms_fig1_lightcurves_mcrab.png"); plt.close(fig)

#         fig 2: median spectra per class and star         #
fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, feat in zip(axs, FEATS):
    for s in SRC:
        d = data[s]
        X = d[f"X_{feat}"][:, 1:] if feat == "full" else d[f"X_{feat}"]
        f = np.fft.rfftfreq(C.WIN, C.BIN_DAYS)[1:]
        for cls, ls in ((0, ":"), (1, "-")):
            m = d["y"] == cls
            if m.sum() >= 5:
                ax.loglog(f, 10 ** np.median(X[m], 0), ls, color=COL[s], lw=1.3,
                          label=f"{s} {'active' if cls else 'quiet'}")
    ax.set_xlabel("frequency [cycles / day]")
    ax.set_title({"full": "full PDS (Crab units)", "noise": "noise-normalised", "slope": "slope only (sum = 1)"}[feat])
axs[0].set_ylabel("median power")
axs[2].legend(fontsize=7, ncol=2)
fig.suptitle("Median power spectra by source and class (solid = active, dotted = quiet)")
fig.tight_layout(); fig.savefig(C.FIGS / "ms_fig2_spectra_by_source.png"); plt.close(fig)

#            fig 3: AUC heatmap + Sco X-1 test             #
runs = [r for r in res if r != "fine_tune_GX339"]
cells = [(r, s) for r in runs for s in ("Aql X-1", "GX 339-4", "V404 Cyg") if s in res[r]["full"]]
rows = ["baseline (mean flux)"] + [f"MLP {f}" for f in FEATS]
M = np.full((len(rows), len(cells)), np.nan)
for j, (r, s) in enumerate(cells):
    M[0, j] = res[r]["full"][s]["baseline"].get("roc_auc", np.nan)
    for i, f in enumerate(FEATS):
        M[i + 1, j] = res[r][f][s]["mlp"].get("roc_auc", np.nan)
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), width_ratios=[2.2, 1])
im = axs[0].imshow(M, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
for (i, j), v in np.ndenumerate(M):
    axs[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
axs[0].set_xticks(range(len(cells)), [f"{r.split(':')[0]}  -> {s}" for r, s in cells], rotation=30, ha="right")
axs[0].set_yticks(range(len(rows)), rows); axs[0].grid(False)
axs[0].set_title("ROC AUC on a source the model never saw\n(A: Aql, B: Aql+GX, C: Aql+V404, D: GX)")
fig.colorbar(im, ax=axs[0], fraction=0.03)
w = 0.2
for i, f in enumerate(FEATS):
    v = [res[r][f]["Sco X-1"]["mlp"]["frac_pred_active"] for r in runs]
    axs[1].bar(np.arange(len(runs)) + (i - 1) * w, v, w, label=f"MLP {f}")
axs[1].axhline(1, color="k", ls="--", lw=0.8)
axs[1].set_xticks(range(len(runs)), [r.split(":")[0] for r in runs])
axs[1].set_ylabel("fraction of windows called active"); axs[1].set_ylim(0, 1.05)
axs[1].set_title("Sco X-1 stress test (always bright,\ncorrect answer = 1.0)")
axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(C.FIGS / "ms_fig3_cross_source_results.png"); plt.close(fig)

#            fig 4: V404 Cyg 2015 with model B             #
d = data["V404 Cyg"]
lc = pd.read_csv(C.PROC / "lc_V404_Cyg.csv")
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
m = (lc.mjd > 57100) & (lc.mjd < 57500)
axs[0].plot(lc.mjd[m], lc.mcrab[m], ".", ms=2, color="k")
axs[0].set_yscale("symlog", linthresh=30); axs[0].set_ylabel("mCrab (BAT 15-50 keV here)")
axs[0].set_title("V404 Cyg, never seen in training: June 2015 outburst (model B, trained on Aql X-1 + GX 339-4)")
tm = d["t_start"] + C.WIN * C.BIN_DAYS / 2
k = (tm > 57100) & (tm < 57500)
o = np.argsort(tm[k])
tt = tm[k][o]
ib = np.where(np.diff(tt) > 2 * C.STRIDE * C.BIN_DAYS)[0] + 1        # no lines across dropped windows (gaps)
for f in FEATS:
    axs[1].plot(np.insert(tt, ib, np.nan), np.insert(pred[f"B|{f}|V404_Cyg"][k][o], ib, np.nan), "-o", ms=3, label=f"MLP {f}")
axs[1].fill_between(np.insert(tt, ib, np.nan), 0, np.insert(d["y"][k][o].astype(float), ib, np.nan), step="mid",
                    color="#d9480f", alpha=0.15, label="true active")
axs[1].set_ylabel("P(active)"); axs[1].set_xlabel("MJD (window centre)"); axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(C.FIGS / "ms_fig4_v404_2015.png"); plt.close(fig)

#                    fig 5: fine-tuning                    #
ft = res["fine_tune_GX339"]
fig, ax = plt.subplots(figsize=(8, 4.2))
w = 0.25
for i, (k, lab) in enumerate((("zero_shot", "Aql model, untouched"), ("fine_tuned", "Aql model fine-tuned on early GX"),
                              ("scratch_on_early_gx", "trained on early GX only"))):
    ax.bar(np.arange(3) + (i - 1) * w, [ft[f][k]["roc_auc"] for f in FEATS], w, label=lab)
ax.set_xticks(range(3), [f"MLP {f}" for f in FEATS]); ax.set_ylim(0.5, 1.02)
ax.set_ylabel("ROC AUC on GX 339-4 after MJD %.0f" % ft["cut_mjd"])
ax.set_title(f"Fine-tuning test: {ft['n_late']} later GX 339-4 windows ({ft['late_active']} active)")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(C.FIGS / "ms_fig5_fine_tuning.png"); plt.close(fig)
print("figures written")
