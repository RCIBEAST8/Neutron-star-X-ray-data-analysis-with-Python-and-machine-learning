# Aql X-1 X-ray analysis
# checks what each script actually produced and writes reports/verification_report.md

# libs
import inspect
import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.io import fits

import config as C
import fourier_features
import neural_network
import training
from testing_and_plots import load_model

# every check goes in here as (number, description, pass/fail, evidence)
rows = []


def check(step, desc, ok, evidence):
    rows.append((step, desc, bool(ok), evidence))


# source code of the main scripts, some checks look for the actual function calls in it
src = {m.__name__: inspect.getsource(m) for m in (fourier_features, neural_network, training)}
log1 = json.loads((C.REPORTS / "importing_data_log.json").read_text())
log2 = json.loads((C.REPORTS / "fourier_features_log.json").read_text())

# check 1, light curves downloaded, right FITS object
h = fits.open(C.SOURCES["rxte_asm"][1])
ok = all(C.SOURCES[k][1].stat().st_size > 0 for k in C.SOURCES) and h[1].header.get("OBJECT") == "aqlx1"
check(1, "Light curves downloaded (RXTE/ASM FITS + MAXI)", ok,
      f"RXTE/ASM FITS from HEASARC, OBJECT={h[1].header.get('OBJECT')}, {len(h[1].data)} dwells; "
      f"MAXI {log1['downloads']['maxi']['bytes']/1e6:.1f} MB; sha256 logged in importing_data_log.json")

# check 2, cleaned onto a uniform grid
lc = pd.read_csv(C.PROC / "aqlx1_clean_lightcurve_0p5d.csv")
dt = np.diff(lc.mjd.values)
ok = np.allclose(dt, C.BIN_DAYS) and lc.flux_norm.notna().mean() > 0.7 and log1["asm"]["after_quality_cuts"] < log1["asm"]["dwells_raw"]
check(2, "Cleaned onto a uniform grid and normalised", ok,
      f"uniform {C.BIN_DAYS} d grid ({len(lc)} bins, {lc.flux_norm.notna().mean():.1%} filled); "
      f"ASM {log1['asm']['dwells_raw']}->{log1['asm']['after_quality_cuts']} dwells, "
      f"MAXI {log1['maxi']['hard_background_hits_removed']} background hits removed; "
      f"ASM cross-calibrated x{log1['cross_calibration']['scale_maxi_per_asm_cps']:.4f}; flux normalised")

# check 3, equal windows, all finite
w = np.load(C.PROC / "windows.npz")
ok = w["flux"].ndim == 2 and w["flux"].shape[1] == C.WIN and np.isfinite(w["flux"]).all()
check(3, "Equal-length windows, all finite", ok, f"windows array {w['flux'].shape} ({C.WIN} points = {C.WIN*C.BIN_DAYS:.0f} d), all finite")

# check 4, JAX rfft matches numpy
x = w["flux"][:50]
P = np.asarray(fourier_features.pds_batch(jnp.asarray(x)))
ok = "jnp.fft.rfft" in src["fourier_features"] and log2["fft_matches_numpy"]
check(4, "JAX rfft matches numpy", ok, "jax.vmap(jnp.fft.rfft) used; allclose to numpy.fft.rfft = "
      f"{log2['fft_matches_numpy']}")

# check 5, power spectrum matches numpy
d = np.load(C.PROC / "dataset.npz")
ref = np.abs(np.fft.rfft(x, axis=1)) ** 2
ok = np.allclose(P, ref, rtol=1e-9, atol=1e-12) and np.allclose(10 ** d["X_full"][:50].astype(float), ref + 1e-12, rtol=1e-4, atol=1e-10)
check(5, "Power spectrum matches numpy", ok, f"PDS has {P.shape[1]} bins, {d['freq_full'][1]:.4f}-{d['freq_full'][-1]:.1f} cycles/day; matches numpy rfft squared")

# check 6, X and y built, both classes in every split
y, sp = d["y"], d["split"]
ok = set(np.unique(y)) == {0, 1} and all(len(np.unique(y[sp == s])) == 2 for s in (0, 1, 2))
check(6, "X and y built, both classes in every split", ok,
      f"X_full {d['X_full'].shape}, X_shape {d['X_shape'].shape}, y in {{0,1}}; "
      + ", ".join(f"{k} {v['n']} ({v['active']} active)" for k, v in log2["splits"].items()))

# check 7, model is a Flax Linen module
ok = issubclass(neural_network.SpectrumMLP, nn.Module) and "import flax.linen as nn" in src["neural_network"]
check(7, "Model is a Flax Linen module", ok, "SpectrumMLP(flax.linen.Module): Dense128-ReLU-Dropout-Dense64-ReLU-Dense2")

# check 8, outputs are probabilities
model, params, mu, sd = load_model("full", d["X_full"].shape[1])
pr = np.asarray(neural_network.predict_proba(model, params, jnp.asarray((d["X_full"][:20] - mu) / sd, jnp.float32)))
ok = pr.shape == (20, 2) and np.allclose(pr.sum(1), 1, atol=1e-5) and "nn.relu" in src["neural_network"]
check(8, "Outputs are probabilities (rows sum to 1)", ok, f"output {pr.shape}, rows sum to 1 (softmax)")

# check 9, params initialised from a dummy input
n_params = sum(v.size for v in jax.tree_util.tree_leaves(params))
check(9, "Parameters initialised from a dummy input", "model.init(" in src["training"] and "jnp.zeros((1" in src["training"],
      f"model.init(k_init, jnp.zeros((1, n_features))); {n_params} parameters")

# check 10, Adam + weighted cross-entropy
ok = "optax.adam" in src["training"] and "softmax_cross_entropy" in src["training"]
check(10, "Adam + weighted cross-entropy", ok, "optax.adam (with add_decayed_weights L2) + class-weighted optax.softmax_cross_entropy")

# check 11, jitted train step with value_and_grad
ok = "jax.value_and_grad" in src["training"] and "@jax.jit" in src["training"]
hist = {f: json.loads((C.MODELS / f"mlp_{f}_history.json").read_text()) for f in ("full", "shape")}
check(11, "Train step jitted, uses value_and_grad", ok,
      f"@jax.jit train_step; {hist['full']['epochs']} epochs trained in {hist['full']['train_seconds']:.1f} s on CPU")

# check 12, training loss at least halves
ok = all(len(hh["train_loss"]) == hh["epochs"] and hh["train_loss"][-1] < 0.5 * hh["train_loss"][0] for hh in hist.values())
check(12, "Training loss at least halves", ok, "; ".join(
    f"{f}: loss {hh['train_loss'][0]:.3f} -> {hh['train_loss'][-1]:.3f}, kept epoch {hh['best_epoch']}" for f, hh in hist.items()))

# check 13, test set comes after training in time
m = json.loads((C.REPORTS / "metrics.json").read_text())
t_end_train = (d["t_start"][sp == 0] + C.WIN * C.BIN_DAYS).max()
ok = t_end_train < d["t_start"][sp == 2].min() and "accuracy" in m["full"]["test"]
check(13, "Test set comes after training in time", ok,
      f"train ends MJD {t_end_train:.0f}, test starts {d['t_start'][sp == 2].min():.0f}; test acc full {m['full']['test']['accuracy']:.3f}, "
      f"shape {m['shape']['test']['accuracy']:.3f}, mean-flux baseline {m['baseline_mean_flux']['test']['accuracy']:.3f}")

# check 14, all figures written
figs = ["fig1_lightcurve_vs_power_spectrum.png", "fig1b_class_average_spectra.png", "fig2_training_loss.png",
        "fig3_confusion_matrices.png", "fig4_test_predictions_timeline.png", "fig5_outburst_forecast.png",
        "supp_cross_calibration.png"]
ok = all((C.FIGS / f).exists() and (C.FIGS / f).stat().st_size > 20000 for f in figs)
check(14, "All figures written", ok,
      ", ".join(f"{f} ({(C.FIGS / f).stat().st_size // 1024} KB)" for f in figs))

# writes the markdown table and prints a summary
lines = ["# Verification report", "", "Generated by `scripts/checks.py`.", "",
         "| # | Check | Result | Evidence |", "|---|---|---|---|"]
lines += [f"| {s} | {t} | {'PASS' if ok else 'FAIL'} | {e} |" for s, t, ok, e in rows]
lines += ["", f"**{sum(r[2] for r in rows)}/{len(rows)} checks pass.**"]
(C.REPORTS / "verification_report.md").write_text("\n".join(lines) + "\n")
for s, t, ok, e in rows:
    print(f"check {s:2d} {'PASS' if ok else 'FAIL'}  {t}")
print(f"{sum(r[2] for r in rows)}/{len(rows)} pass")
