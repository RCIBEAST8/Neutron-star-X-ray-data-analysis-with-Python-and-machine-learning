# multi-source extension, experiments
# cross-source experiments with the same Flax MLP (neural_network.SpectrumMLP)

# code breakdown
# 1. leave-one-source-out: train on some stars, test on a star the model has never seen
# 2. fine-tuning: take the Aql X-1 model and adapt it with GX 339-4's early years, test on its later years,
#    compared with (a) the Aql model untouched and (b) a model trained on early GX 339-4 only
# each run is done for all three feature sets (full / noise / slope, see ms_data.py) and compared with a baseline
# that just thresholds the window's mean flux in Crab units. the loss is weighted so every training star counts
# equally, and both classes count equally inside each star

# libs
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

import ms_config as C
from neural_network import SpectrumMLP, predict_proba
from testing_and_plots import metrics

FEATS = ("full", "noise", "slope")
EPOCHS, BATCH, LR, WD = 200, 64, 3e-4, 0.03   # same training settings as the main Aql X-1 code
SPLITS = {   # run name: (stars it trains on, stars it gets tested on)
    "A: train Aql":            (["Aql X-1"],             ["GX 339-4", "V404 Cyg", "Sco X-1"]),
    "B: train Aql + GX 339-4": (["Aql X-1", "GX 339-4"], ["V404 Cyg", "Sco X-1"]),
    "C: train Aql + V404":     (["Aql X-1", "V404 Cyg"], ["GX 339-4", "Sco X-1"]),
    "D: train GX 339-4":       (["GX 339-4"],            ["Aql X-1", "V404 Cyg", "Sco X-1"]),
}


# loads one star's windows and features
def load(name):
    return dict(np.load(C.PROC / f"ds_{name.replace(' ', '_')}.npz"))


# last 20% of a star's windows (in time) for validation, with a purge gap
def time_val_split(t0, frac=0.8):
    cut = np.quantile(t0, frac)
    tr = t0 + C.WIN * C.BIN_DAYS <= cut - C.PURGE_DAYS
    va = t0 >= cut
    return tr, va


# weights so every star, and every class inside it, counts the same
def sample_weights(src, y):
    w = np.zeros(len(y), np.float32)
    for s in np.unique(src):
        for c in (0, 1):
            m = (src == s) & (y == c)
            if m.any():
                w[m] = 1.0 / m.sum()
    return w / w.mean()


# trains one model (or fine-tunes one if init is given), keeps the epoch with the lowest val loss
def train(X, y, w, Xv, yv, wv, init=None, epochs=EPOCHS, lr=LR, seed=C.SEED):
    model = SpectrumMLP()
    key = jax.random.PRNGKey(seed)
    params = init if init is not None else model.init(key, jnp.zeros((1, X.shape[1])))["params"]
    tx = optax.chain(optax.add_decayed_weights(WD), optax.adam(lr))
    st = tx.init(params)

    # weighted cross-entropy, dropout only on while training
    def wxent(p, xb, yb, wb, rng=None, train_=False):
        kw = {"rngs": {"dropout": rng}} if train_ else {}
        lg = model.apply({"params": p}, xb, train=train_, **kw)
        return jnp.sum(optax.softmax_cross_entropy(lg, jax.nn.one_hot(yb, 2)) * wb) / jnp.sum(wb)

    # one jitted training step
    @jax.jit
    def step(p, s, xb, yb, wb, rng):
        l, g = jax.value_and_grad(lambda q: wxent(q, xb, yb, wb, rng, True))(p)
        u, s = tx.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    vloss = jax.jit(lambda p: wxent(p, Xv, yv, wv))
    rng = np.random.default_rng(seed)
    best, hist = (float(vloss(params)), params, -1), {"train": [], "val": []}
    for ep in range(epochs):
        perm, ls = rng.permutation(len(y)), []
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            key, k = jax.random.split(key)
            params, st, l = step(params, st, X[b], y[b], w[b], k)
            ls.append(float(l))
        v = float(vloss(params))
        hist["train"].append(float(np.mean(ls))); hist["val"].append(v)
        if v < best[0]:
            best = (v, params, ep)
    return model, best[1], best[2], hist


# metrics on one test star. Sco X-1 is always active so AUC doesn't exist for it,
# only the fraction it calls active means anything
def evaluate(p, y, base=None):
    out = {"n": int(len(y)), "n_active": int(y.sum()), "frac_pred_active": float(np.mean(p >= 0.5))}
    if 0 < y.sum() < len(y):
        m = metrics(y, p)
        out.update({k: m[k] for k in ("accuracy", "balanced_accuracy", "recall", "specificity", "roc_auc", "confusion")})
    else:
        out["note"] = "single class: AUC undefined, only the fraction predicted active is meaningful"
    return out


# best mean-flux threshold on the training stars (by balanced accuracy)
def baseline(train_sets, wm_tr, y_tr):
    cand = np.quantile(wm_tr, np.linspace(0, 1, 400))
    bacc = [0.5 * (np.mean(wm_tr[y_tr == 1] >= c) + np.mean(wm_tr[y_tr == 0] < c)) for c in cand]
    return float(cand[int(np.argmax(bacc))])


#                       experiments                        #
def main():
    data = {s: load(s) for s in C.SOURCES}
    results, preds = {}, {}

    #            leave-one-source-out runs             #
    for run, (train_src, test_src) in SPLITS.items():
        results[run] = {}
        for feat in FEATS:
            Xtr, ytr, src_tr, Xva, yva, src_va, wm_tr = [], [], [], [], [], [], []
            for s in train_src:
                d = data[s]
                tr, va = time_val_split(d["t_start"])
                Xtr.append(d[f"X_{feat}"][tr]); ytr.append(d["y"][tr]); src_tr += [s] * tr.sum()
                Xva.append(d[f"X_{feat}"][va]); yva.append(d["y"][va]); src_va += [s] * va.sum()
                wm_tr.append(d["window_mean"][tr])
            Xtr, ytr, Xva, yva = np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(Xva), np.concatenate(yva)
            src_tr, src_va, wm_tr = np.array(src_tr), np.array(src_va), np.concatenate(wm_tr)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            nz = lambda a: jnp.asarray((a - mu) / sd, jnp.float32)   # noqa: E731
            model, params, best_ep, hist = train(nz(Xtr), jnp.asarray(ytr), jnp.asarray(sample_weights(src_tr, ytr)),
                                                 nz(Xva), jnp.asarray(yva), jnp.asarray(sample_weights(src_va, yva)))
            tag = run.split(":")[0]
            (C.MODELS / f"{tag}_{feat}_params.msgpack").write_bytes(serialization.to_bytes(params))
            np.savez(C.MODELS / f"{tag}_{feat}_norm.npz", mu=mu, sd=sd)
            r = {"best_epoch": best_ep, "train_windows": int(len(ytr)), "loss_history": hist}
            thr = baseline(train_src, wm_tr, ytr)
            r["baseline_threshold_crab"] = thr
            for s in test_src:
                d = data[s]
                p = np.asarray(predict_proba(model, params, nz(d[f"X_{feat}"])))[:, 1]
                pb_score = 1 / (1 + np.exp(-(d["window_mean"] - thr) / 0.005))
                r[s] = {"mlp": evaluate(p, d["y"]), "baseline": evaluate(pb_score, d["y"])}
                preds[f"{tag}|{feat}|{s}"] = p
            results[run][feat] = r
            print(f"{run:25s} {feat:5s} best ep {best_ep:3d} | " + " | ".join(
                f"{s}: AUC {r[s]['mlp'].get('roc_auc', float('nan')):.3f} (base {r[s]['baseline'].get('roc_auc', float('nan')):.3f}) "
                f"bal {r[s]['mlp'].get('balanced_accuracy', float('nan')):.2f} fracAct {r[s]['mlp']['frac_pred_active']:.2f}"
                for s in test_src))

    #             fine-tuning on GX 339-4              #
    # Aql model -> first half of GX 339-4, tested on the second half
    gx = data["GX 339-4"]
    t0 = gx["t_start"]
    cut = np.quantile(t0, 0.5)
    early = t0 + C.WIN * C.BIN_DAYS <= cut - C.PURGE_DAYS
    late = t0 >= cut
    e_tr, e_va = time_val_split(np.where(early, t0, np.inf)[early])
    idx_e = np.where(early)[0]
    ft = {"cut_mjd": float(cut), "n_early": int(early.sum()), "n_late": int(late.sum()),
          "late_active": int(gx["y"][late].sum())}
    for feat in FEATS:
        aql = data["Aql X-1"]
        tr, va = time_val_split(aql["t_start"])
        nrm = np.load(C.MODELS / f"A_{feat}_norm.npz")
        mu, sd = nrm["mu"], nrm["sd"]
        nz = lambda a: jnp.asarray((a - mu) / sd, jnp.float32)   # noqa: E731
        model = SpectrumMLP()
        tmpl = model.init(jax.random.PRNGKey(0), jnp.zeros((1, aql[f"X_{feat}"].shape[1])))["params"]
        p_aql = serialization.from_bytes(tmpl, (C.MODELS / f"A_{feat}_params.msgpack").read_bytes())
        Xe, ye = gx[f"X_{feat}"][idx_e], gx["y"][idx_e]
        Xl, yl = gx[f"X_{feat}"][late], gx["y"][late]
        we = sample_weights(np.zeros(len(ye)), ye)
        a, b = idx_e[e_tr], idx_e[e_va]
        sl_tr, sl_va = np.isin(idx_e, a), np.isin(idx_e, b)
        out = {"zero_shot": evaluate(np.asarray(predict_proba(model, p_aql, nz(Xl)))[:, 1], yl)}
        _, p_ft, ep_ft, _ = train(nz(Xe[sl_tr]), jnp.asarray(ye[sl_tr]), jnp.asarray(we[sl_tr]),
                                  nz(Xe[sl_va]), jnp.asarray(ye[sl_va]), jnp.asarray(we[sl_va]),
                                  init=p_aql, epochs=100, lr=1e-4)
        out["fine_tuned"] = evaluate(np.asarray(predict_proba(model, p_ft, nz(Xl)))[:, 1], yl)
        out["fine_tuned"]["best_epoch"] = ep_ft
        mu2, sd2 = Xe[sl_tr].mean(0), Xe[sl_tr].std(0) + 1e-6
        nz2 = lambda a: jnp.asarray((a - mu2) / sd2, jnp.float32)   # noqa: E731
        _, p_sc, ep_sc, _ = train(nz2(Xe[sl_tr]), jnp.asarray(ye[sl_tr]), jnp.asarray(we[sl_tr]),
                                  nz2(Xe[sl_va]), jnp.asarray(ye[sl_va]), jnp.asarray(we[sl_va]))
        out["scratch_on_early_gx"] = evaluate(np.asarray(predict_proba(model, p_sc, nz2(Xl)))[:, 1], yl)
        ft[feat] = out
        preds[f"FT|{feat}|zero"] = np.asarray(predict_proba(model, p_aql, nz(Xl)))[:, 1]
        preds[f"FT|{feat}|tuned"] = np.asarray(predict_proba(model, p_ft, nz(Xl)))[:, 1]
        print(f"fine-tune {feat:5s}: zero-shot AUC {out['zero_shot']['roc_auc']:.3f} bal {out['zero_shot']['balanced_accuracy']:.2f} | "
              f"fine-tuned AUC {out['fine_tuned']['roc_auc']:.3f} bal {out['fine_tuned']['balanced_accuracy']:.2f} | "
              f"scratch AUC {out['scratch_on_early_gx']['roc_auc']:.3f} bal {out['scratch_on_early_gx']['balanced_accuracy']:.2f}")
    results["fine_tune_GX339"] = ft
    (C.REPORTS / "ms_results.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(C.PROC / "ms_predictions.npz", **{k.replace(" ", "_"): v for k, v in preds.items()})


#                        main code                         #
if __name__ == "__main__":
    main()
