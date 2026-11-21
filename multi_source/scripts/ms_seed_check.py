# multi-source extension, seed check
# retrains the Aql-only models (full, noise and slope) with three more seeds to make sure the cross-source
# results in ms_results.json aren't down to a lucky (or unlucky) starting point

# libs
import json

import jax.numpy as jnp
import numpy as np

import ms_config as C
from ms_experiments import load, sample_weights, time_val_split, train
from neural_network import predict_proba
from testing_and_plots import metrics

aql = load("Aql X-1")
tests = {s: load(s) for s in ("GX 339-4", "V404 Cyg", "Sco X-1")}
tr, va = time_val_split(aql["t_start"])
out = {}
for feat in ("full", "noise", "slope"):
    X = aql[f"X_{feat}"]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    nz = lambda a: jnp.asarray((a - mu) / sd, jnp.float32)   # noqa: E731
    for seed in (1, 2, 3):
        m, p, ep, _ = train(nz(X[tr]), jnp.asarray(aql["y"][tr]), jnp.asarray(sample_weights(np.zeros(tr.sum()), aql["y"][tr])),
                            nz(X[va]), jnp.asarray(aql["y"][va]), jnp.asarray(sample_weights(np.zeros(va.sum()), aql["y"][va])),
                            seed=seed)
        r = {"best_epoch": ep}
        for s, d in tests.items():
            pr = np.asarray(predict_proba(m, p, nz(d[f"X_{feat}"])))[:, 1]
            r[s] = metrics(d["y"], pr)["roc_auc"] if 0 < d["y"].sum() < len(d["y"]) else {"frac_pred_active": float(np.mean(pr >= 0.5))}
        out[f"{feat}_seed{seed}"] = r
        print(feat, seed, r)
(C.REPORTS / "ms_seed_check.json").write_text(json.dumps(out, indent=1))
