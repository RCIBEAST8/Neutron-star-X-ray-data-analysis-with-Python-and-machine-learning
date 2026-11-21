# multi-source extension, side check
# re-runs the cross-source results with the Swift/BAT splice taken out of V404 Cyg

# code breakdown
# - V404 Cyg (MAXI/ASM windows only) has no outburst left, so it turns into a false alarm test on an unseen
#   star, the fraction of its quiet windows each model calls active
# - the BAT 15-50 keV windows covering the June 2015 outburst get scored on their own, as a separate hard-band
#   check, never mixed into a soft-band metric
# - run C (Aql X-1 + V404 Cyg -> GX 339-4) gets retrained with V404's BAT windows left out

# libs
import json

import jax.numpy as jnp
import numpy as np

import ms_config as C
from ms_experiments import FEATS, load, sample_weights, time_val_split, train
from neural_network import predict_proba
from testing_and_plots import metrics

# saved predictions from ms_experiments.py, V404 split into soft (MAXI/ASM) and BAT windows
pred = dict(np.load(C.PROC / "ms_predictions.npz"))
v = load("V404 Cyg")
soft = v["instrument"] != "BAT"
bat = ~soft
out = {"v404_soft_windows": int(soft.sum()), "v404_soft_active": int(v["y"][soft].sum()),
       "v404_bat_windows": int(bat.sum()), "v404_bat_active": int(v["y"][bat].sum())}

# false alarms and outburst recall on V404 for the runs that never trained on it
for tag in ("A", "B", "D"):
    for f in FEATS:
        p = pred[f"{tag}|{f}|V404_Cyg"]
        out[f"{tag}_{f}"] = {
            "v404_soft_false_alarm_rate": float(np.mean(p[soft] >= 0.5)),
            "v404_bat_outburst_recall": float(np.mean(p[bat & (v["y"] == 1)] >= 0.5)),
            "v404_bat_quiet_false_alarm": float(np.mean(p[bat & (v["y"] == 0)] >= 0.5)),
        }

# run C again, without the spliced BAT windows
aql, gx = load("Aql X-1"), load("GX 339-4")
for f in FEATS:
    Xtr, ytr, src_tr, Xva, yva, src_va = [], [], [], [], [], []
    for name, d, keep in (("Aql X-1", aql, np.ones(len(aql["y"]), bool)), ("V404 Cyg", v, soft)):
        tr, va = time_val_split(d["t_start"])
        tr, va = tr & keep, va & keep
        Xtr.append(d[f"X_{f}"][tr]); ytr.append(d["y"][tr]); src_tr += [name] * tr.sum()
        Xva.append(d[f"X_{f}"][va]); yva.append(d["y"][va]); src_va += [name] * va.sum()
    Xtr, ytr, Xva, yva = map(np.concatenate, (Xtr, ytr, Xva, yva))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    nz = lambda a: jnp.asarray((a - mu) / sd, jnp.float32)   # noqa: E731
    m, params, ep, _ = train(nz(Xtr), jnp.asarray(ytr), jnp.asarray(sample_weights(np.array(src_tr), ytr)),
                             nz(Xva), jnp.asarray(yva), jnp.asarray(sample_weights(np.array(src_va), yva)))
    p = np.asarray(predict_proba(m, params, nz(gx[f"X_{f}"])))[:, 1]
    out[f"C_nosplice_{f}"] = {"gx339_auc": metrics(gx["y"], p)["roc_auc"], "best_epoch": ep,
                              "sco_frac_active": float(np.mean(np.asarray(predict_proba(m, params, nz(load("Sco X-1")[f"X_{f}"])))[:, 1] >= 0.5))}

(C.REPORTS / "ms_no_splice.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
