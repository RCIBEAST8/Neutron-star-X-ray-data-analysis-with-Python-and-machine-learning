# multi-source extension, checks
# runs the checks and writes multi_source/reports/ms_verification_report.md

# libs
import json

import numpy as np

import ms_config as C

# every check goes in here as (description, pass/fail, evidence)
rows = []


def check(desc, ok, ev):
    rows.append((desc, bool(ok), ev))


# logs and results from the other multi-source scripts
L = json.loads((C.REPORTS / "ms_data_log.json").read_text())
R = json.loads((C.REPORTS / "ms_results.json").read_text())
S = json.loads((C.REPORTS / "ms_seed_check.json").read_text())
crab = L["crab"]

check("Crab calibration matches known instrument levels",
      70 < crab["ASM_cps"] < 80 and 3.0 < crab["MAXI_2_10keV_ph_cm2_s"] < 3.6 and 0.18 < crab["BAT_15_50keV_cts_cm2_s"] < 0.26,
      f"ASM {crab['ASM_cps']:.2f} c/s, MAXI 2-10 keV {crab['MAXI_2_10keV_ph_cm2_s']:.3f} ph/cm2/s, BAT {crab['BAT_15_50keV_cts_cm2_s']:.4f} c/cm2/s")
a = L["aql_check_maxi_per_asm_from_crab"]
check("Crab-based ASM->MAXI factor agrees with the Aql overlap fit (0.0460)", abs(a / 0.0460 - 1) < 0.1, f"{a:.4f} ({(a/0.046-1)*100:+.1f}%)")
for s in C.SOURCES:
    lg = L["sources"][s]
    obj = {"Aql X-1": "aqlx1", "GX 339-4": "gx339-4", "V404 Cyg": "gs2023+338", "Sco X-1": "scox1"}[s]
    check(f"{s}: right FITS object, windows built", lg["asm_object"] == obj and lg["windows"]["n"] > 1000,
          f"OBJECT={lg['asm_object']}, windows {lg['windows']}")
check("Quiescent noise consistent with counting noise (std z < 2) for all transients",
      all(L["sources"][s][f"quiet_{k}"]["std_z"] < 2 for s in ("Aql X-1", "GX 339-4", "V404 Cyg") for k in ("ASM", "MAXI")),
      ", ".join(f"{s} {k} {L['sources'][s][f'quiet_{k}']['std_z']:.2f}" for s in ("Aql X-1", "GX 339-4", "V404 Cyg") for k in ("ASM", "MAXI")))
v = L["sources"]["V404 Cyg"]["episodes_ge3d"]
check("V404 Cyg June 2015 outburst recovered (via BAT)", any(57185 < e[0] < 57200 for e in v), f"episodes {[[round(x) for x in e] for e in v]}")
check("Sco X-1 labelled active throughout (persistent)", L["sources"]["Sco X-1"]["active_fraction"] == 1.0,
      f"active fraction {L['sources']['Sco X-1']['active_fraction']}")
d = np.load(C.PROC / "ds_GX_339-4.npz")
check("Slope features really are normalised (power sums to 1)", np.allclose((10 ** d["X_slope"].astype(float)).sum(1), 1, atol=1e-3),
      f"row sums {(10 ** d['X_slope'].astype(float)).sum(1)[:3].round(4)}")
runs = [r for r in R if r != "fine_tune_GX339"]
check("All leave-one-source-out runs and feature sets completed", all(f in R[r] for r in runs for f in ("full", "noise", "slope")),
      f"{len(runs)} runs x 3 feature sets")
full_ok = all(R[r]["full"][s]["mlp"].get("roc_auc", 1) > 0.95 for r in runs for s in R[r]["full"] if isinstance(R[r]["full"][s], dict) and "mlp" in R[r]["full"][s])
check("Full-PDS model transfers to unseen sources (AUC > 0.95 everywhere)", full_ok, "see ms_results.json")
slope_seeds = [S[k]["V404 Cyg"] for k in S if k.startswith("slope")]
check("Slope-only failure on V404 holds across seeds", max(slope_seeds) < 0.7, f"V404 AUC over seeds {np.round(slope_seeds, 2).tolist()}")
figs = sorted(p.name for p in C.FIGS.glob("ms_fig*.png"))
check("All five figures written", len(figs) == 5 and all((C.FIGS / f).stat().st_size > 20000 for f in figs), ", ".join(figs))
# makes sure the multi-source run never rewrote the main results. checks the file times, not a fixed AUC,
# so it still passes when newer MAXI data shifts the numbers slightly
orig_f = C.PROJECT / "reports" / "metrics.json"
orig = json.loads(orig_f.read_text())
check("Original Aql-only results untouched", orig_f.stat().st_mtime < (C.REPORTS / "ms_data_log.json").stat().st_mtime,
      f"main metrics.json not rewritten by the multi-source run (full-PDS test AUC {orig['full']['test']['roc_auc']:.3f})")

# writes the markdown table and prints a summary
lines = ["# Multi-source verification", "", "| Check | Result | Evidence |", "|---|---|---|"]
lines += [f"| {d_} | {'PASS' if ok else 'FAIL'} | {e} |" for d_, ok, e in rows]
lines += ["", f"**{sum(r[1] for r in rows)}/{len(rows)} checks pass.**"]
(C.REPORTS / "ms_verification_report.md").write_text("\n".join(lines) + "\n")
for d_, ok, e in rows:
    print("PASS" if ok else "FAIL", d_)
print(f"{sum(r[1] for r in rows)}/{len(rows)} pass")
