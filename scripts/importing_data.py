# Aql X-1 X-ray analysis
# downloads, cleans and chunks the X-ray light curves

# code breakdown
# downloads RXTE/ASM (HEASARC FITS), MAXI/GSC (RIKEN) and Swift/BAT (BAT is only there for context),
# throws out bad data, puts everything on one uniform 0.5 day grid, scales ASM onto the MAXI 2-10 keV
# scale, normalises it, flags every bin as quiet or active, then cuts the whole timeline into equal
# 128 point windows

# libs
import hashlib
import json
import urllib.request

import numpy as np
import pandas as pd
from astropy.io import fits

import config as C


#                         download                         #
# grabs each light curve if it's not already on disk, and logs the url, size and sha256 of each one
# (ends up in reports/importing_data_log.json)
def download():
    log = {}
    for name, (url, dest) in C.SOURCES.items():
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, dest)
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        log[name] = {"url": url, "file": dest.name, "bytes": dest.stat().st_size, "sha256": sha}
        print(f"  {name:10s} {dest.stat().st_size/1e6:7.2f} MB  {dest.name}")
    return log


#                          clean                           #
# ASM: drops non-finite rows and zero errors, then the ASM team's chi2 and exposure cuts
def load_asm(log):
    d = fits.open(C.SOURCES["rxte_asm"][1])[1].data
    df = pd.DataFrame({"mjd": d["TIME"].astype(float) + C.ASM_MJDREF,
                       "rate": d["RATE"].astype(float), "err": d["ERROR"].astype(float),
                       "rdchi": d["RDCHI_SQ"].astype(float), "expo": d["TIMEDEL"].astype(float)})
    n0 = len(df)
    df = df[np.isfinite(df.rate) & np.isfinite(df.err) & (df.err > 0)]
    n1 = len(df)
    df = df[(df.rdchi <= C.ASM_MAX_RDCHI) & (df.expo >= C.ASM_MIN_EXPO)]
    log["asm"] = {"dwells_raw": n0, "after_finite": n1, "after_quality_cuts": len(df),
                  "cuts": f"reduced chi2 <= {C.ASM_MAX_RDCHI}, exposure >= {C.ASM_MIN_EXPO} s"}
    return df[["mjd", "rate", "err"]]


# MAXI: drops rows with a zero error in any sub-band, then the hard 10-20 keV background hits
def load_maxi(log):
    M = np.loadtxt(C.SOURCES["maxi"][1])
    n0 = len(M)
    M = M[np.all(np.isfinite(M), axis=1)]
    M = M[np.all(M[:, [4, 6, 8]] > 0, axis=1)]          # a few rows have 0 error in a sub-band
    n1 = len(M)
    hard = (M[:, 7] / M[:, 8] > C.MAXI_HARD_SIG) & (M[:, 7] > C.MAXI_HARD_FRAC * M[:, 1])
    M = M[~hard]
    rate = M[:, 3] + M[:, 5]                             # 2-4 + 4-10 keV = 2-10 keV (closest to the ASM band, skips the noisy 10-20 keV)
    err = np.hypot(M[:, 4], M[:, 6])
    log["maxi"] = {"orbits_raw": n0, "after_finite_and_nonzero_err": n1,
                   "hard_background_hits_removed": int(hard.sum()), "kept": len(M),
                   "band": "2-10 keV (2-4 + 4-10), ph/cm2/s"}
    return pd.DataFrame({"mjd": M[:, 0], "rate": rate, "err": err})


# averages every point that lands in a bin, weighted by 1/error^2 (inverse variance)
# empty bins come out as NaN
def bin_uniform(df, edges):
    idx = np.digitize(df.mjd.values, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    idx, r, e = idx[ok], df.rate.values[ok], df.err.values[ok]
    w = 1 / e**2
    n = len(edges) - 1
    sw = np.bincount(idx, w, n)
    sr = np.bincount(idx, w * r, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sw > 0, sr / sw, np.nan), np.where(sw > 0, 1 / np.sqrt(sw), np.nan)


# works out the ASM -> MAXI 2-10 keV scale factor from the 2009-2011 overlap, only using bins where
# both detect the source (over 3 sigma), straight line fit through the origin
def cross_calibrate(fa, ea, fm, em, t, log):
    both = np.isfinite(fa) & np.isfinite(fm) & (fa / ea > 3) & (fm / em > 3) & (t >= C.SWITCH_MJD)
    x, y, sx, sy = fa[both], fm[both], ea[both], em[both]
    a = np.sum(x * y / sy**2) / np.sum(x**2 / sy**2)
    for _ in range(20):                                   # iterate so the ASM errors count too (errors in both axes)
        w = 1 / (sy**2 + a**2 * sx**2)
        a = np.sum(w * x * y) / np.sum(w * x**2)
    resid = (y - a * x) * np.sqrt(w)
    log["cross_calibration"] = {"scale_maxi_per_asm_cps": float(a), "n_bins": int(both.sum()),
                                "reduced_chi2": float(np.sum(resid**2) / (both.sum() - 1)),
                                "overlap": [float(C.SWITCH_MJD), float(t[np.isfinite(fa)].max())]}
    return a, both


# puts both instruments on the same grid, ASM gets scaled onto MAXI, then each bin gets its state
def clean(log):
    asm, maxi = load_asm(log), load_maxi(log)
    t0 = np.floor(asm.mjd.min())
    t1 = np.ceil(maxi.mjd.max())
    edges = np.arange(t0, t1 + C.BIN_DAYS, C.BIN_DAYS)
    tc = edges[:-1] + C.BIN_DAYS / 2
    fa, ea = bin_uniform(asm, edges)
    fm, em = bin_uniform(maxi, edges)
    a, both = cross_calibrate(fa, ea, fm, em, tc, log)

    use_maxi = tc >= C.SWITCH_MJD
    flux = np.where(use_maxi, fm, a * fa)
    err = np.where(use_maxi, em, a * ea)
    inst = np.where(use_maxi, "MAXI", "ASM")

    # each bin's state from a short running mean (empty bins ignored)
    s = pd.Series(flux).rolling(C.ACTIVE_SMOOTH, center=True, min_periods=3).mean().values
    active = np.where(np.isfinite(s), (s > C.ACTIVE_THRESH).astype(float), np.nan)

    filled = np.isfinite(flux)
    norm = np.nanpercentile(flux[filled], 99.9)           # normalised so the big outburst peaks sit around 1
    log["grid"] = {"bin_days": C.BIN_DAYS, "n_bins": len(tc), "filled_fraction": float(filled.mean()),
                   "filled_fraction_asm": float(filled[~use_maxi].mean()),
                   "filled_fraction_maxi": float(filled[use_maxi].mean()),
                   "norm_flux_unit_ph_cm2_s": float(norm), "active_threshold_ph_cm2_s": C.ACTIVE_THRESH,
                   "active_fraction_of_bins": float(np.nanmean(active))}
    lc = pd.DataFrame({"mjd": tc, "instrument": inst, "flux": flux, "err": err,
                       "flux_norm": flux / norm, "err_norm": err / norm,
                       "flux_smooth": s, "active": active,
                       "asm_scaled": a * fa, "maxi": fm})
    lc.to_csv(C.PROC / "aqlx1_clean_lightcurve_0p5d.csv", index=False)
    return lc, both


#                          chunk                           #
# cuts the timeline into 128 point windows every 4 days, throws out the bad ones and labels the rest
def chunk(lc, log):
    f = lc.flux_norm.values
    e = lc.err_norm.values
    act = lc.active.values
    inst = lc.instrument.values
    t = lc.mjd.values
    medE = {k: np.nanmedian(e[inst == k]) for k in ("ASM", "MAXI")}
    X, E, MASK, T0, INS, AF = [], [], [], [], [], []
    for i in range(0, len(f) - C.WIN + 1, C.STRIDE):
        sl = slice(i, i + C.WIN)
        if len(set(inst[sl])) > 1:                        # a window can't straddle the ASM/MAXI switch
            continue
        m = np.isfinite(f[sl])
        if m.mean() < C.MIN_FILL:
            continue
        x = np.arange(C.WIN)
        xf = np.interp(x, x[m], f[sl][m])                 # small gaps filled in linearly
        ef = np.where(m, e[sl], medE[inst[i]])
        a = act[sl][m & np.isfinite(act[sl])]
        X.append(xf); E.append(ef); MASK.append(m); T0.append(t[i]); INS.append(inst[i])
        AF.append(a.mean() if len(a) else 0.0)
    X, E, MASK = np.array(X), np.array(E), np.array(MASK)
    T0, INS, AF = np.array(T0), np.array(INS), np.array(AF)
    y = (AF >= C.LABEL_FRAC).astype(np.int32)
    np.savez_compressed(C.PROC / "windows.npz", flux=X, err=E, mask=MASK, t_start=T0,
                        instrument=INS, active_frac=AF, label=y, bin_days=C.BIN_DAYS)
    log["windows"] = {"n": len(X), "length_points": C.WIN, "length_days": C.WIN * C.BIN_DAYS,
                      "stride_days": C.STRIDE * C.BIN_DAYS, "min_fill": C.MIN_FILL,
                      "label_rule": f"active if >= {C.LABEL_FRAC:.0%} of filled bins active",
                      "n_active": int(y.sum()), "n_quiescent": int((y == 0).sum()),
                      "n_asm": int((INS == "ASM").sum()), "n_maxi": int((INS == "MAXI").sum())}
    return X, y


#                        main code                         #
if __name__ == "__main__":
    log = {"downloads": download()}
    lc, _ = clean(log)
    X, y = chunk(lc, log)
    (C.REPORTS / "importing_data_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != "downloads"}, indent=2))
