# multi-source extension, data
# cleans, Crab-normalises, labels and windows every star, then builds three Fourier feature sets

# code breakdown
# feature sets (all from jax.numpy.fft.rfft via fourier_features.pds_batch)
#   full   log10 |FFT|^2 of the window in Crab units, DC kept (carries the absolute flux)
#   noise  DC removed, divided by the window's expected white-noise power (carries variance vs noise)
#   slope  DC removed, divided by its own total power, so only the SHAPE of the spectrum is left

# libs
import hashlib
import json
import urllib.request

import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.io import fits

import ms_config as C
from fourier_features import pds_batch


#                         download                         #
# grabs any raw file that's missing and logs the url + sha256 for all of them
def download():
    log = {}
    for fname, url in C.DOWNLOADS.items():
        dest = C.RAW / fname
        if not dest.exists():
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(url, dest)
        log[fname] = {"url": url, "bytes": dest.stat().st_size, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    return log


#                          clean                           #
# the Crab's median rate in each instrument, everything gets divided by this to get Crab units
def crab_levels():
    h = fits.open(C.CRAB_FILES["ASM"])[1]
    asm = float(np.nanmedian(h.data["RATE"]))
    M = np.loadtxt(C.CRAB_FILES["MAXI"])
    M = M[np.all(np.isfinite(M), 1) & (M[:, 4] > 0) & (M[:, 6] > 0)]
    maxi = float(np.median(M[:, 3] + M[:, 5]))
    B = np.loadtxt(C.CRAB_FILES["BAT"], comments="#", usecols=(1, 10))
    bat = float(np.median(B[B[:, 1] == 0, 0]))
    return {"ASM_cps": asm, "MAXI_2_10keV_ph_cm2_s": maxi, "BAT_15_50keV_cts_cm2_s": bat}


# Swift/BAT orbit light curve (only used for V404's 2015 outburst), in mCrab
def load_bat(path, crab, log):
    B = np.loadtxt(path, comments="#", usecols=(0, 1, 2, 10))
    n0 = len(B)
    B = B[np.all(np.isfinite(B), 1) & (B[:, 2] > 0) & (B[:, 3] == 0)]    # DATA_FLAG == 0 (good data only)
    log.update({"bat_orbits": n0, "bat_kept": len(B)})
    return pd.DataFrame({"mjd": B[:, 0] / 86400 + 51910.0, "rate": B[:, 1] / crab * 1e3, "err": B[:, 2] / crab * 1e3})


# ASM dwells with the chi2 and exposure cuts, in mCrab
def load_asm(path, chi_cut, crab, log):
    h = fits.open(path)[1]
    d = h.data
    ref = h.header["MJDREFI"] + h.header.get("MJDREFF", 0.0)
    df = pd.DataFrame({"mjd": d["TIME"].astype(float) + ref, "rate": d["RATE"].astype(float),
                       "err": d["ERROR"].astype(float), "chi": d["RDCHI_SQ"].astype(float),
                       "expo": d["TIMEDEL"].astype(float)})
    n0 = len(df)
    df = df[np.isfinite(df.rate) & (df.err > 0) & (df.chi <= chi_cut) & (df.expo >= C.ASM_MIN_EXPO)]
    log.update({"asm_object": h.header.get("OBJECT"), "asm_dwells": n0, "asm_kept": len(df), "asm_chi2_cut": chi_cut})
    return pd.DataFrame({"mjd": df.mjd, "rate": df.rate / crab * 1e3, "err": df.err / crab * 1e3})


# MAXI orbits (2-4 + 4-10 keV), drops zero errors and isolated hard spikes, in mCrab
def load_maxi(path, crab, log):
    M = np.loadtxt(path)
    n0 = len(M)
    M = M[np.all(np.isfinite(M), 1) & np.all(M[:, [2, 4, 6, 8]] > 0, 1)]
    z_hard = M[:, 7] / M[:, 8]
    z_tot = M[:, 1] / M[:, 2]
    hard = (z_hard > C.MAXI_HARD_SIG) & (M[:, 7] > C.MAXI_HARD_FRAC * M[:, 1])
    t = M[:, 0]
    quiet_nb = np.zeros(len(M), bool)                      # both neighbouring orbits (within 0.2 d) undetected
    quiet_nb[1:-1] = ((np.abs(t[:-2] - t[1:-1]) < 0.2) & (np.abs(t[2:] - t[1:-1]) < 0.2)
                      & (z_tot[:-2] < 3) & (z_tot[2:] < 3))
    drop = hard & quiet_nb
    M = M[~drop]
    log.update({"maxi_orbits": n0, "maxi_hard_isolated_removed": int(drop.sum()),
                "maxi_hard_but_kept_not_isolated": int((hard & ~quiet_nb).sum()), "maxi_kept": len(M)})
    return pd.DataFrame({"mjd": M[:, 0], "rate": (M[:, 3] + M[:, 5]) / crab * 1e3,
                         "err": np.hypot(M[:, 4], M[:, 6]) / crab * 1e3})


# same inverse-variance binning as the main Aql X-1 code
def bin_uniform(df, edges):
    idx = np.digitize(df.mjd.values, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    w = 1 / df.err.values[ok] ** 2
    n = len(edges) - 1
    sw = np.bincount(idx[ok], w, n)
    sr = np.bincount(idx[ok], w * df.rate.values[ok], n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sw > 0, sr / sw, np.nan), np.where(sw > 0, 1 / np.sqrt(sw), np.nan)


#                     process one star                     #
# clean, grid, label, window and features for one star
def process(name, crab):
    asm_f, maxi_f, chi_cut, kind = C.SOURCES[name]
    log = {"type": kind}
    asm = load_asm(asm_f, chi_cut, crab["ASM_cps"], log)
    maxi = load_maxi(maxi_f, crab["MAXI_2_10keV_ph_cm2_s"], log)
    edges = np.arange(np.floor(asm.mjd.min()), np.ceil(maxi.mjd.max()) + C.BIN_DAYS, C.BIN_DAYS)
    tc = edges[:-1] + C.BIN_DAYS / 2
    fa, ea = bin_uniform(asm, edges)
    fm, em = bin_uniform(maxi, edges)
    use_m = tc >= C.MAXI_START
    flux, err = np.where(use_m, fm, fa), np.where(use_m, em, ea)
    inst = np.where(use_m, "MAXI", "ASM").astype(object)
    if name in C.BAT_OVERRIDE:
        bfile, lo, hi = C.BAT_OVERRIDE[name]
        fb, eb = bin_uniform(load_bat(bfile, crab["BAT_15_50keV_cts_cm2_s"], log), edges)
        ub = (tc >= lo) & (tc < hi)
        flux, err = np.where(ub, fb, flux), np.where(ub, eb, err)
        inst[ub] = "BAT"
        log["bat_override_mjd"] = [lo, hi]
    inst = inst.astype(str)
    s = pd.Series(flux).rolling(C.ACTIVE_SMOOTH, center=True, min_periods=3).mean().values
    active = np.where(np.isfinite(s), (s > C.ACTIVE_MCRAB).astype(float), np.nan)
    lc = pd.DataFrame({"mjd": tc, "instrument": inst, "mcrab": flux, "err_mcrab": err, "smooth": s, "active": active})
    lc.to_csv(C.PROC / f"lc_{name.replace(' ', '_')}.csv", index=False)

    # sanity checks, quiet-time noise and the list of active episodes
    for k in ("ASM", "MAXI", "BAT"):
        q = (inst == k) & (active == 0) & np.isfinite(flux)
        z = flux[q] / err[q]
        log[f"quiet_{k}"] = {"bins": int(q.sum()), "median_mcrab": float(np.median(flux[q])) if q.any() else None,
                             "std_z": float(np.std(z)) if q.any() else None}
    idx = np.where(np.nan_to_num(active) > 0)[0]
    eps = []
    if len(idx):
        grp = np.cumsum(np.r_[True, np.diff(idx) > 4])
        for g in np.unique(grp):
            ii = idx[grp == g]
            if len(ii) >= 6:
                eps.append([float(tc[ii[0]]), float(tc[ii[-1]]), float(np.nanmax(flux[ii[0]:ii[-1] + 1]))])
    log["episodes_ge3d"] = eps
    log["active_fraction"] = float(np.nanmean(active))

    # windows (same rules as the main Aql X-1 code)
    X, E, T0, INS, AF = [], [], [], [], []
    for i in range(0, len(flux) - C.WIN + 1, C.STRIDE):
        sl = slice(i, i + C.WIN)
        if len(set(inst[sl])) > 1:
            continue
        m = np.isfinite(flux[sl])
        if m.mean() < C.MIN_FILL:
            continue
        x = np.arange(C.WIN)
        X.append(np.interp(x, x[m], flux[sl][m]) / 1e3)                 # Crab units
        E.append(np.where(m, err[sl], np.nanmedian(err[sl])) / 1e3)
        a = active[sl][m & np.isfinite(active[sl])]
        T0.append(tc[i]); INS.append(inst[i]); AF.append(a.mean() if len(a) else 0.0)
    X, E = np.array(X), np.array(E)
    y = (np.array(AF) >= C.LABEL_FRAC).astype(np.int32)

    # the three feature sets
    P = np.asarray(pds_batch(jnp.asarray(X)))
    Pm = np.asarray(pds_batch(jnp.asarray(X - X.mean(1, keepdims=True))))[:, 1:]
    feats = {"full": np.log10(P + 1e-12),
             "noise": np.log10(Pm / (C.WIN * np.mean(E**2, 1, keepdims=True)) + 1e-6),
             "slope": np.log10(Pm / Pm.sum(1, keepdims=True) + 1e-12)}
    np.savez_compressed(C.PROC / f"ds_{name.replace(' ', '_')}.npz", flux=X, err=E, y=y, t_start=np.array(T0),
                        instrument=np.array(INS), active_frac=np.array(AF), window_mean=X.mean(1),
                        **{f"X_{k}": v.astype(np.float32) for k, v in feats.items()})
    log["windows"] = {"n": int(len(y)), "active": int(y.sum()),
                      **{k.lower(): int((np.array(INS) == k).sum()) for k in ("ASM", "MAXI", "BAT")}}
    return log


#                        main code                         #
if __name__ == "__main__":
    dl = download()
    crab = crab_levels()
    out = {"crab": crab, "aql_check_maxi_per_asm_from_crab": crab["MAXI_2_10keV_ph_cm2_s"] / crab["ASM_cps"],
           "active_threshold_mcrab": C.ACTIVE_MCRAB, "downloads": dl, "sources": {}}
    for name in C.SOURCES:
        out["sources"][name] = process(name, crab)
    (C.REPORTS / "ms_data_log.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(crab), "MAXI/ASM from Crab:", round(out["aql_check_maxi_per_asm_from_crab"], 4))
    for name, lg in out["sources"].items():
        print(f"\n== {name} ({lg['type']}) ASM {lg['asm_dwells']}->{lg['asm_kept']}  MAXI {lg['maxi_orbits']}->{lg['maxi_kept']} "
              f"(isolated hard hits removed {lg['maxi_hard_isolated_removed']}, hard but kept {lg['maxi_hard_but_kept_not_isolated']})")
        print(f"   quiet ASM {lg['quiet_ASM']}  quiet MAXI {lg['quiet_MAXI']}  active frac {lg['active_fraction']:.3f}")
        print(f"   windows {lg['windows']}  episodes>=3d: {len(lg['episodes_ge3d'])}")
        for a, b, pk in lg["episodes_ge3d"]:
            if name in ("V404 Cyg", "GX 339-4") and pk > 100:
                print(f"     MJD {a:.0f}-{b:.0f} ({b-a:.0f} d) peak {pk:.0f} mCrab")
