# Aql X-1 X-ray analysis
# Fourier features for every 64 day window

# code breakdown
# runs jax.numpy.fft.rfft on every window (vmapped, and checked against numpy), squares it to get the power
# density spectrum (PDS = |FFT|^2), then builds the dataset, X = log power per frequency, y = 0 quiet / 1 outburst
#
# saves two feature sets, because the labels come from flux and flux leaks straight into the spectrum
#   "full"   log10 PDS of the raw window with the DC bin kept. the obvious version, but the DC bin is just
#            (window sum)^2 so it basically hands the model the label
#   "shape"  DC removed and each PDS divided by its own expected white-noise power N*<sigma^2>. this takes out
#            the instrument noise difference (ASM is ~1.6x noisier than MAXI), so the model has to use how much
#            the window varies, and on what timescales, against its own noise, not just the raw flux level

# libs
import json

import jax
import jax.numpy as jnp
import numpy as np

import config as C

jax.config.update("jax_enable_x64", True)   # 64 bit floats, same precision as numpy


# FFTs every window in one go (vmap) and squares it, |FFT|^2 = power
@jax.jit
def pds_batch(x):
    F = jax.vmap(jnp.fft.rfft)(x)
    return jnp.abs(F) ** 2


# train / val / test split by time, with one window of gap at each boundary so nothing leaks across
# (0 = train, 1 = val, 2 = test, -1 = in a gap, not used)
def split_ids(t0):
    t1 = t0 + C.WIN * C.BIN_DAYS
    train = t1 <= C.VAL_START - C.PURGE_DAYS
    val = (t0 >= C.VAL_START) & (t1 <= C.TEST_START - C.PURGE_DAYS)
    test = t0 >= C.TEST_START
    return np.where(train, 0, np.where(val, 1, np.where(test, 2, -1)))


#                        main code                         #
if __name__ == "__main__":
    w = np.load(C.PROC / "windows.npz")
    x, err, y, t0 = w["flux"], w["err"], w["label"], w["t_start"]
    n = x.shape[1]

    # raw windows, DC bin kept
    P_full = np.asarray(pds_batch(jnp.asarray(x)))
    ref = np.abs(np.fft.rfft(x, axis=1)) ** 2
    fft_ok = bool(np.allclose(P_full, ref, rtol=1e-9, atol=1e-12))

    # mean taken off first, then divided by the noise level
    xm = x - x.mean(axis=1, keepdims=True)
    P_shape = np.asarray(pds_batch(jnp.asarray(xm)))[:, 1:]
    noise = n * np.mean(err**2, axis=1, keepdims=True)    # what pure white noise would give per bin
    P_shape = P_shape / noise

    freq = np.fft.rfftfreq(n, d=C.BIN_DAYS)               # cycles per day (tops out at 1 per day because of the 0.5 d bins)
    X_full = np.log10(P_full + 1e-12).astype(np.float32)
    X_shape = np.log10(P_shape + 1e-6).astype(np.float32)
    split = split_ids(t0)

    np.savez_compressed(C.PROC / "dataset.npz", X_full=X_full, X_shape=X_shape, y=y.astype(np.int32),
                        split=split, t_start=t0, freq_full=freq, freq_shape=freq[1:],
                        instrument=w["instrument"], window_mean=x.mean(axis=1))

    names = {0: "train", 1: "val", 2: "test", -1: "purged"}
    info = {"fft_matches_numpy": fft_ok, "n_windows": len(y),
            "X_full_shape": list(X_full.shape), "X_shape_shape": list(X_shape.shape),
            "freq_range_per_day": [float(freq[1]), float(freq[-1])],
            "nyquist_per_day": float(freq[-1]),
            "splits": {names[s]: {"n": int((split == s).sum()), "active": int(y[split == s].sum())}
                       for s in (0, 1, 2, -1)}}
    (C.REPORTS / "fourier_features_log.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
