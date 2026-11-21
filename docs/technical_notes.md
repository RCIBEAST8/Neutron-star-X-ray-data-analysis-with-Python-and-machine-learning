# Aql X-1: Fourier features + a Flax neural net

This is the full technical write-up of the Python side. The short version is in the [README](../README.md). File paths below are relative to the top of the repo.

Aql X-1 is a neutron star low-mass X-ray binary that goes into outburst roughly once a year. This project takes 30 years of its X-ray light curve, chops it into 64 day windows, turns each window into a power spectrum with JAX, and trains a small Flax network to tell quiescence from outburst. It also tries (and fails, more on that below) to predict an outburst before it starts.

Everything runs on CPU in about 20 seconds once the data is downloaded.

## Data

- RXTE/ASM dwell-by-dwell light curve, 1.5-12 keV, Jan 1996 to Sep 2011. FITS file straight from the HEASARC archive.
- MAXI/GSC 1-orbit light curve, Aug 2009 to Oct 2026, from RIKEN. I use 2-10 keV (the 2-4 and 4-10 keV bands added) because it's closest to the ASM band and skips the noisy 10-20 keV band.
- Swift/BAT 1-orbit light curve, downloaded for reference but not used in the model (Aql X-1 is mostly too soft for it).

URLs and SHA-256 checksums for every file are in `reports/importing_data_log.json`.

## How it works

- `importing_data.py` downloads the three light curves and cleans them. ASM dwells with reduced chi2 > 1.5 or exposure < 30 s get dropped (the ASM team's own recommendation), MAXI orbits with zero errors or a 10-20 keV-only spike (particle background hits) get dropped, then everything is binned onto a uniform 0.5 day grid. I tried 0.25 d first but only 60% of ASM bins had data. Both rates are already background subtracted by each instrument team's own processing so I don't subtract anything else. ASM is put on the MAXI scale using the 2009-2011 overlap (1 ASM count/s = 0.046 MAXI ph/cm2/s, fit through the origin, see `figures/supp_cross_calibration.png`), and flux is normalised so the big outburst peaks sit around 1.

  The merged timeline is cut into 128-point windows (64 days) every 4 days. Windows that are more than 30% empty or straddle the ASM/MAXI switch are thrown away, small gaps inside a window are filled linearly. That leaves 2026 windows.
- `fourier_features.py` runs `jax.numpy.fft.rfft` on every window (vmapped, and checked against numpy) and squares it to get the power spectrum, 65 frequencies from 1/64 to 1 cycle per day. X = log10 power per frequency, y = 1 if at least 25% of the window is in outburst, else 0. A bin counts as "in outburst" if its 2.5 day running mean is above 0.08 ph/cm2/s (2-10 keV).
- `neural_network.py` is the network, `SpectrumMLP`, a Flax Linen module: Dense 128 > ReLU > Dropout > Dense 64 > ReLU > Dense 2, 16.7k parameters. The forward pass returns logits and softmax gives the class probabilities. Training uses `optax.softmax_cross_entropy` on the logits because it's more numerically stable than log(softmax).
- `training.py` trains it with Adam (lr 3e-4), L2 weight decay and class-weighted cross-entropy, since there are about 3 quiet windows for every active one. The train step uses `jax.value_and_grad` inside `@jax.jit`, so 200 epochs of shuffled minibatches take about 4 seconds. I keep the epoch with the lowest validation loss.
- `testing_and_plots.py` scores everything on a test set the model never saw and makes the plots in `figures/`.

`scripts/checks.py` checks each stage against the real outputs and writes `reports/verification_report.md`.

## How I split the data

Splitting randomly would cheat, since neighbouring windows overlap by 60 days. So it's split by time: train 1996 to early 2017, validation 2017 to 2020, test Dec 2020 to Oct 2026. I also drop any window within 64 days of a boundary so nothing leaks across.

What actually matters is how many separate outbursts each split has, since the windows overlap and one outburst shows up in lots of them. Train has 29 outburst onsets (16 big ones), validation 6 (5 big), test 7 (6 big).

## Results on the test set

207 windows, 48 of them active.

| Model | Accuracy | Balanced accuracy | ROC AUC |
|---|---|---|---|
| MLP on the full power spectrum | 0.899 | 0.919 | 0.989 |
| MLP on the noise-normalised spectrum (DC removed) | 0.797 | 0.861 | 0.976 |
| Baseline: threshold on the window's mean flux | 0.903 | 0.937 | 0.995 |

The honest bit: all three are basically tied, and the one-line baseline is just as good as the network. With only about 6 independent outbursts in the test set, the gaps between 0.976, 0.989 and 0.995 AUC are well within the noise, so I can't rank them. That's not a bug either. The labels are defined from flux, and the zero-frequency bin of the FFT basically is the window's total flux, so any model with access to it has the answer handed to it.

I also trained a "shape" version that throws away the DC bin and divides every spectrum by its own expected white-noise level. That removes the difference between the two instruments (ASM is about 1.6x noisier than MAXI) but it doesn't remove flux, because the variance of an outburst window still grows with how bright the outburst is. So its 0.976 AUC mostly means "this window varies way more than its noise", which is still flux in disguise, not proof that the spectral shape alone does the job. A proper shape-only test divides each spectrum by its own total power so only the slope is left. I ran that later across several stars, see the multi-source section below.

Most of the "false positives" in `fig4` line up with small flares (like mid 2022) that don't reach the 25% rule, so they're partly a labelling threshold thing rather than the model being wrong.

Training loss without weight decay went to almost zero while validation loss climbed from epoch 8 onward, a classic overfit on only 1.5k windows. Adding L2 decay and dropout 0.5 (chosen on the validation set only) fixed that, you can see the flat validation curves in `fig2`.

## Predicting an outburst before it starts

`forecasting.py` takes only fully quiet 64 day windows and asks whether an outburst starts in the next 14 days. It doesn't work. Test AUC is 0.44 for the network and 0.39 for a "flux over the last 4 days" baseline, which is coin flipping. There are only 34 onsets in training and 6 in test, and the X-ray light curve in quiescence is basically noise at this cadence, so there's nothing in it to learn from. Physically that makes sense. Optical monitoring regularly catches these outbursts rising before the X-ray all-sky monitors see anything, and disc instability models predict the X-rays lag the optical by several days (Russell et al. 2019, arXiv:1903.04519). So if I wanted to actually forecast these I'd need ZTF optical data in the input, not just X-rays.

## Limits

- The bins are 0.5 days, so the highest frequency in the spectrum is 1 cycle per day. The 550 Hz spin (Casella et al. 2008, arXiv:0708.1110), kHz QPOs and the 18.95 hour orbit are all way out of reach. Those need NICER event files at sub-millisecond resolution, which is a different project.
- The ASM to MAXI scale factor fit has a reduced chi2 of 2.7 because the two energy bands aren't identical. Good enough for labelling states, not for precise fluxes.
- A couple of short MAXI episodes near solar conjunction (end of Dec 2024 and end of Dec 2025) still get labelled active. They're hard, background-like events rather than real outbursts.

## Running it

```bash
uv venv ~/venvs/jax --python 3.12
uv pip install --python ~/venvs/jax/bin/python jax flax optax matplotlib pandas astropy
bash run_all.sh
```

Versions used: jax 0.11.2, flax 0.12.10, optax 0.2.8, astropy 8.0.1, CPU only.

## Folder layout

- `data/raw/` the downloaded light curves (not in the repo, `run_all.sh` downloads them)
- `data/processed/` cleaned 0.5 d light curve, the windows, and the X/y dataset
- `models/` trained parameters, normalisation stats, loss histories
- `figures/` all plots
- `reports/` logs, metrics and the verification report
- `scripts/` the code

## Extending it to other X-ray binaries (`multi_source/`)

I wanted to see if the model works on other stars, not just Aql X-1, so `multi_source/` adds GX 339-4 (black hole transient), V404 Cyg (black hole transient) and Sco X-1 (neutron star, but persistent, it's never quiet). The original Aql results above aren't touched by any of this, the new stuff has its own data, models and figures.

What changed to make the sources comparable:

- Everything is in Crab units now. I measured each instrument's Crab level from the Crab's own light curves (ASM 75.55 c/s, MAXI 2-10 keV 3.306 ph/cm2/s, BAT 0.218 c/cm2/s) instead of using the Aql-specific overlap fit. As a check the Crab method gives an ASM to MAXI factor of 0.0438, within 5% of the 0.046 I fitted for Aql.
- The "active" threshold is the Aql one converted to Crab units, 24 mCrab, for every star.
- Sco X-1 gets the ASM team's looser chi2 cut (8 instead of 1.5) because it's so bright.
- The MAXI hard-spike filter only removes isolated single orbits now, so it can't eat real hard-state outburst data from GX 339-4 or V404.
- MAXI's public V404 light curve has the June 2015 outburst cut out (it hit about 37 Crab), so for MJD 57150-57450 I swap in Swift/BAT 15-50 keV. That's a different energy band, so take V404's outburst windows as a rough test.
- A window counts as active if 15% of it is active (was 25%), otherwise V404's 13 day outburst never gets labelled. Windows need 60% data coverage (was 70%) because MAXI only fills about 62% of V404's bins.

I also added a third feature set, "slope", where every power spectrum is divided by its own total power. That throws away the flux AND the amount of variability, so the only thing left is the shape of the spectrum. This is the test the Aql-only section above said I hadn't run.

Then I trained on some stars and tested on ones the model had never seen (`ms_experiments.py`).

| Trained on | Tested on | full PDS | noise-normalised | slope only | mean-flux baseline |
|---|---|---|---|---|---|
| Aql X-1 | GX 339-4 | 0.991 | 0.841 | 0.825 | 0.996 |
| Aql X-1 | V404 Cyg | 1.000 | 0.994 | 0.414 | 1.000 |
| Aql X-1 + GX 339-4 | V404 Cyg | 1.000 | 1.000 | 0.477 | 1.000 |
| Aql X-1 + V404 Cyg | GX 339-4 | 0.989 | 0.882 | 0.840 | 0.996 |
| GX 339-4 | Aql X-1 | 0.976 | 0.910 | 0.934 | 0.989 |

(ROC AUC on the unseen star.) Sco X-1 has no quiet windows so AUC doesn't exist for it, instead I checked what fraction of its windows each model calls active (should be all of them): full PDS 100%, noise-normalised 80-100% (depends on the seed and which stars it trained on), slope only 1-6%.

So, honestly:

- The model does "adapt" to new stars, but only because flux in Crab units means the same thing for every source. The full-PDS model gets 0.98-1.00 on stars it never saw, and the one-line mean-flux threshold still matches or beats it every time.
- The spectral shape on its own only partly carries over. Between Aql X-1 and GX 339-4 the slope-only model gets 0.82-0.93, which makes sense since both have month-long outbursts with a slow rise and decay, so the steep red-noise slope looks similar. On V404 it's at chance (0.41-0.48), because V404's outburst was one 13 day spike measured in a different band, nothing like what it trained on. It's never better than the flux threshold either. It also barely trains, its best validation epoch is the first one every time, and three more seeds gave the same thing (`reports/ms_seed_check.json`), so it's not bad luck.
- Slope isn't a perfectly clean shape test either. A window that's mostly white noise comes out flat and one with a big trend comes out steep, so signal-to-noise still leaks in a bit.
- Sco X-1 getting called quiet 94-99% of the time by the slope model isn't really the model being wrong. Once you scale out the total power, Sco X-1's spectrum is as flat as quiescent noise (`ms_fig2`) because it never has an outburst profile. It's only "active" by my flux-based label, and that label just doesn't suit a persistent source.
- Fine-tuning the Aql model on GX 339-4's first half and testing on its second half didn't help. For the full-PDS model the validation set rejected every update so it stayed exactly the same (0.989, it already worked). For the noise and slope models there was room to improve (0.86 and 0.89) but AUC didn't move and balanced accuracy actually dropped (0.77 to 0.73, 0.80 to 0.70), and training from scratch on early GX 339-4 did a bit better (0.89, 0.90).
- V404 is a thin test. All 11 of its active windows come from one outburst, and that outburst is in BAT's band, not MAXI's.

So at 0.5 day sampling, flux does almost all the work. Shape helps a bit when the new star's outbursts look like the training stars', not at all when they don't, and the network never beats a flux threshold. To get something the network can actually learn that a threshold can't, I'd need either faster timing data (NICER, where QPOs and band-limited noise differ between hard and soft states) or a harder question than quiet vs outburst, like hard vs soft state from MAXI's energy bands.

Run it with `bash multi_source/run_multi_source.sh` (about 5 minutes, it downloads any missing light curves itself, URLs and checksums end up in `multi_source/reports/ms_data_log.json`). Checks are in `multi_source/reports/ms_verification_report.md` (15/15 pass).
