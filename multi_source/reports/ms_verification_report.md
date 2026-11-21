# Multi-source verification

| Check | Result | Evidence |
|---|---|---|
| Crab calibration matches known instrument levels | PASS | ASM 75.55 c/s, MAXI 2-10 keV 3.306 ph/cm2/s, BAT 0.2181 c/cm2/s |
| Crab-based ASM->MAXI factor agrees with the Aql overlap fit (0.0460) | PASS | 0.0438 (-4.9%) |
| Aql X-1: right FITS object, windows built | PASS | OBJECT=aqlx1, windows {'n': 2414, 'active': 752, 'asm': 986, 'maxi': 1428, 'bat': 0} |
| GX 339-4: right FITS object, windows built | PASS | OBJECT=gx339-4, windows {'n': 2392, 'active': 887, 'asm': 891, 'maxi': 1501, 'bat': 0} |
| V404 Cyg: right FITS object, windows built | PASS | OBJECT=gs2023+338, windows {'n': 2060, 'active': 11, 'asm': 1185, 'maxi': 816, 'bat': 59} |
| Sco X-1: right FITS object, windows built | PASS | OBJECT=scox1, windows {'n': 2460, 'active': 2460, 'asm': 985, 'maxi': 1475, 'bat': 0} |
| Quiescent noise consistent with counting noise (std z < 2) for all transients | PASS | Aql X-1 ASM 1.48, Aql X-1 MAXI 1.60, GX 339-4 ASM 1.59, GX 339-4 MAXI 1.74, V404 Cyg ASM 1.17, V404 Cyg MAXI 1.17 |
| V404 Cyg June 2015 outburst recovered (via BAT) | PASS | episodes [[57188, 57201, 4515]] |
| Sco X-1 labelled active throughout (persistent) | PASS | active fraction 1.0 |
| Slope features really are normalised (power sums to 1) | PASS | row sums [1. 1. 1.] |
| All leave-one-source-out runs and feature sets completed | PASS | 4 runs x 3 feature sets |
| Full-PDS model transfers to unseen sources (AUC > 0.95 everywhere) | PASS | see ms_results.json |
| Slope-only failure on V404 holds across seeds | PASS | V404 AUC over seeds [0.53, 0.32, 0.53] |
| All five figures written | PASS | ms_fig1_lightcurves_mcrab.png, ms_fig2_spectra_by_source.png, ms_fig3_cross_source_results.png, ms_fig4_v404_2015.png, ms_fig5_fine_tuning.png |
| Original Aql-only results untouched | PASS | main metrics.json not rewritten by the multi-source run (full-PDS test AUC 0.989) |

**15/15 checks pass.**
