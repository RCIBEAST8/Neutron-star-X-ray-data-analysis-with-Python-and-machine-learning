# Aql X-1 X-ray analysis
# shared settings, every other script imports this

# code breakdown
# folder paths, where each light curve downloads from, and every number the cleaning, chunking and
# train/val/test split use, so they only ever get changed in one place

# libs
from pathlib import Path

# folders (made automatically if they don't exist yet)
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
for p in (RAW, PROC, MODELS, FIGS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

#                       data sources                       #
SOURCES = {
    "rxte_asm": ("https://heasarc.gsfc.nasa.gov/FTP/xte/data/archive/ASMProducts/"
                 "definitive_1dwell/lightcurves/xa_aqlx1_d1.lc", RAW / "rxte_asm_aqlx1_dwell.lc"),
    "maxi": ("http://maxi.riken.jp/star_data/J1911+005/J1911+005_g_lc_1orb_all.dat",
             RAW / "maxi_aqlx1_1orbit.dat"),
    "swift_bat": ("https://swift.gsfc.nasa.gov/results/transients/AqlX-1.orbit.lc.txt",
                  RAW / "swift_bat_aqlx1_1orbit.txt"),
}
ASM_MJDREF = 49353.0          # ASM time zero, from the FITS header (MJDREFI, TIMEUNIT = d)

#                         cleaning                         #
ASM_MAX_RDCHI = 1.5           # ASM team's own recommendation, dwells with reduced chi2 over 1.5 get dropped
ASM_MIN_EXPO = 30.0           # same for exposures under 30 s
MAXI_HARD_SIG = 5.0           # MAXI orbit counts as a background hit if 10-20 keV is over 5 sigma
MAXI_HARD_FRAC = 0.5          # and it's over 50% of the 2-20 keV rate (basically particle hits)
BIN_DAYS = 0.5                # grid size in days, tried 0.25 d first but only 60% of ASM bins had data
SWITCH_MJD = 55054.0          # first MAXI orbit, ASM before this and MAXI from here on
ACTIVE_SMOOTH = 5             # bins in the running mean for each bin's state (5 x 0.5 d = 2.5 d)
ACTIVE_THRESH = 0.08          # 2-10 keV in MAXI units (ph/cm2/s), above this a bin counts as active

#                         chunking                         #
WIN = 128                     # points per window (128 x 0.5 d = 64 days)
STRIDE = 8                    # new window every 8 bins (4 days)
MIN_FILL = 0.70               # windows more than 30% empty get thrown away
LABEL_FRAC = 0.25             # window counts as an outburst if at least 25% of its bins are active

#                 train / val / test split                 #
# split by time not randomly, neighbouring windows overlap by 60 days so a random split
# would basically put the same data in train and test. one window of gap either side of each boundary too
VAL_START = 57800.0           # Feb 2017
TEST_START = 59200.0          # Dec 2020
PURGE_DAYS = WIN * BIN_DAYS

SEED = 0
