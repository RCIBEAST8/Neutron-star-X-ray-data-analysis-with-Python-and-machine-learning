# multi-source extension, settings
# Aql X-1, GX 339-4, V404 Cyg and Sco X-1

# code breakdown
# reuses the main Aql X-1 code in ../../scripts for the FFT, model and loss, and keeps all of its own data,
# models and figures inside multi_source/ so the original results don't get touched

# libs
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "scripts"))          # so fourier_features, neural_network etc. import from the main scripts folder

# folders
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"

SOURCES = {
    #  name        ASM file                                            MAXI file                                              ASM chi2 cut  type
    "Aql X-1":  (PROJECT / "data/raw/rxte_asm_aqlx1_dwell.lc", PROJECT / "data/raw/maxi_aqlx1_1orbit.dat", 1.5, "NS transient"),
    "GX 339-4": (RAW / "asm_gx339-4.lc",                        RAW / "maxi_gx339-4.dat",                    1.5, "BH transient"),
    "V404 Cyg": (RAW / "asm_v404cyg.lc",                        RAW / "maxi_v404cyg.dat",                    1.5, "BH transient"),
    "Sco X-1":  (RAW / "asm_scox1.lc",                          RAW / "maxi_scox1.dat",                      8.0, "NS persistent Z source"),
}
_ASM = "https://heasarc.gsfc.nasa.gov/FTP/xte/data/archive/ASMProducts/definitive_1dwell/lightcurves/"
_MAXI = "http://maxi.riken.jp/star_data/"
_BAT = "https://swift.gsfc.nasa.gov/results/transients/"
DOWNLOADS = {   # raw file -> source URL (Aql X-1 files come from the main data/raw folder)
    "asm_gx339-4.lc": _ASM + "xa_gx339-4_d1.lc",
    "asm_v404cyg.lc": _ASM + "xa_gs2023+338_d1.lc",
    "asm_scox1.lc": _ASM + "xa_scox1_d1.lc",
    "asm_crab.lc": _ASM + "xa_crab_d1.lc",
    "maxi_gx339-4.dat": _MAXI + "J1702-487/J1702-487_g_lc_1orb_all.dat",
    "maxi_v404cyg.dat": _MAXI + "J2024+338/J2024+338_g_lc_1orb_all.dat",
    "maxi_scox1.dat": _MAXI + "J1619-156/J1619-156_g_lc_1orb_all.dat",
    "maxi_crab.dat": _MAXI + "J0534+220/J0534+220_g_lc_1orb_all.dat",
    "bat_v404cyg_orbit.txt": _BAT + "weak/V404Cyg.orbit.lc.txt",
    "bat_crab_orbit.txt": _BAT + "Crab.orbit.lc.txt",
}
# ASM team rule, dwells with reduced chi2 over 1.5 get dropped (8.0 for Sco X-1 because it's so bright),
# same for exposures under 30 s
ASM_MIN_EXPO = 30.0
# Crab light curves, used to put every instrument in Crab units
CRAB_FILES = {"ASM": RAW / "asm_crab.lc", "MAXI": RAW / "maxi_crab.dat", "BAT": RAW / "bat_crab_orbit.txt"}

# MAXI's public V404 Cyg light curve has the June 2015 outburst cut out (gap MJD 57186-57217, it hit ~37 Crab)
# so for that stretch only V404 uses Swift/BAT 15-50 keV in BAT Crab units instead. different band, so V404's
# outburst windows are only a rough test
BAT_OVERRIDE = {"V404 Cyg": (RAW / "bat_v404cyg_orbit.txt", 57150.0, 57450.0)}

MAXI_HARD_SIG = 5.0          # only removes isolated single-orbit 10-20 keV spikes (both neighbours quiet) so real hard-state outbursts survive
MAXI_HARD_FRAC = 0.5         # and the spike is over 50% of the 2-20 keV rate
BIN_DAYS = 0.5               # same 0.5 d grid as the main Aql X-1 code
MAXI_START = 55058.0         # switch from ASM to MAXI (first full day all four MAXI files have data)
ACTIVE_SMOOTH = 5            # 2.5 d running mean for each bin's state
ACTIVE_MCRAB = 24.2          # = the 0.08 ph/cm2/s (2-10 keV) Aql X-1 threshold, in Crab units

WIN, STRIDE, MIN_FILL, LABEL_FRAC = 128, 8, 0.60, 0.15   # 0.60 not 0.70 since MAXI only fills ~62% of V404's bins, 0.15 so a ~10 d outburst (V404 2015) still counts
PURGE_DAYS = WIN * BIN_DAYS
SEED = 0

# makes the folders if they don't exist yet
for p in (RAW, PROC, MODELS, FIGS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
