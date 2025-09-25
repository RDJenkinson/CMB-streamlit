# CMB_streamlit.py
import os
import sys
import numpy as np
import camb
from camb import model
import matplotlib
matplotlib.use("Agg")  # headless backend suitable for Streamlit
import matplotlib.pyplot as plt
from astropy.io import fits
import streamlit as st
import requests

# ----------------------
# Defaults
# ----------------------
DEFAULTS = {
    "H0": 67.4,
    "ombh2": 0.0224,
    "omch2": 0.1200,
    "Omega_L": 0.6862,
    "lensed": False,
}

# ----------------------
# Helper: base path (works inside PyInstaller or normal)
# ----------------------
if getattr(sys, "frozen", False):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# ----------------------
# Planck FITS filename and fallback download URL
# ----------------------
PLANCK_FILENAME = "COM_PowerSpect_CMB_R2.02.fits"
PLANCK_PATH = os.path.join(BASE_PATH, PLANCK_FILENAME)
PLANCK_FITS_URL = "https://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/COM_PowerSpect_CMB_R2.02.fits"

def ensure_planck_fits(local_path=PLANCK_PATH, url=PLANCK_FITS_URL):
    """Ensure the Planck FITS exists locally. If not, attempt to download it."""
    if os.path.exists(local_path):
        return local_path
    # try to download
    try:
        st.info("Planck FITS not found locally. Attempting download from IRSA...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success("Downloaded Planck FITS to app folder.")
        return local_path
    except Exception as e:
        st.error("Could not download Planck FITS automatically.")
        st.write("Please download the file manually from the Planck release and place it in the app folder.")
        st.write("Planck data URL:", url)
        raise

@st.cache_data(show_spinner=False)
def load_planck_binned_tt(fits_path):
    """Load TTHILBIN extension from the Planck FITS file."""
    hdul = fits.open(fits_path)
    if "TTHILBIN" not in hdul:
        st.error("Planck FITS file does not contain expected 'TTHILBIN' extension.")
        st.write(hdul.info())
        raise RuntimeError("TTHILBIN not found in FITS file.")
    tt = hdul["TTHILBIN"].data
    ell = np.asarray(tt["ELL"], dtype=float)
    D_ell = np.asarray(tt["D_ELL"], dtype=float)
    err = np.asarray(tt["ERR"], dtype=float)
    hdul.close()
    return ell, D_ell, err

# ----------------------
# CAMB model (cached)
# ----------------------
@st.cache_data(show_spinner=False)
def get_tt_spectrum_cached(H0, ombh2, omch2, omk, lensed):
    """Return (ls, Cl_TT) computed by CAMB — cached for interactivity."""
    pars = camb.set_params(
        H0=float(H0),
        ombh2=float(ombh2),
        omch2=float(omch2),
        mnu=0.0,
        omk=float(omk),
        tau=0.054,
        As=2.1e-9,
        ns=0.965,
        lmax=2500,
        DoLensing=bool(lensed),
    )
    pars.NonLinear = model.NonLinear_none
    # speed/accuracy tradeoff for interactivity
    pars.AccuracyBoost = 0.5
    pars.lAccuracyBoost = 0.5
    pars.lSampleBoost = 0.5

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    if lensed:
        CL = powers["total"][:, 0]  # TT lensed
    else:
        CL = powers["unlensed_scalar"][:, 0]
    ls = np.arange(CL.shape[0])
    return ls, CL

# ----------------------
# Derived Omegas helper
# ----------------------
def compute_omegas(H0, ombh2, omch2, Omega_L, mnu=0.0):
    h = H0 / 100.0
    Omega_b = ombh2 / (h**2)
    Omega_cdm = omch2 / (h**2)
    Omega_m = Omega_b + Omega_cdm
    Omega_nu = mnu / (93.14 * h**2) if mnu > 0 else 0.0
    Omega_k = 1.0 - (Omega_b + Omega_cdm + Omega_nu + Omega_L)
    return Omega_k, Omega_b, Omega_cdm, Omega_m, Omega_L, h

# ----------------------
# Streamlit UI: initialize session state defaults if missing
# ----------------------
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.set_page_config(page_title="Interactive CMB (CAMB)", layout="wide")
st.title("Interactive CMB TT spectrum (CAMB)")

# Sidebar: parameter controls with LaTeX labels above sliders
st.sidebar.header("Cosmological parameters")

st.sidebar.markdown("**Hubble parameter**")
st.sidebar.latex(r"H_0 \; [\mathrm{km\,s^{-1}\,Mpc^{-1}}]")
H0 = st.sidebar.slider("", min_value=50.0, max_value=80.0, value=st.session_state["H0"],
                       step=0.1, key="H0", format="%.1f")

st.sidebar.markdown("**Baryon density**")
st.sidebar.latex(r"\Omega_b h^2")
ombh2 = st.sidebar.slider("", min_value=0.005, max_value=0.10, value=st.session_state["ombh2"],
                          step=0.0001, key="ombh2", format="%.4f")

st.sidebar.markdown("**Cold dark matter density**")
st.sidebar.latex(r"\Omega_{c} h^2")
omch2 = st.sidebar.slider("", min_value=0.01, max_value=0.40, value=st.session_state["omch2"],
                          step=0.0005, key="omch2", format="%.4f")

st.sidebar.markdown("**Dark energy**")
st.sidebar.latex(r"\Omega_\Lambda")
Omega_L = st.sidebar.slider("", min_value=0.0, max_value=0.98, value=st.session_state["Omega_L"],
                            step=0.001, key="Omega_L", format="%.4f")

st.sidebar.write("")  # small gap
lensed = st.sidebar.checkbox("Show lensed spectrum (slower)", value=st.session_state["lensed"], key="lensed")

# Reset button
if st.sidebar.button("Reset to defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Notes**\n\nSliders show parameter values directly. Use the lensed checkbox to compare to Planck (slow).")

# ----------------------
# Load Planck data
# ----------------------
try:
    planck_file = ensure_planck_fits()
    ell_planck, D_planck, err_planck = load_planck_binned_tt(planck_file)
except Exception:
    st.stop()

# ----------------------
# Compute derived quantities and model
# ----------------------
omk, Omega_b, Omega_cdm, Omega_m, Omega_L_used, h_used = compute_omegas(H0, ombh2, omch2, Omega_L)

with st.spinner("Computing CAMB spectrum (cached)..."):
    ls_model, cl_tt_model = get_tt_spectrum_cached(H0, ombh2, omch2, omk, lensed)

# ----------------------
# Plot (NO conversion between D and C; plot Planck data values directly)
# ----------------------
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

ax.plot(ls_model, cl_tt_model, color="orange", linewidth=1.6, label="CAMB model (C_ell)")

# Plot Planck binned values directly against model (no conversion)
ax.errorbar(ell_planck, D_planck, yerr=err_planck, fmt="o", markersize=4,
            color="white", ecolor="lightgrey", capsize=2, label="Planck 2018 (binned)")

ax.set_xlim(2, 2000)
ax.set_xlabel(r"$\ell$", color="white")
ax.set_ylabel(r"$C_\ell\; [\mu\mathrm{K}^2]$", color="white")
ax.set_title("CMB TT Power Spectrum — model vs Planck (no conversion)", color="white")

ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

# Summary box on right (derived values only)
col1, col2 = st.columns([3, 1])
with col1:
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.markdown("### Derived quantities")
    st.write(f"H₀ = {H0:.2f} km/s/Mpc  (h = {h_used:.4f})")
    st.write(f"Ω_m (matter) = {Omega_m:.5f}")
    st.write(f"Ω_k (curvature) = {omk:.6f}")
    st.write("")
    st.markdown("**Model options**")
    st.write(f"Lensed: {'Yes' if lensed else 'No'}")
    st.write("CAMB accuracy reduced for speed / caching is used for interactivity.")

st.markdown("---")
st.markdown("**Footer:** This app uses CAMB to compute the CMB power spectrum. For speed we reduce some accuracy settings; enable lensed for a closer match to Planck (slower).")


