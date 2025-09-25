import os, sys
import numpy as np
import camb
from camb import model
import streamlit as st
from astropy.io import fits
import matplotlib.pyplot as plt

# ------------------------------
# Defaults and Session State
# ------------------------------
DEFAULTS = {
    "ombh2": 0.0224,   # Ω_b h²
    "omch2": 0.120,    # Ω_cdm h²
    "Omega_L": 0.6862, # Ω_Λ
}

# Initialise defaults if not already in session state
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------
# Functions
# ------------------------------
def compute_omegas(H0, ombh2, omch2, Omega_L, mnu=0.0):
    h = H0 / 100.0
    Omega_b = ombh2 / h**2
    Omega_cdm = omch2 / h**2
    Omega_m = Omega_b + Omega_cdm
    Omega_nu = mnu / (93.14 * h**2) if mnu > 0 else 0.0
    Omega_k = 1.0 - (Omega_b + Omega_cdm + Omega_nu + Omega_L)
    return Omega_k, Omega_b, Omega_cdm, Omega_m, Omega_L, h

def get_tt_spectrum(H0, ombh2, omch2, omk):
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=0.0,
        omk=omk,
        tau=0.054,
        As=2.1e-9,
        ns=0.965,
        lmax=2500,
        DoLensing=False
    )
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    CL = powers["unlensed_scalar"][:, 0]
    ls = np.arange(CL.shape[0])
    return ls, CL

def reset_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# ------------------------------
# Page layout
# ------------------------------
st.set_page_config(page_title="CMB Power Spectrum Explorer", layout="wide")
st.title("CMB Power Spectrum Explorer")

st.markdown(
    """
Use the sliders to adjust cosmological parameters and compare the CAMB model with Planck 2018 TT data.  
Click **Reset to Defaults** to restore the original parameter values.
"""
)

# ------------------------------
# Planck 2018 TT binned data
# ------------------------------
base_path = os.path.dirname(__file__)
planck_fits_path = os.path.join(base_path, 'COM_PowerSpect_CMB_R2.02.fits')
hdul = fits.open(planck_fits_path)
tt = hdul['TTHILBIN'].data
ell_planck = tt['ELL']        # multipoles
D_planck = tt['D_ELL']        # D_l
err_planck = tt['ERR']        # error

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Cosmological Parameters")

ombh2 = st.sidebar.slider("Ω₍b₎ h²", 0.005, 0.1, st.session_state["ombh2"], 0.0005, key="ombh2")
omch2 = st.sidebar.slider("Ω₍cdm₎ h²", 0.01, 0.40, st.session_state["omch2"], 0.0005, key="omch2")
Omega_L = st.sidebar.slider("Ω_Λ", 0.0, 0.98, st.session_state["Omega_L"], 0.005, key="Omega_L")

if st.sidebar.button("Reset to Defaults"):
    reset_defaults()
    st.experimental_rerun()

# ------------------------------
# Compute model
# ------------------------------
H0 = 67.4  # fixed Hubble constant
omk_val, Omega_b_val, Omega_c_val, Omega_m_val, _, h_val = compute_omegas(
    H0, ombh2, omch2, Omega_L
)
ls_model, cl_tt_model = get_tt_spectrum(H0, ombh2, omch2, omk_val)

# ------------------------------
# Plotting
# ------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

ax.plot(ls_model, cl_tt_model, lw=2, color='orange', label='Model')
ax.errorbar(ell_planck, D_planck, yerr=err_planck, fmt='o', markersize=4,
            capsize=2, color='white', ecolor='lightgrey', label='Planck 2018')

ax.set_xlim([2, 2000])
ax.set_xlabel(r"$\ell$", color='white')
ax.set_ylabel(r"$C_\ell \; [\mu\mathrm{K}^2]$", color='white')
ax.set_title("TT Power Spectrum (Model vs Planck)", color='white')

ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

text_box = ax.text(
    0.98, 0.98,
    rf"$h = {h_val:.3f}$\n"
    rf"$\Omega_k = {omk_val:.4f}$\n"
    rf"$\Omega_m = {Omega_m_val:.4f}$",
    transform=ax.transAxes,
    ha='right', va='top', fontsize=11,
    color='white',
    bbox=dict(facecolor='black', alpha=0.8, edgecolor='white')
)

st.pyplot(fig)
