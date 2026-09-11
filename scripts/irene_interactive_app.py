#!/usr/bin/env python3
import os
import glob
import csv
import hashlib
from pathlib import Path

import numpy as np
import tables as tb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from invisible_cities.cities.components import build_pmap_dual_gain
from invisible_cities.cities.components import fourier_filter
from invisible_cities.cities.components import calibrate_fibers_hg
from invisible_cities.cities.components import calibrate_fibers_lg
from invisible_cities.cities.components import get_actual_sipm_thr
from invisible_cities.cities.components import zero_suppress_wfs_hg
from invisible_cities.cities.components import zero_suppress_wfs_lg
from invisible_cities.calib.calib_sensors_functions import mask_sensors
from invisible_cities.calib.calib_sensors_functions import subtract_baseline_and_calibrate
from invisible_cities.calib.calib_sensors_functions import binnedmodes
from invisible_cities.calib.calib_sensors_functions import means
from invisible_cities.calib.calib_sensors_functions import modes
from invisible_cities.database import load_db
from invisible_cities.core import system_of_units as units
from invisible_cities.core.configure import read_config_file
from invisible_cities.types.symbols import BlsMode
from invisible_cities.types.symbols import SiPMThreshold

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("ICTDIR", str(ROOT_DIR))
ANALYSIS_DIR = Path("/analysis")
SIPM_POSITIONS_CSV = ROOT_DIR / "scripts" / "hddemo_db_elecid_positions.csv"
AUTHORIZED_PASSWORD_HASH = "a2242ead55c94c3deb7cf2340bfef9d5bcaca22dfe66e646745ee4371c633fc8"

# Candidate-selection defaults loaded from the Irene config file.
CONFIG_FILE = ROOT_DIR / "invisible_cities" / "config" / "irene.conf"
CFG = read_config_file(str(CONFIG_FILE)) if CONFIG_FILE.exists() else {}

N_BASELINE_DEFAULT = int(CFG.get("n_baseline", 2800))
N_MAW_S1_DEFAULT = int(CFG.get("n_maw_s1", 10))
N_MAW_S2_DEFAULT = int(CFG.get("n_maw_s2", 1))
S1_PADDING_DEFAULT = int(CFG.get("s1_pading", 10))
FIBER_CUTOFF_MHZ_DEFAULT = float(CFG.get("fiber_cutoff_freq_MHz", 3.0))
THR_CSUM_S1_DEFAULT = float(CFG.get("thr_csum_s1", 100.0))
THR_CSUM_S2_DEFAULT = float(CFG.get("thr_csum_s2", 50.0))

S1_TMIN_US_DEFAULT = float(CFG.get("s1_tmin", 50 * units.mus)) / units.mus
S1_TMAX_US_DEFAULT = float(CFG.get("s1_tmax", 250 * units.mus)) / units.mus
S1_STRIDE_DEFAULT = int(CFG.get("s1_stride", 3))
S1_LMIN_DEFAULT = int(CFG.get("s1_lmin", 20))
S1_LMAX_DEFAULT = int(CFG.get("s1_lmax", 100))
S1_REBIN_STRIDE_DEFAULT = int(CFG.get("s1_rebin_stride", 1))

S2_TMIN_US_DEFAULT = float(CFG.get("s2_tmin", 245 * units.mus)) / units.mus
S2_TMAX_US_DEFAULT = float(CFG.get("s2_tmax", 290 * units.mus)) / units.mus
S2_STRIDE_DEFAULT = int(CFG.get("s2_stride", 2))
S2_LMIN_DEFAULT = int(CFG.get("s2_lmin", 150))
S2_LMAX_DEFAULT = int(CFG.get("s2_lmax", 100000))
S2_REBIN_STRIDE_DEFAULT = int(CFG.get("s2_rebin_stride", 40))

THR_SIPM_S2_DEFAULT = float(CFG.get("thr_sipm_s2", 1.5))
PMT_SAMP_WID_NS_DEFAULT = float(CFG.get("pmt_samp_wid", 25 * units.ns)) / units.ns
SIPM_SAMP_WID_US_DEFAULT = float(CFG.get("sipm_samp_wid", 1 * units.mus)) / units.mus
FIBER_SAMP_WID_NS_DEFAULT = float(CFG.get("fiber_samp_wid", 25 * units.ns)) / units.ns


def write_parameters_to_file(file_path, params):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    values = dict(params)

    int_keys = {
        "n_baseline",
        "n_maw_s1",
        "n_maw_s2",
        "s1_tmin",
        "s1_tmax",
        "s1_stride",
        "s1_lmin",
        "s1_lmax",
        "s1_rebin_stride",
        "s1_pading",
        "s2_tmin",
        "s2_tmax",
        "s2_stride",
        "s2_lmin",
        "s2_lmax",
        "s2_rebin_stride",
        "pmt_samp_wid",
        "fiber_samp_wid",
        "sipm_samp_wid",
    }
    float_keys = {
        "fiber_cutoff_freq_MHz",
        "thr_csum_s1",
        "thr_csum_s2",
        "thr_sipm_s2",
    }

    def format_value(key, value):
        if key in int_keys:
            return str(int(value))
        if key in float_keys:
            return repr(float(value))
        return str(value)

    template = f"""files_in = '$ICDIR/database/test_data/electrons_40keV_z25_RWF.h5'

# REPLACE /tmp with your output directory
file_out = '/tmp/electrons_40keV_z25_PMP.h5'

# compression library
compression = 'ZLIB4'

# run number 0 is for MC
run_number = 1
detector_db = 'hddemojb'

# How frequently to print events
print_mod = 1

# max number of events to run
event_range =  999

n_baseline =   {format_value('n_baseline', values.get('n_baseline', 2800))} # for a window of 800 mus

# Set MAW for calibrated sums
n_maw_s1 = {format_value('n_maw_s1', values.get('n_maw_s1', 20))}
n_maw_s2 = {format_value('n_maw_s2', values.get('n_maw_s2', 1))}
thr_maw =   0.1 * adc

fiber_cutoff_freq_MHz = {format_value('fiber_cutoff_freq_MHz', values.get('fiber_cutoff_freq_MHz', 3.0))}

# Set thresholds for calibrated sum
thr_csum_s1 = {format_value('thr_csum_s1', values.get('thr_csum_s1', 2.0))} * pes
thr_csum_s2 = {format_value('thr_csum_s2', values.get('thr_csum_s2', 4.0))} * pes

# Set thresholds for SiPM
thr_sipm      = {format_value('thr_sipm_s2', values.get('thr_sipm_s2', 1.5))} * pes
thr_sipm_type = common

# Set parameters to search for S1
# Notice that in MC file S1 is in t=100 mus
s1_tmin       = {format_value('s1_tmin', values.get('s1_tmin', 50))} * mus # position of S1 in MC files at 100 mus
s1_tmax       = {format_value('s1_tmax', values.get('s1_tmax', 250))} * mus # change tmin and tmax if S1 not at 100 mus
s1_stride     =   {format_value('s1_stride', values.get('s1_stride', 5))}       # minimum number of 25 ns bins in S1 searches
s1_lmin       =   {format_value('s1_lmin', values.get('s1_lmin', 30))}       # 8 x 25 = 200 ns
s1_lmax       =  {format_value('s1_lmax', values.get('s1_lmax', 250))}       # 20 x 25 = 500 ns
s1_rebin_stride = {format_value('s1_rebin_stride', values.get('s1_rebin_stride', 1))}       # Do not rebin S1 by default
s1_pading     = {format_value('s1_pading', values.get('s1_pading', 10))}      # Add 10 samples on both sides of each S1 peak index

# Set parameters to search for S2
s2_tmin     =    {format_value('s2_tmin', values.get('s2_tmin', 245))} * mus # assumes S1 at 100 mus, change if S1 not at 100 mus
s2_tmax     =    {format_value('s2_tmax', values.get('s2_tmax', 290))} * mus # end of the window
s2_stride   =     {format_value('s2_stride', values.get('s2_stride', 2))}       #  40 x 25 = 1   mus
s2_lmin     =    {format_value('s2_lmin', values.get('s2_lmin', 250))}       # 100 x 25 = 2.5 mus
s2_lmax     = {format_value('s2_lmax', values.get('s2_lmax', 100000))}       # maximum value of S2 width
s2_rebin_stride = {format_value('s2_rebin_stride', values.get('s2_rebin_stride', 40))}       # Rebin by default, 40 25 ns time bins to make one 1us time bin

# Set S2Si parameters
thr_sipm_s2 = {format_value('thr_sipm_s2', values.get('thr_sipm_s2', 1.5))} * pes  # Threshold for the full sipm waveform

pmt_samp_wid  = {format_value('pmt_samp_wid', values.get('pmt_samp_wid', 25))} * ns
fiber_samp_wid = {format_value('fiber_samp_wid', values.get('fiber_samp_wid', 25))} * ns
sipm_samp_wid = {format_value('sipm_samp_wid', values.get('sipm_samp_wid', 1))} * mus
"""

    path.write_text(template)

def split_contiguous(indices: np.ndarray):
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    return np.split(indices, breaks)


def split_with_stride(indices: np.ndarray, stride: int):
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > stride)[0] + 1
    return np.split(indices, breaks)


def padded_s1_regions(indices, stride, t_us, sample_width_ns, padding=10):
    regions = []
    for seg in split_with_stride(np.asarray(indices, dtype=int), stride):
        if len(seg) == 0:
            continue
        start = max(0, int(seg[0]) - padding)
        end = int(seg[-1]) + padding
        regions.append((float(t_us[start]),
                        float(t_us[end] + sample_width_ns * 1e-3)))
    return regions


def analyze_candidate(seg, t_us, sample_width_ns, tmin_us, tmax_us, lmin, lmax):
    t0 = float(t_us[seg[0]])
    t1 = float(t_us[seg[-1]] + sample_width_ns * 1e-3)
    width = int(seg[-1] + 1 - seg[0])

    reasons = []
    if t0 < tmin_us:
        reasons.append(f"starts before tmin ({t0:.3f} < {tmin_us:.3f} us)")
    if t1 > tmax_us:
        reasons.append(f"ends after tmax ({t1:.3f} > {tmax_us:.3f} us)")
    if not (lmin <= width <= lmax):
        reasons.append(f"length out of range ({width} not in [{lmin}, {lmax}] bins)")

    return {
        "segment": seg,
        "t0": t0,
        "t1": t1,
        "width_bins": width,
        "passed": len(reasons) == 0,
        "reasons": reasons,
    }


def classify_candidate_segments(indices, stride, t_us, sample_width_ns, tmin_us, tmax_us, lmin, lmax):
    candidates = split_with_stride(np.asarray(indices, dtype=int), stride)
    analyzed, selected, rejected = [], [], []

    for seg in candidates:
        if len(seg) == 0:
            continue
        info = analyze_candidate(seg, t_us, sample_width_ns, tmin_us, tmax_us, lmin, lmax)
        analyzed.append(info)
        (selected if info["passed"] else rejected).append(seg)

    return analyzed, selected, rejected


def format_stage_a_block(label, analyzed):
    selected = sum(c["passed"] for c in analyzed)
    rejected = len(analyzed) - selected
    lines = [f"{label} candidates: {len(analyzed)} | selected: {selected} | rejected: {rejected}"]

    for i, c in enumerate(analyzed, 1):
        status = "SELECTED" if c["passed"] else "REJECTED"
        lines.append(
            f"  {label} {i:02d}: {c['t0']:8.3f}-{c['t1']:8.3f} us | "
            f"width={c['width_bins']:4d} bins | {status}"
        )
        if not c["passed"]:
            for reason in c["reasons"]:
                lines.append(f"         - {reason}")

    return lines


def build_stage_a_single_report(label, analyzed, n_in_pmap=None, pmap_error=None):
    lines = [f"=== Stage A ({label}): Candidate split + time/length selection ==="]
    lines.extend(format_stage_a_block(label, analyzed))
    lines.append("")
    lines.append("PMAP object built:")

    if n_in_pmap is not None:
        lines.append(f"  n{label} in pmap = {n_in_pmap}")
    else:
        lines.append(f"  n{label} in pmap = unavailable")
        if pmap_error:
            lines.append(f"  reason: {pmap_error}")

    return "\n".join(lines)


def build_stage_a_single_markdown(label, analyzed, n_in_pmap=None, pmap_error=None):
    selected = sum(c["passed"] for c in analyzed)
    rejected = len(analyzed) - selected

    lines = [
        f"**Summary**  ",
        f"Candidates: **{len(analyzed)}** | Selected: **{selected}** | Rejected: **{rejected}**",
        "",
        "**Candidate Details**",
    ]

    if not analyzed:
        lines.append("- No candidate regions found above threshold.")
    else:
        for i, c in enumerate(analyzed, 1):
            status = "Selected" if c["passed"] else "Rejected"
            lines.append(
                f"- **{label} {i:02d}**: {c['t0']:.3f}-{c['t1']:.3f} us | "
                f"width {c['width_bins']} bins | **{status}**"
            )
            if not c["passed"] and c["reasons"]:
                for reason in c["reasons"]:
                    lines.append(f"  - reason: {reason}")

    lines.append("")
    lines.append("**PMAP**")
    if n_in_pmap is not None:
        lines.append(f"- {label} peaks in PMAP: **{n_in_pmap}**")
    else:
        lines.append(f"- {label} peaks in PMAP: unavailable")
        if pmap_error:
            lines.append(f"- PMAP build note: {pmap_error}")

    return "\n".join(lines)


def build_stage_a_report(s1_analyzed, s2_analyzed, pmap_evt, pmap_error):
    lines = ["=== Stage A: Candidate split + time/length selection ==="]
    lines.extend(format_stage_a_block("S1", s1_analyzed))
    lines.extend(format_stage_a_block("S2", s2_analyzed))
    lines.append("")
    lines.append("PMAP object built:")

    if pmap_evt is not None:
        lines.append(f"  nS1 in pmap = {len(pmap_evt.s1s)}")
        lines.append(f"  nS2 in pmap = {len(pmap_evt.s2s)}")
    else:
        lines.append("  nS1 in pmap = unavailable")
        lines.append("  nS2 in pmap = unavailable")
        if pmap_error:
            lines.append(f"  reason: {pmap_error}")

    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def discover_run_numbers(analysis_dir: str):
    runs = [int(p.name) for p in Path(analysis_dir).iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(runs)


@st.cache_data(show_spinner=False)
def discover_ldc_files(analysis_dir: str, run_number: int, ldc: int):
    pattern = str(Path(analysis_dir) / str(run_number) / "hdf5" / "data" / f"ldc{ldc}" / "*.waveforms.h5")
    return sorted(glob.glob(pattern))


def inject_sidebar_number_styles():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] {
            background: #ffffff;
            border: 1.5px solid #9ca3af;
            border-radius: 0.55rem;
            padding: 0.15rem 0.35rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        }

        section[data-testid="stSidebar"] div[data-testid="stNumberInput"]:focus-within {
            border-color: #0b5fff;
            box-shadow: 0 0 0 3px rgba(11, 95, 255, 0.12);
        }

        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_labeled_number_input(label, **kwargs):
    label_col, input_col = st.columns([2, 1])
    label_col.markdown(f"**{label}**")
    with input_col:
        return st.number_input(label, label_visibility="collapsed", **kwargs)


def write_parameters_to_file(file_path, params):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    values = dict(params)
    template = f"""files_in = '$ICDIR/database/test_data/electrons_40keV_z25_RWF.h5'

# REPLACE /tmp with your output directory
file_out = '/tmp/electrons_40keV_z25_PMP.h5'

# compression library
compression = 'ZLIB4'

# run number 0 is for MC
run_number = 1
detector_db = 'hddemojb'

# How frequently to print events
print_mod = 1

# max number of events to run
event_range =  999

n_baseline =   {values.get('n_baseline', 2800)} # for a window of 800 mus

# Set MAW for calibrated sums
n_maw_s1 = {values.get('n_maw_s1', 20)}
n_maw_s2 = {values.get('n_maw_s2', 1)}
thr_maw =   0.1 * adc

fiber_cutoff_freq_MHz = {values.get('fiber_cutoff_freq_MHz', 3)}

# Set thresholds for calibrated sum
thr_csum_s1 = {values.get('thr_csum_s1', 2.0)} * pes
thr_csum_s2 = {values.get('thr_csum_s2', 4.0)} * pes

# Set thresholds for SiPM
thr_sipm      = {values.get('thr_sipm_s2', 1.5)} * pes
thr_sipm_type = common

# Set parameters to search for S1
# Notice that in MC file S1 is in t=100 mus
s1_tmin       = {values.get('s1_tmin', 50)} * mus # position of S1 in MC files at 100 mus
s1_tmax       = {values.get('s1_tmax', 250)} * mus # change tmin and tmax if S1 not at 100 mus
s1_stride     =   {values.get('s1_stride', 5)}       # minimum number of 25 ns bins in S1 searches
s1_lmin       =   {values.get('s1_lmin', 30)}       # 8 x 25 = 200 ns
s1_lmax       =  {values.get('s1_lmax', 250)}       # 20 x 25 = 500 ns
s1_rebin_stride = {values.get('s1_rebin_stride', 1)}       # Do not rebin S1 by default
s1_pading     = {values.get('s1_pading', 10)}      # Add 10 samples on both sides of each S1 peak index

# Set parameters to search for S2
s2_tmin     =    {values.get('s2_tmin', 245)} * mus # assumes S1 at 100 mus, change if S1 not at 100 mus
s2_tmax     =    {values.get('s2_tmax', 290)} * mus # end of the window
s2_stride   =     {values.get('s2_stride', 2)}       #  40 x 25 = 1   mus
s2_lmin     =    {values.get('s2_lmin', 250)}       # 100 x 25 = 2.5 mus
s2_lmax     = {values.get('s2_lmax', 100000)}       # maximum value of S2 width
s2_rebin_stride = {values.get('s2_rebin_stride', 40)}       # Rebin by default, 40 25 ns time bins to make one 1us time bin

# Set S2Si parameters
thr_sipm_s2 = {values.get('thr_sipm_s2', 1.5)} * pes  # Threshold for the full sipm waveform

pmt_samp_wid  = {values.get('pmt_samp_wid', 25)} * ns
fiber_samp_wid = {values.get('fiber_samp_wid', 25)} * ns
sipm_samp_wid = {values.get('sipm_samp_wid', 1)} * mus
"""

    path.write_text(template)


@st.cache_data(show_spinner=False)
def get_dataset_shape(file_path: str):
    with tb.open_file(file_path, "r") as h5in:
        shape = h5in.root.RD.fiberrwf_hg.shape
    return shape


@st.cache_data(show_spinner=False)
def load_event(file_path: str, event_idx: int):
    with tb.open_file(file_path, "r") as h5in:
        fiber_hg = h5in.root.RD.fiberrwf_hg[event_idx]
        fiber_lg = h5in.root.RD.fiberrwf_lg[event_idx]
        sipm_wf = h5in.root.RD.sipmrwf[event_idx]
        sipm_sensors = h5in.root.Sensors.DataSiPM[:]
        event_no = int(h5in.root.Run.events[event_idx][0])
    return fiber_hg, fiber_lg, sipm_wf, sipm_sensors, event_no


@st.cache_data(show_spinner=False)
def load_sipm_positions(csv_path: str):
    out = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            elecid = int(row["ElecID"])
            out[elecid] = (float(row["X"]), float(row["Y"]))
    return out


def fiber_channel_exclusion_grid(channel_ids, n_cols=8, default_excluded=None):
    default_excluded = {int(ch) for ch in (default_excluded or [])}

    for start in range(0, len(channel_ids), n_cols):
        row = channel_ids[start : start + n_cols]
        cols = st.columns(len(row))
        for col, ch in zip(cols, row):
            key = f"exclude_fiber_channel_{int(ch)}"
            if key not in st.session_state:
                st.session_state[key] = int(ch) in default_excluded
            col.markdown(
                f"<div style='text-align:center; font-size:0.9rem; font-weight:600; margin-bottom:0.15rem;'>{int(ch)}</div>",
                unsafe_allow_html=True,
            )
            col.checkbox("", key=key, label_visibility="collapsed")

    return [int(ch) for ch in channel_ids if st.session_state.get(f"exclude_fiber_channel_{int(ch)}", False)]


def get_s2_windows_us(pmap_evt, s2_selected, t_us, fiber_samp_wid_ns):
    windows = []

    if pmap_evt is not None and len(pmap_evt.s2s):
        try:
            for s2 in pmap_evt.s2s:
                times = np.asarray(s2.times, dtype=float)
                if len(times) == 0:
                    continue
                # PMAP times are in ns; convert to us.
                t0_us = float(times[0]) * 1e-3
                if len(times) > 1:
                    dt_us = float(np.median(np.diff(times))) * 1e-3
                else:
                    dt_us = float(fiber_samp_wid_ns) * 1e-3
                t1_us = float(times[-1]) * 1e-3 + dt_us
                windows.append((t0_us, t1_us))
            if windows:
                return windows
        except Exception:
            pass

    # Fallback to selected S2 candidate windows in fiber time.
    for seg in s2_selected:
        windows.append((float(t_us[seg[0]]), float(t_us[seg[-1]] + float(fiber_samp_wid_ns) * 1e-3)))
    return windows


def sipm_s2_charge_map_figure(
    sipm_wf_evt,
    sipm_sensors,
    positions_by_elecid,
    s2_windows_us,
    sipm_samp_wid_us,
    sipm_thr,
    detector_db,
    run_number,
    excluded_channels=None,
):
    if not s2_windows_us:
        return None, 0

    excluded_channels = {int(ch) for ch in (excluded_channels or [])}

    n_samples = sipm_wf_evt.shape[1]
    t_sipm_us = np.arange(n_samples, dtype=float) * float(sipm_samp_wid_us)
    mask = np.zeros_like(t_sipm_us, dtype=bool)
    for t0, t1 in s2_windows_us:
        mask |= (t_sipm_us >= float(t0)) & (t_sipm_us <= float(t1))
    if not np.any(mask):
        return None, 0

    baseline_n = max(10, min(50, n_samples // 5))
    baseline = np.median(sipm_wf_evt[:, :baseline_n], axis=1)
    corrected = sipm_wf_evt - baseline[:, None]
    corrected = np.where(corrected > 0, corrected, 0.0)
    q = np.sum(corrected[:, mask], axis=1) * float(sipm_samp_wid_us)

    x_vals, y_vals, q_vals, amp_vals, labels, sensor_indices = [], [], [], [], [], []
    for i in range(len(sipm_sensors)):
        elecid = int(sipm_sensors[i]["channel"])
        if elecid not in positions_by_elecid:
            continue
        try:
            adc_to_pes = get_sipm_adc_to_pes(detector_db, run_number, elecid)
        except KeyError:
            continue
        calibrated = subtract_baseline_and_calibrate(
            sipm_wf_evt[i][np.newaxis, :],
            np.asarray([adc_to_pes], dtype=float),
            bls_mode=BlsMode.mean,
        )[0]
        calibrated = np.where(calibrated > 0, calibrated, 0.0)
        x, y = positions_by_elecid[elecid]
        x_vals.append(x)
        y_vals.append(y)
        q_vals.append(float(q[i]))
        amp_vals.append(float(np.max(calibrated[mask])))
        labels.append(elecid)
        sensor_indices.append(i)

    if not x_vals:
        return None, 0

    q_vals = np.asarray(q_vals, dtype=float)
    amp_vals = np.asarray(amp_vals, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    sensor_indices_arr = np.asarray(sensor_indices, dtype=int)
    excluded_mask = np.isin(labels_arr, list(excluded_channels))
    included_mask = ~excluded_mask
    selected_mask = (amp_vals >= float(sipm_thr)) & included_mask

    included_x = np.asarray(x_vals, dtype=float)[included_mask]
    included_y = np.asarray(y_vals, dtype=float)[included_mask]
    included_q = q_vals[included_mask]
    included_labels = labels_arr[included_mask]
    included_sensor_indices = sensor_indices_arr[included_mask]

    excluded_x = np.asarray(x_vals, dtype=float)[excluded_mask]
    excluded_y = np.asarray(y_vals, dtype=float)[excluded_mask]
    excluded_q = q_vals[excluded_mask]
    excluded_labels = labels_arr[excluded_mask]
    excluded_sensor_indices = sensor_indices_arr[excluded_mask]

    masked_x = np.asarray(x_vals, dtype=float)[selected_mask]
    masked_y = np.asarray(y_vals, dtype=float)[selected_mask]
    masked_q = q_vals[selected_mask]
    masked_labels = labels_arr[selected_mask]
    masked_sensor_indices = sensor_indices_arr[selected_mask]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("All mapped SiPMs", "SiPMs passing threshold selection"),
        horizontal_spacing=0.08,
    )
    if included_q.size:
        base_trace = go.Scatter(
            x=included_x,
            y=included_y,
            mode="markers",
            marker=dict(
                size=10,
                color=included_q,
                colorscale="Turbo",
                colorbar=dict(title="Integrated charge"),
                line=dict(color="black", width=0.4),
            ),
            text=[f"ElecID {eid}<br>Q={qq:.2f}" for eid, qq in zip(included_labels, included_q)],
            customdata=included_sensor_indices,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
        fig.add_trace(base_trace, row=1, col=1)

    if excluded_q.size:
        fig.add_trace(
            go.Scatter(
                x=excluded_x,
                y=excluded_y,
                mode="markers",
                marker=dict(
                    size=10,
                    color="#b8b8b8",
                    line=dict(color="#666666", width=0.4),
                    symbol="x",
                ),
                text=[f"Excluded ElecID {eid}<br>Q={qq:.2f}" for eid, qq in zip(excluded_labels, excluded_q)],
                customdata=excluded_sensor_indices,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if masked_q.size:
        fig.add_trace(
            go.Scatter(
                x=masked_x,
                y=masked_y,
                mode="markers",
                marker=dict(
                    size=10,
                    color=masked_q,
                    colorscale="Turbo",
                    showscale=False,
                    line=dict(color="black", width=0.4),
                ),
                text=[f"ElecID {eid}<br>Q={qq:.2f}" for eid, qq in zip(masked_labels, masked_q)],
                customdata=masked_sensor_indices,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="SiPM integrated charge in S2 valid window(s)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=520,
        font=dict(color="black"),
    )
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1, scaleanchor="x")
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Y", row=1, col=2, scaleanchor="x2")
    return fig, int(np.count_nonzero(included_mask))


def sipm_waveform_figure(
    sipm_wf_evt,
    sensor_idx,
    elecid,
    thr_sipm_s2,
    sipm_samp_wid_us,
    detector_db,
    run_number,
    pmap_windows_us=None,
):
    raw = np.asarray(sipm_wf_evt[sensor_idx], dtype=float)
    n_samples = raw.size
    t_us = np.arange(n_samples, dtype=float) * float(sipm_samp_wid_us)

    adc_to_pes = get_sipm_adc_to_pes(detector_db, run_number, elecid)
    calibrated = subtract_baseline_and_calibrate(
        raw[np.newaxis, :],
        np.asarray([adc_to_pes], dtype=float),
        bls_mode=BlsMode.mean,
    )[0]
    sipm_thr = get_actual_sipm_thr(SiPMThreshold.common, float(thr_sipm_s2), detector_db, run_number)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(x=t_us, y=raw, mode="lines", name="raw ADC", line=dict(width=1.2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_us, y=calibrated, mode="lines", name="calibrated (pes)", line=dict(width=1.2)),
        row=2,
        col=1,
    )
    fig.add_hline(y=float(sipm_thr), line_dash="dash", line_color="#d94a4a", row=2, col=1)
    if pmap_windows_us:
        for t0, t1 in pmap_windows_us:
            fig.add_vrect(
                x0=float(t0),
                x1=float(t1),
                fillcolor="#1f77b4",
                opacity=0.16,
                line_width=0,
                row=2,
                col=1,
            )
    fig.update_layout(
        title=f"Selected SiPM waveform: ElecID {elecid}",
        height=620,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(title_text="Time (us)", row=2, col=1)
    fig.update_yaxes(title_text="ADC", row=1, col=1)
    fig.update_yaxes(title_text="pes", row=2, col=1)
    return fig


@st.cache_data(show_spinner=False)
def get_sipm_adc_to_pes(detector_db: str, run_number: int, elecid: int) -> float:
    datasipm = load_db.DataSiPM(detector_db, run_number)
    row = datasipm.loc[datasipm.ChannelID == elecid]
    if row.empty:
        raise KeyError(f"SiPM ElecID {elecid} not found in DataSiPM for {detector_db}, run {run_number}")
    return float(abs(row.adc_to_pes.iloc[0]))


def get_selected_sipm_index(selection_state):
    if not selection_state:
        return None

    if isinstance(selection_state, dict):
        selected_points = selection_state.get("points", [])
    else:
        selected_points = getattr(selection_state, "points", [])

    if not selected_points:
        return None

    point = selected_points[0]
    if isinstance(point, dict):
        customdata = point.get("customdata")
        point_index = point.get("point_index")
    else:
        customdata = getattr(point, "customdata", None)
        point_index = getattr(point, "point_index", None)

    if customdata is not None:
        if isinstance(customdata, (list, tuple, np.ndarray)) and len(customdata) > 0:
            return int(customdata[0])
        return int(customdata)

    return int(point_index) if point_index is not None else None


def overlay_plot(t_us, a, b, title, name_a, name_b, y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_us, y=a, mode="lines", name=name_a, line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=t_us, y=b, mode="lines", name=name_b, line=dict(width=1.2)))
    fig.update_layout(
        title=title,
        xaxis_title="Time (us)",
        yaxis_title=y_title,
        height=350,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def baseline_diagnostic_plot(t_us, waveform, n_baseline, title, bin_size=None):
    baseline_n = min(int(n_baseline), len(waveform))
    baseline_samples = np.asarray(waveform[:baseline_n])
    if bin_size is None:
        distribution_samples = baseline_samples
        baseline_mode = float(means(baseline_samples[np.newaxis, :])[0])
        baseline_label = "mean"
        distribution_title = "Distribution of baseline samples; mean marked"
    else:
        distribution_samples = (baseline_samples // int(bin_size)) * int(bin_size)
        baseline_mode = float(means(baseline_samples[np.newaxis, :])[0])
        baseline_label = "mean"
        distribution_title = f"Distribution of baseline samples (bin size {int(bin_size)}); mean marked"
    baseline_times = np.asarray(t_us[:baseline_n])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.16,
        subplot_titles=(
            f"Waveform and baseline window (first {baseline_n} samples)",
            distribution_title,
        ),
    )
    fig.add_trace(
        go.Scatter(x=t_us, y=waveform, mode="lines", name="waveform", line=dict(width=1.1)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_times,
            y=baseline_samples,
            mode="markers",
            name="baseline samples",
            marker=dict(size=4, color="#f08c46"),
        ),
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0=float(baseline_times[0]),
        x1=float(baseline_times[-1]),
        fillcolor="#f08c46",
        opacity=0.14,
        line_width=0,
        row=1,
        col=1,
    )

    values, counts = np.unique(distribution_samples, return_counts=True)
    fig.add_trace(
        go.Bar(
            x=values,
            y=counts,
            name="sample count",
            marker=dict(color="#17324d", line=dict(color="#0b1f33", width=0.8)),
            width=int(bin_size) if bin_size is not None else None,
            hovertemplate="ADC bin=%{x}<br>count=%{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_vline(
        x=baseline_mode,
        line_dash="dash",
        line_color="#d94841",
        annotation_text=f"{baseline_label} = {baseline_mode:g}",
        annotation_position="top right",
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title,
        height=650,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(title_text="Time (us)", row=1, col=1)
    fig.update_yaxes(title_text="ADC", row=1, col=1)
    distribution_x_title = "ADC bin lower edge" if bin_size is not None else "ADC value"
    fig.update_xaxes(title_text=distribution_x_title, row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    return fig


def threshold_plot(
    t_us,
    y,
    thr,
    thr_label,
    title,
    selected,
    rejected,
    selected_color,
    rejected_color,
    allowed_window=None,
    extra_regions=None,
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_us, y=y, mode="lines", name="summed waveform", line=dict(width=1.2)))
    fig.add_hline(y=thr, line_dash="dash")

    if allowed_window is not None:
        t0, t1 = allowed_window
        fig.add_vrect(
            x0=float(t0),
            x1=float(t1),
            fillcolor="#a5d8ff",
            opacity=0.12,
            line_width=0,
        )

    if extra_regions:
        for t0, t1 in extra_regions:
            fig.add_vrect(
                x0=float(t0),
                x1=float(t1),
                fillcolor="#f4a261",
                opacity=0.22,
                line_width=0,
            )

    for i, seg in enumerate(selected):
        fig.add_vrect(
            x0=float(t_us[seg[0]]),
            x1=float(t_us[seg[-1]]),
            fillcolor=selected_color,
            opacity=0.35,
            line_width=1,
            line_color=selected_color,
        )

    for i, seg in enumerate(rejected):
        fig.add_vrect(
            x0=float(t_us[seg[0]]),
            x1=float(t_us[seg[-1]]),
            fillcolor=rejected_color,
            opacity=0.28,
            line_width=1,
            line_color=rejected_color,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (us)",
        yaxis_title="pes",
        height=350,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def main():
    st.set_page_config(page_title="Irene Interactive Pipeline", layout="wide")
    inject_sidebar_number_styles()
    st.title("Irene Interactive Pipeline")
    st.caption("Select run/event/channel and tune pipeline parameters live.")

    available_runs = discover_run_numbers(str(ANALYSIS_DIR))
    if not available_runs:
        st.error(f"No run directories found under {ANALYSIS_DIR}")
        st.stop()

    with st.sidebar:
        st.header("Input")
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()

        run_number = st.selectbox("Run number", options=available_runs[::-1], index=0)

        ldc = st.selectbox("LDC", options=[1, 2], index=0)

        ldc_files = discover_ldc_files(str(ANALYSIS_DIR), int(run_number), int(ldc))
        if not ldc_files:
            st.error(f"No files found for run {run_number}, ldc{ldc}")
            st.stop()

        file_names = [Path(f).name for f in ldc_files]
        selected_file_name = st.selectbox("File", options=file_names[::-1], index=0)
        selected_file = str(
            Path(ANALYSIS_DIR) / str(run_number) / "hdf5" / "data" / f"ldc{ldc}" / selected_file_name
        )

        n_events, n_fibers, n_samples = get_dataset_shape(selected_file)

        event_idx_requested = st.number_input(
            "Event index (run-level)",
            min_value=0,
            max_value=int(n_events) - 1,
            value=min(12, int(n_events) - 1),
            step=1,
        )
        event_idx = int(event_idx_requested)

        st.caption(f"File: {selected_file_name} | events: {n_events}")

        fiber_ch_requested = st.number_input(
            "Fiber channel",
            min_value=0,
            max_value=10000000,
            value=4,
            step=1,
        )
        fiber_ch = min(int(fiber_ch_requested), int(n_fibers) - 1)
        if int(fiber_ch_requested) != fiber_ch:
            st.warning(f"Requested fiber channel {int(fiber_ch_requested)} exceeds max {int(n_fibers) - 1}. Using {fiber_ch}.")

        detector_db = st.selectbox("Detector DB", ["hddemojb", "hddemo"], index=0)
        
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        st.header("Parameters")
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        n_baseline = sidebar_labeled_number_input("N_BASELINE", min_value=100, max_value=n_samples, value=N_BASELINE_DEFAULT, step=100)
        n_maw_s1 = sidebar_labeled_number_input("N_MAW_S1", min_value=1, max_value=5000, value=N_MAW_S1_DEFAULT, step=1)
        n_maw_s2 = sidebar_labeled_number_input("N_MAW_S2", min_value=1, max_value=5000, value=N_MAW_S2_DEFAULT, step=1)
        fiber_cutoff_mhz = sidebar_labeled_number_input("FIBER_CUTOFF_FREQ_MHZ", min_value=0.1, max_value=100.0, value=FIBER_CUTOFF_MHZ_DEFAULT, step=0.1)
        thr_csum_s1 = sidebar_labeled_number_input("THR_CSUM_S1 (pes)", min_value=0.0, max_value=1e6, value=THR_CSUM_S1_DEFAULT, step=1.0)
        thr_csum_s2 = sidebar_labeled_number_input("THR_CSUM_S2 (pes)", min_value=0.0, max_value=1e6, value=THR_CSUM_S2_DEFAULT, step=1.0)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        st.subheader("S1 selection")
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        s1_tmin_us = sidebar_labeled_number_input("s1_tmin (us)", min_value=0.0, max_value=1e6, value=S1_TMIN_US_DEFAULT, step=1.0)
        s1_tmax_us = sidebar_labeled_number_input("s1_tmax (us)", min_value=0.0, max_value=1e6, value=S1_TMAX_US_DEFAULT, step=1.0)
        s1_stride = sidebar_labeled_number_input("s1_stride", min_value=1, max_value=10000, value=S1_STRIDE_DEFAULT, step=1)
        s1_lmin = sidebar_labeled_number_input("s1_lmin", min_value=1, max_value=1000000, value=S1_LMIN_DEFAULT, step=1)
        s1_lmax = sidebar_labeled_number_input("s1_lmax", min_value=1, max_value=1000000, value=S1_LMAX_DEFAULT, step=1)
        s1_rebin_stride = sidebar_labeled_number_input("s1_rebin_stride", min_value=1, max_value=100000, value=S1_REBIN_STRIDE_DEFAULT, step=1)
        s1_padding = sidebar_labeled_number_input("s1_padding", min_value=0, max_value=1000, value=S1_PADDING_DEFAULT, step=1)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        st.subheader("S2 selection")
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)

        s2_tmin_us = sidebar_labeled_number_input("s2_tmin (us)", min_value=0.0, max_value=1e6, value=S2_TMIN_US_DEFAULT, step=1.0)
        s2_tmax_us = sidebar_labeled_number_input("s2_tmax (us)", min_value=0.0, max_value=1e6, value=S2_TMAX_US_DEFAULT, step=1.0)
        s2_stride = sidebar_labeled_number_input("s2_stride", min_value=1, max_value=10000, value=S2_STRIDE_DEFAULT, step=1)
        s2_lmin = sidebar_labeled_number_input("s2_lmin", min_value=1, max_value=1000000, value=S2_LMIN_DEFAULT, step=1)
        s2_lmax = sidebar_labeled_number_input("s2_lmax", min_value=1, max_value=1000000, value=S2_LMAX_DEFAULT, step=1)
        s2_rebin_stride = sidebar_labeled_number_input("s2_rebin_stride", min_value=1, max_value=100000, value=S2_REBIN_STRIDE_DEFAULT, step=1)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        st.subheader("SiPM selection")
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        thr_sipm_s2 = sidebar_labeled_number_input("thr_sipm_s2 (pes)", min_value=0.0, max_value=1e6, value=THR_SIPM_S2_DEFAULT, step=0.1)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        st.subheader("PMAP sampling")
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        pmt_samp_wid_ns = sidebar_labeled_number_input("pmt_samp_wid (ns)", min_value=1.0, max_value=1000.0, value=PMT_SAMP_WID_NS_DEFAULT, step=1.0)
        fiber_samp_wid = sidebar_labeled_number_input("FIBER_SAMP_WID (ns)", min_value=1.0, max_value=1000.0, value=FIBER_SAMP_WID_NS_DEFAULT, step=1.0)
        sipm_samp_wid_us = sidebar_labeled_number_input("sipm_samp_wid (us)", min_value=0.1, max_value=1000.0, value=SIPM_SAMP_WID_US_DEFAULT, step=0.1)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        fiber_channel_options = list(range(min(36, int(n_fibers))))
        excluded_fiber_channels = fiber_channel_exclusion_grid(
            fiber_channel_options,
            n_cols=8,
            default_excluded=st.session_state.get("excluded_fiber_channels", []),
        )
        st.session_state["excluded_fiber_channels"] = excluded_fiber_channels

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        password = st.text_input("Password", type="password", placeholder="Enter password to save")
        if st.button("Save current values to irene.conf", use_container_width=True):
            if not password:
                st.error("Password required to save.")
            elif hashlib.sha256(password.encode()).hexdigest() != AUTHORIZED_PASSWORD_HASH:
                st.error("Incorrect password.")
            else:
                config_updates = {
                    "n_baseline": int(n_baseline),
                    "n_maw_s1": int(n_maw_s1),
                    "n_maw_s2": int(n_maw_s2),
                    "fiber_cutoff_freq_MHz": float(fiber_cutoff_mhz),
                    "thr_csum_s1": float(thr_csum_s1),
                    "thr_csum_s2": float(thr_csum_s2),
                    "s1_tmin": float(s1_tmin_us),
                    "s1_tmax": float(s1_tmax_us),
                    "s1_stride": int(s1_stride),
                    "s1_lmin": int(s1_lmin),
                    "s1_lmax": int(s1_lmax),
                    "s1_rebin_stride": int(s1_rebin_stride),
                    "s1_pading": int(s1_padding),
                    "s2_tmin": float(s2_tmin_us),
                    "s2_tmax": float(s2_tmax_us),
                    "s2_stride": int(s2_stride),
                    "s2_lmin": int(s2_lmin),
                    "s2_lmax": int(s2_lmax),
                    "s2_rebin_stride": int(s2_rebin_stride),
                    "thr_sipm_s2": float(thr_sipm_s2),
                    "pmt_samp_wid": float(pmt_samp_wid_ns),
                    "fiber_samp_wid": float(fiber_samp_wid),
                    "sipm_samp_wid": float(sipm_samp_wid_us),
                }
                try:
                    write_parameters_to_file(CONFIG_FILE, config_updates)
                    st.success(f"Saved {len(config_updates)} numeric values to {CONFIG_FILE.name}")
                except Exception as exc:
                    st.error(f"Failed to save configuration: {exc}")

    st.info(
        f"Run {int(run_number)} | LDC {ldc} | {selected_file_name} | "
        f"Event {event_idx}/{n_events - 1} | Channel {fiber_ch}/{n_fibers - 1} | Samples {n_samples}"
    )

    try:
        fiber_hg_raw, fiber_lg_raw, sipm_wf_evt, sipm_sensors, event_number = load_event(selected_file, event_idx)
        positions_by_elecid = load_sipm_positions(str(SIPM_POSITIONS_CSV))

        t_us = np.arange(n_samples) * float(fiber_samp_wid) * 1e-3

        bsfiber_hg = -(fiber_hg_raw - means(fiber_hg_raw[:, :int(n_baseline)]))
        bsfiber_lg = -(fiber_lg_raw - means(fiber_lg_raw[:, :int(n_baseline)]))

        fiber_active = np.ones(int(n_fibers), dtype=bool)
        if excluded_fiber_channels:
            fiber_active[np.asarray(excluded_fiber_channels, dtype=int)] = False
        masked_bsfiber_hg = mask_sensors(bsfiber_hg, fiber_active)
        masked_bsfiber_lg = mask_sensors(bsfiber_lg, fiber_active)

        apply_fourier_filter = fourier_filter(float(fiber_samp_wid), float(fiber_cutoff_mhz))
        bsffiber_hg = apply_fourier_filter(masked_bsfiber_hg)
        bsffiber_lg = apply_fourier_filter(masked_bsfiber_lg)

        cal_hg = calibrate_fibers_hg(detector_db, int(run_number), int(n_maw_s1))
        cal_lg = calibrate_fibers_lg(detector_db, int(run_number), int(n_maw_s2))
        cbsfiber_hg_maw, cbsfiber_hg_sum_maw = cal_hg(bsffiber_hg)
        cbsfiber_lg_maw, cbsfiber_lg_sum_maw = cal_lg(bsffiber_lg)

        zs_hg = zero_suppress_wfs_hg(float(thr_csum_s1))
        zs_lg = zero_suppress_wfs_lg(float(thr_csum_s2))
        s1_indices = zs_hg(cbsfiber_hg_sum_maw)
        s2_indices, s2_energies = zs_lg(cbsfiber_lg_sum_maw)

        s1_analyzed, s1_selected, s1_rejected = classify_candidate_segments(
            s1_indices,
            int(s1_stride),
            t_us,
            float(fiber_samp_wid),
            float(s1_tmin_us),
            float(s1_tmax_us),
            int(s1_lmin),
            int(s1_lmax),
        )
        s2_analyzed, s2_selected, s2_rejected = classify_candidate_segments(
            s2_indices,
            int(s2_stride),
            t_us,
            float(fiber_samp_wid),
            float(s2_tmin_us),
            float(s2_tmax_us),
            int(s2_lmin),
            int(s2_lmax),
        )

        pmap_evt = None
        pmap_error = None
        try:
            pmap_builder = build_pmap_dual_gain(
                detector_db,
                int(run_number),
                float(pmt_samp_wid_ns) * units.ns,
                float(sipm_samp_wid_us) * units.mus,
                int(s1_lmax),
                int(s1_lmin),
                int(s1_rebin_stride),
                int(s1_stride),
                float(s1_tmax_us) * units.mus,
                float(s1_tmin_us) * units.mus,
                int(s2_lmax),
                int(s2_lmin),
                int(s2_rebin_stride),
                int(s2_stride),
                float(s2_tmax_us) * units.mus,
                float(s2_tmin_us) * units.mus,
                float(thr_sipm_s2),
                s1_pading=int(s1_padding),
            )
            pmap_evt = pmap_builder(cbsfiber_hg_maw, cbsfiber_lg_maw, s1_indices, s2_indices, None)
        except Exception as e:
            pmap_error = str(e)

        if pmap_evt is not None:
            stage_a_s1_md = build_stage_a_single_markdown("S1", s1_analyzed, n_in_pmap=len(pmap_evt.s1s))
            stage_a_s2_md = build_stage_a_single_markdown("S2", s2_analyzed, n_in_pmap=len(pmap_evt.s2s))
        else:
            stage_a_s1_md = build_stage_a_single_markdown("S1", s1_analyzed, n_in_pmap=None, pmap_error=pmap_error)
            stage_a_s2_md = build_stage_a_single_markdown("S2", s2_analyzed, n_in_pmap=None, pmap_error=pmap_error)

        s2_windows_us = get_s2_windows_us(pmap_evt, s2_selected, t_us, float(fiber_samp_wid))
        sipm_thr = get_actual_sipm_thr(SiPMThreshold.common, float(thr_sipm_s2), detector_db, run_number)
        sipm_map_fig, sipm_mapped = sipm_s2_charge_map_figure(
            sipm_wf_evt,
            sipm_sensors,
            positions_by_elecid,
            s2_windows_us,
            float(sipm_samp_wid_us),
            float(sipm_thr),
            detector_db,
            int(run_number),
        )

    except Exception as exc:
        st.error("Pipeline execution failed with current settings.")
        st.exception(exc)
        st.stop()

    st.subheader(f"Event number: {event_number}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            overlay_plot(
                t_us,
                fiber_hg_raw[fiber_ch],
                bsfiber_hg[fiber_ch],
                "HG: raw vs baseline-subtracted",
                "raw ADC",
                "baseline-subtracted ADC",
                "ADC",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            overlay_plot(
                t_us,
                fiber_lg_raw[fiber_ch],
                bsfiber_lg[fiber_ch],
                "LG: raw vs baseline-subtracted",
                "raw ADC",
                "baseline-subtracted ADC",
                "ADC",
            ),
            use_container_width=True,
        )

    col_baseline_hg, col_baseline_lg = st.columns(2)
    with col_baseline_hg:
        st.plotly_chart(
            baseline_diagnostic_plot(
                t_us,
                fiber_hg_raw[fiber_ch],
                int(n_baseline),
                f"HG baseline diagnostic (mean): channel {fiber_ch}",
                bin_size=64,
            ),
            use_container_width=True,
        )
    with col_baseline_lg:
        st.plotly_chart(
            baseline_diagnostic_plot(
                t_us,
                fiber_lg_raw[fiber_ch],
                int(n_baseline),
                f"LG baseline diagnostic (mean): channel {fiber_ch}",
            ),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            overlay_plot(
                t_us,
                bsfiber_hg[fiber_ch],
                bsffiber_hg[fiber_ch],
                "HG: before vs after Fourier filter",
                "before",
                "after",
                "ADC",
            ),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            overlay_plot(
                t_us,
                bsfiber_lg[fiber_ch],
                bsffiber_lg[fiber_ch],
                "LG: before vs after Fourier filter",
                "before",
                "after",
                "ADC",
            ),
            use_container_width=True,
        )

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(
            overlay_plot(
                t_us,
                bsffiber_hg[fiber_ch],
                cbsfiber_hg_maw[fiber_ch],
                "HG: filtered ADC vs calibrated MAW",
                "filtered ADC",
                "calibrated MAW (pes)",
                "value",
            ),
            use_container_width=True,
        )
    with col6:
        st.plotly_chart(
            overlay_plot(
                t_us,
                bsffiber_lg[fiber_ch],
                cbsfiber_lg_maw[fiber_ch],
                "LG: filtered ADC vs calibrated MAW",
                "filtered ADC",
                "calibrated MAW (pes)",
                "value",
            ),
            use_container_width=True,
        )

    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(
            threshold_plot(
                t_us,
                cbsfiber_hg_sum_maw,
                float(thr_csum_s1),
                f"S1 threshold = {thr_csum_s1:.1f}",
                "HG sum with S1 selected/rejected windows",
                s1_selected,
                s1_rejected,
                "#2faa60",
                "#d94a4a",
                allowed_window=(float(s1_tmin_us), float(s1_tmax_us)),
            ),
            use_container_width=True,
        )
    with col8:
        padded_selected_s1 = padded_s1_regions(
            np.concatenate(s1_selected) if s1_selected else np.array([], dtype=int),
            int(s1_stride),
            t_us,
            float(fiber_samp_wid),
            padding=int(s1_padding),
        )
        st.plotly_chart(
            threshold_plot(
                t_us,
                cbsfiber_lg_sum_maw,
                float(thr_csum_s2),
                f"S2 threshold = {thr_csum_s2:.1f}",
                "LG sum with S2 selected/rejected windows",
                s2_selected,
                s2_rejected,
                "#1e8e5a",
                "#c73e3e",
                allowed_window=(float(s2_tmin_us), float(s2_tmax_us)),
                extra_regions=padded_selected_s1,
            ),
            use_container_width=True,
        )

    with st.expander("Stage A candidate diagnostic", expanded=True):
        left, right = st.columns(2)
        with left:
            st.markdown("### S1 diagnostics")
            st.markdown(stage_a_s1_md)
        with right:
            st.markdown("### S2 diagnostics")
            st.markdown(stage_a_s2_md)

    st.subheader("SiPM S2 Charge Map")
    if sipm_map_fig is None:
        st.info("No S2 window available for SiPM integration with current settings.")
    else:
        st.caption(f"Mapped SiPM sensors: {sipm_mapped}")
        sipm_map_event = st.plotly_chart(
            sipm_map_fig,
            use_container_width=True,
            key="sipm_s2_charge_map",
            on_select="rerun",
            selection_mode="points",
        )

        selected_sipm_idx = get_selected_sipm_index(sipm_map_event.selection)
        if selected_sipm_idx is not None:
            st.session_state["selected_sipm_idx"] = selected_sipm_idx
        else:
            selected_sipm_idx = st.session_state.get("selected_sipm_idx")

        if selected_sipm_idx is not None:
            selected_sipm_idx = int(selected_sipm_idx)
            if 0 <= selected_sipm_idx < len(sipm_sensors):
                selected_elecid = int(sipm_sensors[selected_sipm_idx]["channel"])
                st.caption(
                    f"Selected SiPM ElecID {selected_elecid} | sensor index {selected_sipm_idx}"
                )
                st.plotly_chart(
                    sipm_waveform_figure(
                        sipm_wf_evt,
                        selected_sipm_idx,
                        selected_elecid,
                        float(thr_sipm_s2),
                        float(sipm_samp_wid_us),
                        detector_db,
                        int(run_number),
                        pmap_windows_us=s2_windows_us,
                    ),
                    use_container_width=True,
                )
            else:
                st.warning("Selected SiPM index is outside the available sensor range.")
        else:
            st.info("Click a SiPM marker to show its waveform below.")

    with st.expander("Current configuration"):
        st.json(
            {
                "file": selected_file,
                "run_number": int(run_number),
                "event_idx_requested": int(event_idx_requested),
                "event_idx_local": int(event_idx),
                "n_events_file": int(n_events),
                "detector_db": detector_db,
                "event_idx": int(event_idx),
                "fiber_ch": int(fiber_ch),
                "N_BASELINE": int(n_baseline),
                "N_MAW_S1": int(n_maw_s1),
                "N_MAW_S2": int(n_maw_s2),
                "FIBER_SAMP_WID": float(fiber_samp_wid),
                "FIBER_CUTOFF_FREQ_MHZ": float(fiber_cutoff_mhz),
                "THR_CSUM_S1": float(thr_csum_s1),
                "THR_CSUM_S2": float(thr_csum_s2),
                "s1_tmin_us": float(s1_tmin_us),
                "s1_tmax_us": float(s1_tmax_us),
                "s1_stride": int(s1_stride),
                "s1_lmin": int(s1_lmin),
                "s1_lmax": int(s1_lmax),
                "s1_rebin_stride": int(s1_rebin_stride),
                "s2_tmin_us": float(s2_tmin_us),
                "s2_tmax_us": float(s2_tmax_us),
                "s2_stride": int(s2_stride),
                "s2_lmin": int(s2_lmin),
                "s2_lmax": int(s2_lmax),
                "s2_rebin_stride": int(s2_rebin_stride),
                "thr_sipm_s2": float(thr_sipm_s2),
                "pmt_samp_wid_ns": float(pmt_samp_wid_ns),
                "sipm_samp_wid_us": float(sipm_samp_wid_us),
            }
        )


if __name__ == "__main__":
    main()
