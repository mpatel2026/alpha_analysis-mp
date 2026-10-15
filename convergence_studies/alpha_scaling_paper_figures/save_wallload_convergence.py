"""
Standard deviation of peak wall heat load vs. marker count.

Two phases:
    python save_wallload_convergence.py compute   # subsample + save data to CACHE_FILE
    python save_wallload_convergence.py wetted    # only the wetted-area sampling -> WET_CACHE_FILE
    python save_wallload_convergence.py plot      # load the caches and make the figure
    python save_wallload_convergence.py           # both
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Plotting Parameters ---
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'sans-serif'
})

width_inch = 190 / 25.4
height_inch = width_inch * (6/10)

# --- Configuration ---
inp_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/ISHW/"
CACHE_FILE = "wallload_convergence_data.npz"
WET_CACHE_FILE = "wallload_wetted_data.npz"  # wetted area vs. marker count (wet_marker_count_list)
FIG_FILE = "figs/wall-pload-std-convergence.png"
FIG_FILE_COMBINED = "figs/wall-pload-std-wetted-convergence.png"

# name -> (filename, cell area in m^2)
files = {
    "p001": "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p001cellarea_FLR-mode_decay-profile_05082026.h5",
    #"p003": "G1600-free-reopt_I_10cm-wall_1M-mrk_biot-bfield_p003cellarea_FLR-mode_zero-profile_09102026.h5",
    "p005": "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p005cellarea_FLR-mode_decay-profile_05082026.h5",
    #"p007": "G1600-free-reopt_I_10cm-wall_1M-mrk_biot-bfield_p007cellarea_FLR-mode_zero-profile_09102026.h5",
    "p01":  "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5",
    #"p015": "G1600-free-reopt_I_10cm-wall_1M-mrk_biot-bfield_p015cellarea_FLR-mode_zero-profile_09102026.h5",
    #"p02":  "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p02cellarea_FLR-mode_decay-profile_05082026.h5",
    #"p03":  "G1600-free-reopt_I_10cm-wall_1M-mrk_biot-bfield_p03cellarea_FLR-mode_zero-profile_09102026.h5",
    "p05":  "G1600-free-reopt_I_10cm-wall_1M-mrk_biot-bfield_p05cellarea_FLR-mode_zero-profile_09102026.h5",
}

label_map = {
    "p001": r'0.001 m$^2$',
    #"p003": r'0.003 m$^2$',
    "p005": r'0.005 m$^2$',
    #"p007": r'0.007 m$^2$',
    "p01":  r'0.01 m$^2$',
    #"p015": r'0.015 m$^2$',
    #"p02":  r'0.02 m$^2$',
    #"p03":  r'0.03 m$^2$',
    "p05":  r'0.05 m$^2$',
}

# Cell sizes shown in the figure (all cached data is kept)
PLOT_NAMES = ["p001", "p005", "p01", "p05"]

# Batch sizes (200k gives only 5 batches, so its std is noisy)
marker_count_list = [10000, 20000, 50000, 100000, 200000]
# Batch sizes for the wetted-area panel (1M = the whole run, so a single batch with no spread)
wet_marker_count_list = [100000, 200000, 500000, 1000000]
# Markers are shuffled and split into disjoint (independent) batches: nmrk // N batches per count.
MAX_BATCHES = 1000  # padding size for the cache (NaN-filled)
N_REF_MAX = 1_000_000  # reference curves extend to this many markers
WET_THRESHOLD_WM2 = 1e3  # a cell counts as wetted if its (reweighted) surface heat flux exceeds 1 kW/m^2


# =====================================================================
# Phase 1: data calculation
# =====================================================================
def save_cache(cache):
    names = [n for n in files if n in cache]
    np.savez(CACHE_FILE, names=np.array(names),
             marker_counts=np.array(marker_count_list),
             peak_loads=np.array([cache[n][0] for n in names]),
             wetted_areas=np.array([cache[n][1] for n in names]),
             wet_threshold_wm2=WET_THRESHOLD_WM2)


def load_markers(obj):
    """End-state marker data needed to evaluate wall loads on subsets of markers."""
    all_ids, all_econd, all_tile, all_weight = obj.data.active.getstate(
        "ids", "endcond", "walltile", "weight", state="end"
    )
    all_econd_str = np.char.lower(all_econd.astype(str))
    all_is_wall = (all_econd_str == "wall") | (all_econd == 1) | (all_tile > 0)

    if hasattr(obj, 'getwall_loads'):
        load_func = obj.getwall_loads
    elif hasattr(obj.data.active, 'getwall_loads'):
        load_func = obj.data.active.getwall_loads
    else:
        raise AttributeError("Could not find 'getwall_loads'.")
    return load_func, all_ids, all_is_wall, all_weight


def batch_loads(load_func, all_ids, all_is_wall, all_weight, idx):
    """Peak surface load (W/m^2) and wetted area (m^2) using only markers idx, reweighted to full power."""
    wall_ids = all_ids[idx][all_is_wall[idx]]
    w_sub_sum = np.sum(all_weight[idx])

    p_load, w_area = 0.0, 0.0
    if len(wall_ids) > 0:
        _, areas, loads, _, _ = load_func(weights=True, p_ids=wall_ids)
        reweight = np.sum(all_weight) / w_sub_sum if w_sub_sum > 0 else 0
        if loads is not None and loads.size > 0:
            areas_m2 = np.asarray(areas)
            flux = np.asarray(loads) * reweight / areas_m2  # W/m^2 at full weight
            p_load = np.max(flux)
            w_area = np.sum(areas_m2[flux > WET_THRESHOLD_WM2])
    return p_load, w_area


def compute():
    from a5py import Ascot

    np.random.seed(42)
    cache = {}
    if os.path.exists(CACHE_FILE):
        old = np.load(CACHE_FILE)
        if ('wetted_areas' in old and np.array_equal(old['marker_counts'], marker_count_list)
                and 'wet_threshold_wm2' in old and float(old['wet_threshold_wm2']) == WET_THRESHOLD_WM2):
            cache = {n: (old['peak_loads'][i], old['wetted_areas'][i]) for i, n in enumerate(old['names'])
                     if np.nanmax(old['peak_loads'][i]) > 0}

    for name in files:
        if name in cache:
            print(f"--- {name}: cached, skipping ---", flush=True)
            continue
        print(f"--- Processing: {name} ---", flush=True)
        obj = Ascot(inp_dir + files[name])
        if obj.data.active is None:
            print(f"  WARNING: {name} has no results in file, skipping", flush=True)
            continue
        peak = np.full((len(marker_count_list), MAX_BATCHES), np.nan)
        wet = np.full((len(marker_count_list), MAX_BATCHES), np.nan)

        load_func, all_ids, all_is_wall, all_weight = load_markers(obj)
        nmrk = len(all_ids)

        for ci, marker_count in enumerate(marker_count_list):
            nbatch = nmrk // marker_count
            perm = np.random.permutation(nmrk)
            for t in range(nbatch):
                idx = perm[t * marker_count:(t + 1) * marker_count]
                p_load, w_area = batch_loads(load_func, all_ids, all_is_wall, all_weight, idx)
                peak[ci, t] = p_load
                wet[ci, t] = w_area

            std = np.nanstd(peak[ci], ddof=1)
            print(f"  N={marker_count//1000}k: std = {std:.4e} W/m^2 "
                  f"(mean = {np.nanmean(peak[ci]):.4e})", flush=True)

        # Save after each file so partial progress isn't lost
        cache[name] = (peak, wet)
        save_cache(cache)

    print(f"Saved data to {CACHE_FILE}")


def save_wet_cache(cache):
    names = [n for n in files if n in cache]
    np.savez(WET_CACHE_FILE, names=np.array(names),
             marker_counts=np.array(wet_marker_count_list),
             wetted_areas=np.array([cache[n] for n in names]),
             wet_threshold_wm2=WET_THRESHOLD_WM2)


def compute_wetted():
    """Wetted area vs. marker count at wet_marker_count_list, for PLOT_NAMES only."""
    from a5py import Ascot

    rng = np.random.RandomState(42)
    cache = {}
    if os.path.exists(WET_CACHE_FILE):
        old = np.load(WET_CACHE_FILE)
        if (np.array_equal(old['marker_counts'], wet_marker_count_list)
                and float(old['wet_threshold_wm2']) == WET_THRESHOLD_WM2):
            cache = {n: old['wetted_areas'][i] for i, n in enumerate(old['names'])}

    for name in PLOT_NAMES:
        if name in cache:
            print(f"--- {name} (wetted): cached, skipping ---", flush=True)
            continue
        print(f"--- Processing (wetted): {name} ---", flush=True)
        obj = Ascot(inp_dir + files[name])
        if obj.data.active is None:
            print(f"  WARNING: {name} has no results in file, skipping", flush=True)
            continue
        load_func, all_ids, all_is_wall, all_weight = load_markers(obj)
        nmrk = len(all_ids)
        wet = np.full((len(wet_marker_count_list), MAX_BATCHES), np.nan)

        for ci, marker_count in enumerate(wet_marker_count_list):
            nbatch = nmrk // marker_count
            perm = rng.permutation(nmrk)
            for t in range(nbatch):
                idx = perm[t * marker_count:(t + 1) * marker_count]
                _, wet[ci, t] = batch_loads(load_func, all_ids, all_is_wall, all_weight, idx)
            print(f"  N={marker_count//1000}k: {nbatch} batch(es), "
                  f"A_wetted mean = {np.nanmean(wet[ci]):.2f} m^2", flush=True)

        # Save after each file so partial progress isn't lost
        cache[name] = wet
        save_wet_cache(cache)

    print(f"Saved data to {WET_CACHE_FILE}")


# =====================================================================
# Phase 2: plotting
# =====================================================================
def plot_std(ax, d, key, scale, ylabel):
    names = list(d['names'])
    x = d['marker_counts'] / 1000.0  # kilomarkers
    x_ref = np.linspace(x.min(), N_REF_MAX / 1000.0, 200)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    colors = [f'C{i}' for i in range(10)]  # matplotlib default cycle, as in wall-pload-convergence.png

    for ci, name in enumerate(PLOT_NAMES):
        ni = names.index(name)
        y = np.nanstd(d[key][ni], axis=1, ddof=1) / scale
        ok = y > 0
        # Least-squares fit of y = C / sqrt(x) in log space -> log C = mean(log y + 0.5 log x)
        C = np.exp(np.mean(np.log(y[ok]) + 0.5 * np.log(x[ok])))

        ax.plot(x, y, marker=markers[ci], linestyle='none', markersize=7,
                color=colors[ci], label=label_map[name])
        ax.plot(x_ref, C / np.sqrt(x_ref), '--', color=colors[ci], alpha=0.7)

    ax.plot([], [], '--', color='gray', alpha=0.7, label=r'$1/\sqrt{N}$ Convergence')
    ax.set_xlabel(r'N ($\times 10^3$)')
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="-", alpha=0.3)


def plot():
    d = np.load(CACHE_FILE)

    fig, ax = plt.subplots(figsize=(height_inch, height_inch))
    plot_std(ax, d, 'peak_loads', 1e6, r'$\sigma$ Peak $P_{surface}$ (MW/m$^2$)')
    ax.set_box_aspect(1)
    ax.legend(title='Average Cell Area', fontsize=10, title_fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_FILE, dpi=600, bbox_inches='tight')
    print(f"Saved {FIG_FILE}")
    plt.close(fig)


    # Combined: left = std of peak load, right = wetted area (mean +/- std), as in wall-pload-convergence.png
    with plt.rc_context({k: 10 for k in ['font.size', 'axes.labelsize', 'axes.titlesize',
                                         'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize']}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_inch, height_inch))
        plot_std(ax1, d, 'peak_loads', 1e6, r'$\sigma$ Peak $P_{surface}$ (MW/m$^2$)')
        ax1.set_box_aspect(1)
        ax1.legend(title='Average Cell Area', loc='upper right')

        dw = np.load(WET_CACHE_FILE)
        names = list(dw['names'])
        x = dw['marker_counts'] / 1000.0
        markers = ['o', 's', '^', 'D']
        for ci, name in enumerate(PLOT_NAMES):
            w = dw['wetted_areas'][names.index(name)]
            nb = np.sum(~np.isnan(w), axis=1)
            # No error bar where there is only one batch (N = full run)
            yerr = np.array([np.nanstd(row, ddof=1) if n > 1 else np.nan for row, n in zip(w, nb)])
            ax2.errorbar(x, np.nanmean(w, axis=1), yerr=yerr,
                         marker=markers[ci], linestyle='-', linewidth=2, capsize=5,
                         color=f'C{ci}', label=label_map[name])
        ax2.set_xlabel(r'N ($\times 10^3$)')
        ax2.set_ylabel(r'$A_{wetted}$ (m$^2$)')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_box_aspect(1)

        plt.tight_layout()
        plt.savefig(FIG_FILE_COMBINED, dpi=600, bbox_inches='tight')
        print(f"Saved {FIG_FILE_COMBINED}")
        plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("compute", "all"):
        compute()
    if mode in ("compute", "wetted", "all"):
        compute_wetted()
    if mode in ("plot", "all"):
        plot()
