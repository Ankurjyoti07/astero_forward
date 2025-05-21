import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from astropy.timeseries import LombScargle

# Load light curve data
data_path = "/home/c4011027/PhD_stuff/TESS_proposal/tesszet_Oph_Sector_91_LC_data_interactive.txt"
time, flux = np.loadtxt(data_path, unpack=True, delimiter=',')

# Load frequencies from external file, skipping "----------"
frequency_file = "/home/c4011027/PhD_stuff/TESS_proposal/prologs/freq.txt"
frequencies = []
with open(frequency_file, 'r') as f:
    for line in f:
        if line.strip() == "----------":
            break
        try:
            frequencies.append(float(line.strip()))
        except ValueError:
            continue
frequencies = np.array(frequencies)
periods_all = 1 / frequencies

manual_mode = [False]
manual_frequencies = []
manual_vlines = []
separator_line = "----------\n"

# Compute Lomb-Scargle periodogram
freq_grid = np.linspace(0.01, 20, 10000)
ls = LombScargle(time, flux)
power = ls.power(freq_grid)
period_grid = 1 / freq_grid
sort_idx = np.argsort(period_grid)
period_grid, power = period_grid[sort_idx], power[sort_idx]

# Create 2-panel layout: top periodogram and bottom period spacing
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})
fig.subplots_adjust(hspace=0.05)

# === Panel 1: Periodogram in period space ===
ax1.plot(period_grid, power, color='black')
vlines = []
for p in periods_all:
    line = ax1.axvline(p, color='red', linestyle='--', alpha=0.7)
    vlines.append(line)
for line in manual_vlines:
    ax1.add_line(line)
ax1.set_ylabel("Power")
ax1.set_title("Periodogram (Period Space)")

# === Panel 2: Period Spacing ===
points_scatter = ax2.scatter([], [], color='blue')
ax2.set_ylabel("ΔP [sec]")
ax2.set_xlabel("Period [days]")

selected_periods = []
boundary_lines = []
select_mode = [False]
boundary_mode = [False]
tolerance_days = 0.01

# GLOBAL variable to track mouse x-position over ax1
current_mouse_x = None

def on_mouse_move(event):
    global current_mouse_x
    if event.inaxes == ax1:
        current_mouse_x = event.xdata
    else:
        current_mouse_x = None

def on_key(event):
    global current_mouse_x
    if event.key == 'a':
        select_mode[0] = True
        boundary_mode[0] = False
        manual_mode[0] = False
        print("Selection mode: ON")
    elif event.key == 'n':
        select_mode[0] = False
        boundary_mode[0] = False
        manual_mode[0] = False
        print("Selection mode: OFF")
    elif event.key == 'b':
        boundary_mode[0] = True
        select_mode[0] = False
        manual_mode[0] = False
        print("Boundary mode: ON (click to add)")
    elif event.key == 'c':
        boundary_mode[0] = False
        print("Boundary mode: OFF")
    elif event.key == 'o':
        manual_mode[0] = True
        select_mode[0] = False
        boundary_mode[0] = False
        print("Manual frequency mode: ON")
    elif event.key == 'f':
        manual_mode[0] = False
        print("Manual frequency mode: OFF")
    elif event.key == 'd':
        # Delete closest selected frequency using current_mouse_x
        if not selected_periods:
            return
        if current_mouse_x is None:
            print("Move mouse over periodogram before pressing 'd' to unselect.")
            return
        distances = [abs(p - current_mouse_x) for p in selected_periods]
        min_idx = np.argmin(distances)
        to_remove = selected_periods[min_idx]
        selected_periods.remove(to_remove)

        # Reset line color for deselected frequency
        for line in vlines + manual_vlines:
            x = line.get_xdata()[0]
            if abs(x - to_remove) < 1e-10:
                line.set_color('red' if line in vlines else 'green')
                break

        update_spacing_plot()
        print(f"Unselected frequency near {to_remove:.6f} days")
    elif event.key == 'g':
        # Delete closest manual frequency (only if manual_mode ON)
        if not manual_mode[0] or not manual_frequencies:
            return
        # We can't rely on event.guiEvent.xdata here, so use current_mouse_x again
        if current_mouse_x is None:
            print("Move mouse over periodogram before pressing 'g' to delete manual frequency.")
            return
        manual_periods = [1/f for f in manual_frequencies]
        distances = [abs(p - current_mouse_x) for p in manual_periods]
        min_idx = np.argmin(distances)
        freq_to_remove = manual_frequencies[min_idx]
        period_to_remove = manual_periods[min_idx]

        manual_frequencies.pop(min_idx)
        manual_vlines[min_idx].remove()
        manual_vlines.pop(min_idx)

        if period_to_remove in selected_periods:
            selected_periods.remove(period_to_remove)

        update_frequency_file()
        update_spacing_plot()
        fig.canvas.draw_idle()
        print(f"Deleted manual frequency: {freq_to_remove:.6f}")

def in_shadow(p):
    return any(abs(p - b) < 1e-10 for b in boundary_lines)  # No shaded regions now, just lines

def on_click(event):
    if event.inaxes != ax1 or event.button != 1:
        return

    if boundary_mode[0]:
        boundary_x = event.xdata
        boundary_lines.append(boundary_x)
        ax1.axvline(boundary_x, color='gray', linestyle='-', alpha=0.7)
        update_spacing_plot()
        fig.canvas.draw_idle()
        print(f"Boundary added at: {boundary_x:.6f}")
        return

    if manual_mode[0]:
        freq = 1 / event.xdata
        period = event.xdata
        manual_frequencies.append(freq)
        line = ax1.axvline(period, color='green', linestyle='--', alpha=0.8)
        manual_vlines.append(line)
        update_frequency_file()
        update_spacing_plot()
        print(f"Manually added frequency: {freq:.6f}")
        return

    if fig.canvas.toolbar.mode != '' or not select_mode[0]:
        return

    click_x = event.xdata
    all_lines = vlines + manual_vlines
    distances = [abs(click_x - line.get_xdata()[0]) for line in all_lines]
    nearest_idx = np.argmin(distances)
    nearest_period = all_lines[nearest_idx].get_xdata()[0]

    if distances[nearest_idx] < tolerance_days:
        if nearest_period not in selected_periods:
            selected_periods.append(nearest_period)
            all_lines[nearest_idx].set_color('blue')
            update_spacing_plot()
            print(f"Selected: {nearest_period:.6f} days")

def crosses_boundary(p1, p2):
    mid = 0.5 * (p1 + p2)
    return any(abs(mid - b) < abs(p2 - p1)/2 for b in boundary_lines)

def update_spacing_plot():
    periods = sorted(selected_periods)
    if len(periods) < 2:
        points_scatter.set_offsets(np.empty((0, 2)))
        fig.canvas.draw_idle()
        return

    x_vals = []
    y_vals = []

    for i in range(len(periods) - 1):
        p1, p2 = periods[i], periods[i+1]
        if not crosses_boundary(p1, p2):
            dp = (p2 - p1) * 86400
            x_vals.append(0.5 * (p1 + p2))
            y_vals.append(dp)

    points_scatter.set_offsets(np.c_[x_vals, y_vals])
    fig.canvas.draw_idle()

def update_frequency_file():
    with open(frequency_file, 'r') as f:
        lines = f.readlines()

    if separator_line in lines:
        idx = lines.index(separator_line)
        base = lines[:idx + 1]
    else:
        base = lines + [separator_line]

    # Remove old manual frequencies after separator line
    base = base[:base.index(separator_line)+1] if separator_line in base else base

    for freq in manual_frequencies:
        base.append(f"{freq:.8f}\n")

    with open(frequency_file, 'w') as f:
        f.writelines(base)

# Connect mouse move event to track mouse x-position over ax1
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Connect key and click events
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
