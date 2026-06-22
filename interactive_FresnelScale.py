'''
interactive_FresnelScale.py

Interactive GUI for exploring Fresnel diffraction patterns from stellar occultations.
Sliders control:
  - Distance to asteroid (AU)
  - Wavelength (nm)
  - Velocity on ground (km/s)

Two panels are shown in real-time:
  Left  — pattern vs. Fresnel-scaled distance (dimensionless)
  Right — pattern vs. time (seconds)

Derived quantities (Fresnel scale, fringe period, etc.) are shown as live text.

Author: Joshua Thomas Bartkoske
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import scipy.special as special

# ---------------------------------------------------------------------------
# Physical helpers
# ---------------------------------------------------------------------------

AU_TO_M = 1.495978707e11   # 1 AU in meters

def fresnel_scale(wavelength_m, distance_m):
    """F = sqrt(lambda * D / 2)  [meters]"""
    return np.sqrt(wavelength_m * distance_m / 2.0)

def generate_fresnel(x, distance_m, wavelength_m, I0=1.0):
    """Fresnel diffraction pattern for an infinite straight edge.

    Parameters
    ----------
    x            : array-like, distance from the geometric shadow edge [m]
    distance_m   : distance to the occulting body [m]
    wavelength_m : observing wavelength [m]
    I0           : normalisation coefficient (default 1)

    Returns
    -------
    pattern : ndarray, normalised intensity
    """
    u = np.sqrt(2.0 / (distance_m * wavelength_m)) * np.asarray(x)
    C, S = special.fresnel(u)
    pattern = I0 * 0.5 * (np.square(0.5 + C) + np.square(0.5 + S))
    return pattern

# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------

DIST_AU_DEFAULT  = 2.7          # AU
WAVE_NM_DEFAULT  = 435.0        # nm
VEL_KMS_DEFAULT  = 10.0         # km/s

# Slider ranges
DIST_AU_MIN,  DIST_AU_MAX  = 0.5, 10.0
WAVE_NM_MIN,  WAVE_NM_MAX  = 300.0, 1000.0
VEL_KMS_MIN,  VEL_KMS_MAX  = 1.0,  50.0

# x-axis for the diffraction pattern (distance from shadow edge, meters)
X_M = np.linspace(-1500, 5000, 12000)

# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(13, 7))
fig.canvas.manager.set_window_title("Interactive Fresnel Scale Explorer")

# Reserve space: plots occupy the upper portion, sliders the lower
plt.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.42, wspace=0.25)

ax_fs   = fig.add_subplot(1, 2, 1)   # Fresnel-scaled distance
ax_time = fig.add_subplot(1, 2, 2)   # Time axis

# ---- Slider axes (manually placed) ----------------------------------------
# [left, bottom, width, height]  (all in figure-fraction coordinates)
ax_sl_dist = fig.add_axes([0.12, 0.28, 0.76, 0.03])
ax_sl_wave = fig.add_axes([0.12, 0.21, 0.76, 0.03])
ax_sl_vel  = fig.add_axes([0.12, 0.14, 0.76, 0.03])

sl_dist = widgets.Slider(ax_sl_dist, 'Distance  [AU]',
                         DIST_AU_MIN, DIST_AU_MAX,
                         valinit=DIST_AU_DEFAULT, valstep=0.05,
                         color='steelblue')

sl_wave = widgets.Slider(ax_sl_wave, 'Wavelength  [nm]',
                         WAVE_NM_MIN, WAVE_NM_MAX,
                         valinit=WAVE_NM_DEFAULT, valstep=1.0,
                         color='mediumpurple')

sl_vel  = widgets.Slider(ax_sl_vel,  'Velocity  [km s⁻¹]',
                         VEL_KMS_MIN, VEL_KMS_MAX,
                         valinit=VEL_KMS_DEFAULT, valstep=0.5,
                         color='seagreen')

# Reset button
ax_reset = fig.add_axes([0.45, 0.05, 0.10, 0.04])
btn_reset = widgets.Button(ax_reset, 'Reset', color='lightyellow', hovercolor='gold')

# Info text box (derived quantities)
info_ax = fig.add_axes([0.12, 0.05, 0.30, 0.07])
info_ax.axis('off')
info_text = info_ax.text(0.0, 1.0, '', transform=info_ax.transAxes,
                         fontsize=9.5, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.4', fc='#f0f4ff', ec='#aaaacc'))

# ---------------------------------------------------------------------------
# Initial plot objects (will be updated in-place)
# ---------------------------------------------------------------------------

(line_fs,)   = ax_fs.plot([], [], color='steelblue', lw=1.5)
(line_time,) = ax_time.plot([], [], color='tomato',  lw=1.5)

# Reference line at 0.25 (geometric shadow edge)
for ax in (ax_fs, ax_time):
    ax.axhline(0.25, color='grey', lw=0.7, ls='--', alpha=0.6)
    ax.axhline(1.00, color='grey', lw=0.7, ls='--', alpha=0.6)
    ax.set_ylim(-0.05, 1.55)
    ax.set_ylabel('Normalised intensity')
    ax.grid(True, alpha=0.3)

ax_fs.set_xlabel('Fresnel-scaled distance  [F]')
ax_fs.set_title('Fresnel-scaled axis')

ax_time.set_xlabel('Time  [s]')
ax_time.set_title('Time axis')

# Main title (with derived info)
title_text = fig.suptitle('', fontsize=12, fontweight='bold')

# ---------------------------------------------------------------------------
# Update function
# ---------------------------------------------------------------------------

def update(_=None):
    dist_m  = sl_dist.val * AU_TO_M          # AU → m
    wave_m  = sl_wave.val * 1e-9             # nm → m
    vel_ms  = sl_vel.val  * 1e3              # km/s → m/s

    # --- compute pattern ---
    F       = fresnel_scale(wave_m, dist_m)
    pattern = generate_fresnel(X_M, dist_m, wave_m)

    # --- Fresnel-scaled axis ---
    x_scaled = X_M / F
    line_fs.set_data(x_scaled, pattern)
    ax_fs.set_xlim(x_scaled[0], x_scaled[-1])

    # --- time axis ---
    t_axis = X_M / vel_ms
    line_time.set_data(t_axis, pattern)
    ax_time.set_xlim(t_axis[0], t_axis[-1])

    # --- derived quantities ---
    fringe_time_ms  = (F / vel_ms) * 1e3          # time to cross one Fresnel scale [ms]
    fringe_size_km  = F / 1e3                      # Fresnel scale in km
    dist_km         = dist_m / 1e3
    dist_au         = sl_dist.val

    title_text.set_text(
        f'Distance = {dist_au:.2f} AU  |  '
        f'λ = {sl_wave.val:.0f} nm  |  '
        f'v = {sl_vel.val:.1f} km s⁻¹  |  '
        f'Fresnel scale F = {F:.1f} m'
    )

    info_str = (
        f'Fresnel scale  F = {F:.2f} m  ({fringe_size_km:.4f} km)\n'
        f'Time to cross F  = {fringe_time_ms:.3f} ms\n'
        f'Distance         = {dist_km:.3e} km  ({dist_au:.3f} AU)'
    )
    info_text.set_text(info_str)

    fig.canvas.draw_idle()

# ---------------------------------------------------------------------------
# Reset callback
# ---------------------------------------------------------------------------

def reset(_):
    sl_dist.reset()
    sl_wave.reset()
    sl_vel.reset()

# ---------------------------------------------------------------------------
# Wire up callbacks
# ---------------------------------------------------------------------------

sl_dist.on_changed(update)
sl_wave.on_changed(update)
sl_vel.on_changed(update)
btn_reset.on_clicked(reset)

# Draw initial state
update()

plt.show()
