'''
interactive_FresnelScale.py

Interactive GUI for exploring Fresnel diffraction patterns from stellar occultations.
Sliders control:
  - Distance to asteroid (AU)
  - Wavelength (nm)
  - Velocity on ground (km/s)
  - Stellar angular diameter (mas)

Two panels are shown in real-time:
  Left  — pattern vs. Fresnel-scaled distance (dimensionless)
  Right — pattern vs. time (seconds)

In each panel the raw Fresnel pattern is shown in a muted colour, and — when the
angular-diameter slider is above its minimum — the stellar-diameter-convolved curve
is overlaid in a bold colour.  A CheckButtons widget lets you toggle each curve on/off.

Derived quantities (Fresnel scale, projected stellar radius, fringe period, etc.)
are shown as live text below the sliders.

Author: Joshua Thomas Bartkoske
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import scipy.special as special

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

AU_TO_M   = 1.495978707e11   # 1 AU in metres
MAS_TO_RAD = np.pi / (180.0 * 3600.0 * 1000.0)  # 1 mas in radians

# ---------------------------------------------------------------------------
# Core physics functions
# (based on Occultations2.py / example_fresnelfit.py)
# ---------------------------------------------------------------------------

def fresnel_scale(wavelength_m, distance_m):
    """F = sqrt(λ · D / 2)  [metres]"""
    return np.sqrt(wavelength_m * distance_m / 2.0)


def generate_fresnel(x, distance_m, wavelength_m, I0=1.0):
    """Fresnel diffraction intensity for an infinite straight edge.

    Parameters
    ----------
    x            : array-like  distance from the geometric shadow edge [m]
    distance_m   : float       distance to the occulting body [m]
    wavelength_m : float       observing wavelength [m]
    I0           : float       normalisation coefficient (default 1)
    """
    u = np.sqrt(2.0 / (distance_m * wavelength_m)) * np.asarray(x, dtype=float)
    C, S = special.fresnel(u)
    return I0 * 0.5 * (np.square(0.5 + C) + np.square(0.5 + S))


def convolve_stellar_diameter(fresnel_pattern, x_m, distance_m, angular_diameter_mas):
    """Convolve a Fresnel pattern with the brightness profile of a finite stellar disk.

    Follows the same approach used in Occultations2.py / example_fresnelfit.py:
      1. Convert angular diameter (mas) → projected radius on the ground (metres).
      2. Build a circular (no limb-darkening) brightness-profile kernel.
      3. Pad the pattern, convolve, normalise by sum(kernel), strip padding.

    Parameters
    ----------
    fresnel_pattern     : ndarray  Fresnel intensity array (same length as x_m)
    x_m                 : ndarray  position array [m]
    distance_m          : float    distance to occulter [m]
    angular_diameter_mas: float    stellar angular diameter [mas]

    Returns
    -------
    convolved : ndarray  intensity after stellar-disk convolution, same shape as x_m
                         Returns fresnel_pattern unchanged if angular_diameter_mas <= 0.
    """
    if angular_diameter_mas <= 0.0:
        return fresnel_pattern.copy()

    angular_radius_rad = (angular_diameter_mas / 2.0) * MAS_TO_RAD
    # Alpha: half-width of the stellar disk projected onto the occultation plane [m]
    Alpha = distance_m * np.tan(angular_radius_rad)

    step_size = np.abs(x_m[1] - x_m[0])

    # Guard: kernel must have at least 3 points
    if step_size >= 2.0 * Alpha:
        # Star is unresolved at this sampling — return the bare Fresnel pattern
        return fresnel_pattern.copy()

    # Brightness-profile kernel  (circular disk, no limb-darkening)
    # S(φ) = 2·Alpha·sin(arccos(φ/Alpha))   for φ ∈ [-Alpha, Alpha]
    kernel_x = np.arange(-Alpha, Alpha, step_size)
    kernel    = 2.0 * Alpha * np.sin(np.arccos(np.clip(kernel_x / Alpha, -1, 1)))

    extra = len(kernel)   # padding width to avoid edge artefacts

    # Pad the Fresnel pattern with its boundary values (same as Occultations2.py)
    pad_before = np.ones(extra) * fresnel_pattern[0]
    pad_after  = np.ones(extra) * fresnel_pattern[-1]
    padded     = np.concatenate([pad_before, fresnel_pattern, pad_after])

    convolved  = np.convolve(kernel, padded, mode='same')

    # Normalise by the integral of the kernel
    norm_coeff = np.sum(kernel)
    convolved  = convolved / norm_coeff

    # Strip the padding
    return convolved[extra:-extra]

# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------

DIST_AU_DEFAULT  = 2.7     # AU
WAVE_NM_DEFAULT  = 435.0   # nm
VEL_KMS_DEFAULT  = 10.0    # km/s
ANGD_MAS_DEFAULT = 0.0     # mas  (0 = off)

DIST_AU_MIN,  DIST_AU_MAX  = 0.5,   10.0
WAVE_NM_MIN,  WAVE_NM_MAX  = 300.0, 1000.0
VEL_KMS_MIN,  VEL_KMS_MAX  = 1.0,   50.0
ANGD_MAS_MIN, ANGD_MAS_MAX = 0.0,   2.0    # mas

# x-axis for the diffraction pattern [m]
X_M = np.linspace(-1500, 5000, 12000)

# ---------------------------------------------------------------------------
# Build the figure layout
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 8))
fig.canvas.manager.set_window_title("Interactive Fresnel Scale Explorer")

# Plots: upper region; sliders + controls: lower region
plt.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.48, wspace=0.25)

ax_fs   = fig.add_subplot(1, 2, 1)   # left panel:  Fresnel-scaled distance
ax_time = fig.add_subplot(1, 2, 2)   # right panel: time axis

# ---- Slider axes -----------------------------------------------------------
#  [left, bottom, width, height]
ax_sl_dist = fig.add_axes([0.12, 0.375, 0.76, 0.025])
ax_sl_wave = fig.add_axes([0.12, 0.320, 0.76, 0.025])
ax_sl_vel  = fig.add_axes([0.12, 0.265, 0.76, 0.025])
ax_sl_angd = fig.add_axes([0.12, 0.210, 0.76, 0.025])

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

sl_angd = widgets.Slider(ax_sl_angd, 'Angular diam.  [mas]',
                         ANGD_MAS_MIN, ANGD_MAS_MAX,
                         valinit=ANGD_MAS_DEFAULT, valstep=0.01,
                         color='darkorange')

# ---- Visibility checkboxes -------------------------------------------------
ax_chk = fig.add_axes([0.57, 0.04, 0.18, 0.10])
ax_chk.set_facecolor('#f9f9f9')
chk_labels  = ['Fresnel (raw)', 'Stellar conv.']
chk_init    = [True, True]
chk_buttons = widgets.CheckButtons(ax_chk, chk_labels, chk_init)
# Style the check label colours to match the lines
for lbl, col in zip(chk_buttons.labels, ['steelblue', 'darkorange']):
    lbl.set_color(col)

# ---- Reset button ----------------------------------------------------------
ax_reset = fig.add_axes([0.80, 0.055, 0.10, 0.04])
btn_reset = widgets.Button(ax_reset, 'Reset', color='lightyellow', hovercolor='gold')

# ---- Info text box (derived quantities) ------------------------------------
info_ax = fig.add_axes([0.07, 0.04, 0.46, 0.10])
info_ax.axis('off')
info_text = info_ax.text(0.0, 1.0, '', transform=info_ax.transAxes,
                         fontsize=9.0, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.4',
                                   fc='#f0f4ff', ec='#aaaacc'))

# ---------------------------------------------------------------------------
# Plot line objects (updated in-place every call)
# ---------------------------------------------------------------------------

# Left panel (Fresnel-scaled axis)
(line_fs_raw,)  = ax_fs.plot([], [], color='steelblue',  lw=1.4,
                              alpha=0.7, label='Fresnel (raw)')
(line_fs_conv,) = ax_fs.plot([], [], color='darkorange', lw=2.0,
                              label='Stellar conv.')

# Right panel (time axis)
(line_t_raw,)   = ax_time.plot([], [], color='steelblue',  lw=1.4,
                                alpha=0.7, label='Fresnel (raw)')
(line_t_conv,)  = ax_time.plot([], [], color='darkorange', lw=2.0,
                                label='Stellar conv.')

# Reference horizontal lines
for ax in (ax_fs, ax_time):
    ax.axhline(0.25, color='grey', lw=0.7, ls='--', alpha=0.5)
    ax.axhline(1.00, color='grey', lw=0.7, ls='--', alpha=0.5)
    ax.set_ylim(-0.05, 1.55)
    ax.set_ylabel('Normalised intensity')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

ax_fs.set_xlabel('Fresnel-scaled distance  [F]')
ax_fs.set_title('Fresnel-scaled axis')
ax_time.set_xlabel('Time  [s]')
ax_time.set_title('Time axis')

title_text = fig.suptitle('', fontsize=11, fontweight='bold')

# ---------------------------------------------------------------------------
# Update function — called whenever any slider moves
# ---------------------------------------------------------------------------

def update(_=None):
    dist_m   = sl_dist.val * AU_TO_M      # AU  → m
    wave_m   = sl_wave.val * 1e-9         # nm  → m
    vel_ms   = sl_vel.val  * 1e3          # km/s→ m/s
    angd_mas = sl_angd.val                # mas (stays as-is)

    # ---- raw Fresnel pattern -----------------------------------------------
    F       = fresnel_scale(wave_m, dist_m)
    pattern = generate_fresnel(X_M, dist_m, wave_m)

    # ---- convolved pattern (stellar angular diameter) ----------------------
    conv_pattern = convolve_stellar_diameter(pattern, X_M, dist_m, angd_mas)

    # ---- axes data ---------------------------------------------------------
    x_scaled = X_M / F
    t_axis   = X_M / vel_ms

    line_fs_raw.set_data(x_scaled, pattern)
    line_fs_conv.set_data(x_scaled, conv_pattern)
    ax_fs.set_xlim(x_scaled[0], x_scaled[-1])

    line_t_raw.set_data(t_axis, pattern)
    line_t_conv.set_data(t_axis, conv_pattern)
    ax_time.set_xlim(t_axis[0], t_axis[-1])

    # ---- derived quantities ------------------------------------------------
    fringe_time_ms = (F / vel_ms) * 1e3

    # Projected stellar radius on the ground
    if angd_mas > 0.0:
        Alpha_m  = dist_m * np.tan((angd_mas / 2.0) * MAS_TO_RAD)
        Alpha_km = Alpha_m / 1e3
        Alpha_t_ms = (Alpha_m / vel_ms) * 1e3   # time for light to cross stellar radius
        alpha_str = (f'Projected stellar radius  α = {Alpha_m:.1f} m  '
                     f'({Alpha_km:.4f} km)\n'
                     f'Time to cross α             = {Alpha_t_ms:.3f} ms')
    else:
        alpha_str = 'Angular diameter  =  0  (no stellar convolution)'

    title_text.set_text(
        f'D = {sl_dist.val:.2f} AU   λ = {sl_wave.val:.0f} nm   '
        f'v = {sl_vel.val:.1f} km s⁻¹   θ = {angd_mas:.2f} mas   '
        f'F = {F:.1f} m'
    )

    info_str = (
        f'Fresnel scale  F = {F:.2f} m  ({F/1e3:.5f} km)\n'
        f'Time to cross F  = {fringe_time_ms:.3f} ms\n'
        f'Distance         = {dist_m/AU_TO_M:.3f} AU  ({dist_m/1e3:.3e} km)\n'
        + alpha_str
    )
    info_text.set_text(info_str)

    fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Checkbox callback — toggle line visibility
# ---------------------------------------------------------------------------

def toggle_visibility(label):
    show_raw  = chk_buttons.get_status()[0]
    show_conv = chk_buttons.get_status()[1]

    for line in (line_fs_raw, line_t_raw):
        line.set_visible(show_raw)
    for line in (line_fs_conv, line_t_conv):
        line.set_visible(show_conv)

    fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Reset callback
# ---------------------------------------------------------------------------

def reset(_):
    sl_dist.reset()
    sl_wave.reset()
    sl_vel.reset()
    sl_angd.reset()


# ---------------------------------------------------------------------------
# Wire up all callbacks
# ---------------------------------------------------------------------------

sl_dist.on_changed(update)
sl_wave.on_changed(update)
sl_vel.on_changed(update)
sl_angd.on_changed(update)
btn_reset.on_clicked(reset)
chk_buttons.on_clicked(toggle_visibility)

# Initial draw
update()

plt.show()
