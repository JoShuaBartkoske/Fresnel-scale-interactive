# Fresnel-scale-interactive
An interactive GUI to understand how the diffraction pattern at the edge of an occulting body changes based on various parameters including the distance to the occulting body, the wavelength of light, and the velocity of the occulting body relative to the observer.

All code is developed with the help of Kiro and other AI tools. The goal of this project is as both an example AI project and as an educational tool for use in public contexts.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/joshuabartkoske/Fresnel-scale-interactive.git
cd Fresnel-scale-interactive
```

### 2. Set up the environment

Choose whichever package manager you prefer.

**Using pip (recommended for most users):**
```bash
pip install -r requirements.txt
```

**Using conda:**
```bash
conda env create -f environment.yml
conda activate fresnel-interactive
```

Python 3.9 or higher is required.

### 3. Run the GUI

```bash
python interactive_FresnelScale.py
```

---

## Using the GUI

The window opens with two side-by-side plots showing the Fresnel diffraction pattern for a straight-edge occultation:

- **Left panel** — intensity as a function of Fresnel-scaled distance (dimensionless)
- **Right panel** — the same pattern mapped to a time axis in seconds

Three sliders let you explore how the pattern responds to different physical conditions:

| Slider | Range | Description |
|---|---|---|
| Distance [AU] | 0.5 – 10 AU | Distance from Earth to the occulting body |
| Wavelength [nm] | 300 – 1000 nm | Observing wavelength of light |
| Velocity [km s⁻¹] | 1 – 50 km s⁻¹ | Speed of the shadow crossing the observer |

A live info box below the sliders displays derived quantities including the Fresnel scale in meters and kilometres, and the time it takes the shadow to cross one Fresnel scale.

Press **Reset** at any time to return all sliders to their default values.
