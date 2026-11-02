# PyWopWop: research and plan

Goal: generate rotor sounds for flight simulation (FlightGear first) from the
physics of the rotor, faster than the established rotor-noise codes and with
audio that is fit to listen to, not just to plot.

## 1. What exists

| Tool | What it is | Strengths | Gaps for a flight-sim sound designer |
|---|---|---|---|
| **PSU-WOPWOP** (Penn State; v3.4.4 is what current users run) | Fortran solver of the Ffowcs Williams-Hawkings (FW-H) equation with Farassat's Formulation 1A (F1A). Thickness and loading noise, moving and maneuvering rotors, multiple rotors, source-time-dominant ("advanced time") algorithm. | The reference code for rotor tonal noise; validated against wind-tunnel and flight data. | Closed source (binary on request). Needs blade loads from somewhere else (CAMRAD II, CHARM, CFD, FLOWUnsteady). Broadband is left to BPM in other codes (ANOPP2, BPM.jl). Output is pressure histories and spectra, not loops. |
| **WOPWOP** (NASA Langley, 1986) | The original F1A code, retarded-time algorithm (iterates for the emission time). | Simple, well documented. | Superseded by PSU-WOPWOP; same input problem. |
| **ANOPP2** (NASA) | Noise framework: F1A, compact line sources, BPM self-noise (ASNIFM), propagation. | Tonal and broadband in one framework. | Not public. Works with aero from other codes. |
| **NASA Auralization Framework (NAF)** | Turns ANOPP2 predictions into sound: source-noise hemispheres, F1A synthesis plugin, modulated-broadband plugin, propagation with gain/delay/filters. | The state of the art in rotorcraft auralization: blade-passage modulation of broadband noise matters perceptually; computing at the exact emission angle avoids interpolation artifacts. | Not public. Tonal source data are typically computed at about 5 kHz sample rate (2.5 kHz bandwidth), so impulsive content above that is lost. |
| **FLOWUnsteady + FLOWNoise** (BYU, Julia) | Open aero solver (VPM) that writes PSU-WOPWOP cases and runs BPM.jl. | Open aero front end. | Still needs the closed PSU-WOPWOP binary; about a minute for a 72-microphone hover case. |
| **OpenFAST AeroAcoustics** (NREL) | BPM, Amiet turbulent inflow, Guidati, for wind turbines. | Open, documented broadband models. | Wind-turbine oriented, no tonal FW-H. |
| **fwhFoam / libAcoustics / OpenCFD-FWH / Antares** | Open FW-H post-processors for CFD. | Permeable surfaces, CFD coupling. | Need time-resolved CFD; no rotor aero, no audio. |
| **helinoise** | Python glue from CAMRAD II / Dymore to PSU-WOPWOP. | — | Needs both commercial/closed codes. |
| GPU / fast-multipole FW-H research (Bristol 2024, MDPI Acoustics 2023) | JIT + GPU mid-fidelity aero-acoustics; FMM-accelerated FW-H (about 2x). | Speed. | Research codes, not distributed. |

Sources: [PSU-WOPWOP maneuvering paper](https://www.sciencedirect.com/science/article/abs/pii/S0022460X03008630),
[PSU-WOPWOP validation](https://pure.psu.edu/en/publications/a-validation-study-of-the-psu-wopwop-rotor-noise-prediction-syste/),
[FLOWUnsteady acoustics example](https://flow.byu.edu/FLOWUnsteady/examples/rotorhover-acoustics),
[PSU-WOPWOP + ANOPP2 broadband (ITEA 2023)](https://itea.org/journals/volume-44-2/rotor-broadband-noise-modeling-and-propeller-wing-interaction/),
[NAF modulated broadband plugin (NASA)](https://ntrs.nasa.gov/api/citations/20210017214/downloads/AV21.pdf),
[NAF unsteady F1A auralization (NASA)](https://ntrs.nasa.gov/api/citations/20200002468/downloads/20200002468.pdf),
[BPM, NASA RP-1218](https://ntrs.nasa.gov/citations/19890016302),
[Farassat, NASA TM-2007-214853](https://ntrs.nasa.gov/citations/20070010579),
[OpenFAST noise models](https://openfast.readthedocs.io/en/dev/source/user/aerodyn-aeroacoustics/02-noise-models.html),
[fwhFoam](https://github.com/Sparsh-Sharma/fwhFoam), [OpenCFD-FWH](https://github.com/Z-K-L/OpenCFD-FWH),
[libAcoustics](https://github.com/unicfdlab/libAcoustics), [helinoise](https://github.com/kumar-sumeet/helinoise),
[GPU mid-fidelity aero-acoustics (Bristol)](https://research-information.bris.ac.uk/en/publications/a-gpu-accelerated-mid-fidelity-aerodynamic-and-aeroacoustic-simul/),
[FMM FW-H (MDPI)](https://www.mdpi.com/2624-599X/5/3/48),
[HART II lifting-chord BVI study (Kelly, Glasgow)](https://theses.gla.ac.uk/1513).

## 2. Where they fall short for this job

1. **No aerodynamics.** Every FW-H code needs blade loads from another program.
   A sound designer has a rotor's published geometry and a flight condition,
   not a CAMRAD II licence.
2. **No sound.** Outputs are pressure histories and spectra. Turning them into
   a loop that a simulator can pitch-shift without clicks or aliasing is left
   to the user.
3. **Broadband is a spectrum, not a signal.** The swish of a rotor is
   broadband noise modulated at the blade passing frequency; spectral BPM
   output loses it. Only NASA's non-public NAF synthesises it.
4. **Band limit.** NASA's tonal synthesis runs at about 5 kHz; blade slap and
   high-speed impulsive noise have content well above 2.5 kHz.
5. **Speed is spent where it doesn't need to be.** A steady rotor is periodic
   and its blades are identical; general-purpose codes compute every blade
   over many revolutions and interpolate in observer time.

## 3. How PyWopWop beats them

### Physics (one program, geometry in, sound out)

- **Real blade sections**: lift, drag and pitching moment tables against angle of attack and Mach number (C-81
  decks, GenHel), resampled by monotone cubics so the loads have no kinks to ring in the loading noise; the blade
  surface from the section's coordinates or its NACA definition, camber included.
- **Trimmed blade-element rotor model**: hover BEMT with Prandtl tip loss;
  forward flight with Drees inflow, Glauert momentum, collective and cyclic
  trimmed to the thrust needed for the flight condition, tip-path-plane
  angle from the aircraft's weight, drag and flight path.
- **Elastic blades, for any rotor** (main, tail, proprotor, propeller): rotating-beam flap modes (Galerkin, hinged
  or clamped root, centrifugal stiffening), so a hingeless blade's coning is its bent first mode, and a torsion
  mode. Flap and torsion are coupled through the section CG's offset from the elastic axis (inertia, and the
  centrifugal force through a sloped blade) and through the aerodynamics (lift from elastic twist, moment from
  flap velocity). The periodic response is solved by harmonic balance inside the trim, with the linearised
  aerodynamics as an approximate Jacobian. Data per preset: mode frequencies, elastic axis, CG, Lock number
  (e.g. Bo 105: flap 1.12/rev, torsion 3.79/rev, elastic axis 17.95 %, CG 24.4 % chord). Deflections feed the
  angle of attack (flap velocity, slope, elastic twist) and the wake (release height and blade height: the
  blade-vortex miss distance, which HART II showed decides BVI noise).
- **Blade-vortex interaction (the "wop wop")**: a Beddoes-type prescribed tip
  vortex wake (released at the tip with the peak bound circulation, carried
  by the free stream and the Beddoes inflow distribution, doubled downstream
  of the disc), Vatistas cores growing by Squire's law, Biot-Savart induced
  velocity on the blades, and Küssner indicial response with compressibility
  scaling so the impulsive loads are physically smoothed.
- **F1A thickness noise** from the real blade surface (NACA thickness
  distribution, closed root and tip) and **F1A loading noise** from the
  lifting line, optionally spread over the chord with the thin-airfoil
  loading shape (non-compact loading: the right high-frequency roll-off of
  blade slap; HART II work found lifting-chord models better than lifting
  line for BVI).
- **Broadband**: the BPM model ported line for line from the NASA RP-1218
  Fortran and checked against its published test cases; Amiet turbulent
  inflow noise; both computed per blade segment and azimuth with retarded
  geometry, BPM directivity and Doppler shift, giving a spectral time history
  over the revolution.
- **Propagation**: ISO 9613-1 atmospheric absorption, ground reflection by an
  image observer, spherical spreading; exact Doppler from the FW-H retarded
  time for fly-bys.

### Computational efficiency

- **Closed-form reception time** for observers in uniform motion (co-moving
  cockpit or sound-bank observer, or a ground observer): a quadratic, no
  iteration.
- **Source-time-dominant accumulation** onto the observer grid in a Numba
  kernel, parallel over elements, with kinematics computed analytically on the
  fly (no large intermediate arrays).
- **Periodicity and blade symmetry** for steady flight: one blade, one
  revolution, then the other blades are phase shifts of the harmonics. B times
  less work, and the result is an exact Fourier series.
- **Thickness noise cached** across load conditions (it depends only on
  geometry and kinematics).
- **Spectral load derivatives** (exact for periodic loads) instead of finite
  differences.

### Sound quality

- **Band-limited additive synthesis from the exact harmonics**, as in music
  synthesizers' wavetables: no aliasing by construction, and the band limit is
  set below Nyquist divided by the largest pitch factor the simulator applies,
  so pitch-shifting by rotor RPM never folds harmonics back.
- **Sample-exact seamless loops**: an integer number of revolutions in an
  integer number of samples (the rotor speed is nudged by well under a cent),
  so the loop is a true period, not a crossfade.
- **Modulated broadband** in the spirit of spectral modelling synthesis
  (deterministic + stochastic parts): per-band noise, periodic over the loop,
  shaped by the blade-passage envelope from the spectral time history.
- **Natural unsteadiness**: small random blade-to-blade loading fluctuations,
  blade tracking differences and RPM wander, all periodic over the loop, so a
  steady condition doesn't sound like a drum machine.
- **Full band**: harmonics up to 20 kHz where the physics puts energy there.

### Flight simulation output

- Presets from published data for the Bo 105, UH-60A, UH-1H, XV-15 and a
  small quadcopter propeller, plus TOML files for your own rotors.
- Sound banks: hover, cruise speeds, BVI descent, climb; external directions
  and the cockpit; main and tail rotor separately; loops with consistent
  relative levels, a manifest with absolute SPL, and a FlightGear sound XML
  snippet that blends them by airspeed, vertical speed and rotor RPM.
- Fly-by rendering for a ground observer to audition the result.

## 4. Validation

| Check | Against |
|---|---|
| BPM port | Tables D2 and D3 of NASA RP-1218 (the code's own test output) |
| F1A loading | Gutin's closed-form hover harmonics (far field) |
| F1A thickness and loading | Formulation 1 (numerical observer-time derivative), an independent route to the same answer |
| Retarded time | Direct numerical solution of the retarded-time equation |
| BEMT | Momentum theory (ideal induced power) |
| Blade modes | Published rotating frequencies (Bo 105: second flap predicted 2.83/rev from the first's 1.12, published 2.78) |
| Absorption | ISO 9613-2 table values |
| Loops | Sample-exact periodicity, no energy above the band limit |

## 5. Status and measured results (2026-11-01)

Built and passing (`pywopwop validate`, `pytest`):

| Check | Result |
|---|---|
| BPM port vs NASA RP-1218 table D2 / D3 | 0.006 dB / 0.02 dB (tip column 0.19 dB: the printed table used a 0.3 % different Strouhal number) |
| F1A loading vs Gutin | 0.08 dB |
| F1A vs Formulation 1, unsteady loads, elastic heave | 0.1 % (hover), 0.5 % (fly-by, fixed listener) |
| ISO 9613-1 vs ISO 9613-2 table | equal as printed |
| UH-60A hover figure of merit | 0.73 |
| Bo 105 second flap frequency from the first | 2.825/rev (published 2.78) |
| Loops | nothing above the band limit (-315 dB), no step at the loop point |

Speed on an Apple M1 (8 cores), measured by `pywopwop bench`:

| Task | Time |
|---|---|
| Compact loading, 40 stations x 4 blades, 32768 samples/rev, one observer: classic retarded-time algorithm (Newton per observer sample, as the original WOPWOP), same compiler | 188 ms |
| The same with PyWopWop's source-time-dominant, closed-form, blade-symmetric kernel | 20 ms (9x), identical result |
| UH-60A 65 kt descent: trim, elastic response, wake | 1.6 s |
| F1A at one observer: 160 load points + 1966 surface panels | 0.23 s (about 90 M source evaluations/s) |
| Broadband spectral history | 0.1 s |
| 8 s seamless loop, main and tail rotor, with 8 unsteady revolutions, scintillation and surroundings | about 15 s |
| A bank: 7 conditions x 5 positions, mixes and layers (170 loops) | about 10 min |
| 16 s fly-by past a ground listener, every blade, exact Doppler, ground reflection | about 50 s |

PSU-WOPWOP itself was not timed: its binary is distributed on request by Penn State. The one published figure
found (FLOWUnsteady's example: a 72-microphone hover case in about a minute on a laptop) is for a different problem
size, so no ratio is claimed.

### Listening round (2026-11-01)

The first loops passed every physics check and still sounded like "a square wave generator, or a lazy machine
gun". Measured against the recordings in the FlightGear UH-60 and Bo 105 packages, they were:

| | Recordings | First loops | Now (UH-60A bank) |
|---|---|---|---|
| Correlation of consecutive blade passages | 0.17-0.58 | 0.99 (approach) | 0.14 hover, 0.27 cruise, 0.54 approach |
| Harmonics against noise, 100 Hz-3 kHz | -6 to -11 dB | +6 dB (approach) | -8, -7, -2 dB |
| Approach against cruise, dBA at 100 m | (a few dB in certification data) | +25 dB | +10 dB |

What was missing, now in: pass-to-pass unsteadiness (revolutions with their own turbulent gusts and tip-vortex
wander, morphed blade by blade), blade-wake interaction broadband (Landgrebe's hover wake, Beddoes' in forward
flight, Amiet's inflow noise), the outside listener's ground reflection and diffuse field, turbulent-air
scintillation, a recording chain's 25 Hz high-pass; and the blade-vortex loads were three to six times HART II's
(a lifting line with the measured core), so the effective core and roll-up are now calibrated to them. Absolute
levels are still not calibrated against flight tests.

## 6. Limits (stated, not hidden)

- Linear acoustics: no quadrupole (transonic delocalisation above about
  M 0.9 advancing tip), like PSU-WOPWOP with impermeable surfaces.
- Elastic blades are modal (two flap modes, one torsion mode, uniform mass and stiffness tuned to the published
  frequencies); no lag motion. A blade near a flap-torsion resonance (the Bo 105's 3.79/rev torsion) is sensitive
  to the section data.
- Prescribed, not free, wake; one tip vortex per blade.
- Fuselage scattering, engine, gearbox and main-rotor/tail-rotor wake
  interaction are not modelled.
- One section per rotor. The Bo 105, UH-60A and UH-1H use their published tables (C-81 decks, GenHel, imported
  from the FlightGear JSBSim models' data) and real shapes; the XV-15 and the quadcopter use the smooth section
  model, their tables not being public.
