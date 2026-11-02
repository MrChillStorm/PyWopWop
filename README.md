# PyWopWop

Rotor sounds for flight simulators, computed from the physics of the rotor: the blades' aerodynamics, their
elastic response, the tip-vortex wake that makes the blade slap, and the Ffowcs Williams-Hawkings acoustics. The
output is seamless, alias-free WAV loops, with banks of them per flight condition, for the cockpit and for
outside, and a FlightGear sound configuration that blends them.

You give it a rotor's published geometry: radius, blades, chord, twist, airfoil, rpm, and a few structural
frequencies. No CFD and no external comprehensive analysis are needed.

## What it computes

| Stage | Model |
|---|---|
| Sections | Lift, drag and pitching moment tables against angle of attack and Mach number (C-81 decks, GenHel), smoothed for acoustics; real section shapes (coordinate files, NACA four- and five-digit with camber) for the blade surface |
| Trim | Aircraft weight and drag balanced by the rotor; collective and cyclic for the thrust with no flapping relative to the tip-path plane; tail rotor carries the main rotor's torque |
| Inflow | Blade-element momentum theory with Prandtl tip loss in the hover, Glauert and Drees in forward flight, an empirical bridge through the vortex ring state |
| Elastic blades | Rotating-beam flap modes (hinged or clamped root; a hingeless rotor cones by bending) and a torsion mode, coupled by the section CG's offset and the aerodynamics, solved by harmonic balance |
| Blade-vortex interaction | Beddoes' prescribed tip-vortex wake released from the deflected tip, Vatistas cores growing with age, Biot-Savart velocity sampled over the chord, Wagner's indicial lift response; the effective core and roll-up calibrated so the load swings match HART II's measured ones |
| Unsteadiness | Revolutions computed with their own turbulent gusts (frozen von Kármán field) and tip-vortex wander and strength; each blade morphs between them from one passage to the next, so no two slaps are alike |
| Blade-wake interaction | The turbulence around the older tip vortices met by the following blades (Landgrebe's hover wake, Beddoes' in forward flight), through Amiet's inflow noise: the swish and the crackle around the slap |
| Tonal noise | Farassat's Formulation 1A: thickness from the real blade surface, loading over the chord; closed-form reception times, analytic kinematics including the elastic motion |
| Broadband noise | BPM airfoil self-noise (ported from the NASA RP-1218 Fortran) and Amiet turbulent inflow noise, per segment and azimuth, Doppler shifted: a spectral history over the revolution |
| Synthesis | Additive synthesis from the exact harmonics in sample-exact loops; band noise shaped by the blade-passage envelope; small random gusts, blade-tracking differences and rpm wander |
| Propagation | ISO 9613-1 absorption; turbulent-air scintillation (per octave band, slowly fluctuating delay and level growing with distance and frequency); for fly-bys, exact Doppler |
| Surroundings | Outside listeners stand on the ground: the reflection (image source, Delany-Bazley grass) and a light diffuse outdoor field (`--environment free/ground/open/urban`) |

[docs/PLAN.md](docs/PLAN.md) has the survey of the established codes (PSU-WOPWOP, WOPWOP, ANOPP2 and NASA's
auralization framework, FLOWUnsteady, OpenFAST and the open FW-H solvers) and how this one differs.

## Install

```bash
pipx install git+https://github.com/MrChillStorm/PyWopWop.git
```

or, from a checkout, `pip install -e .` (Python 3.11 or newer; NumPy, SciPy and Numba).

## Use

```bash
pywopwop presets
pywopwop trim uh60a --speed 65 --climb -800
pywopwop render uh1h --speed 70 --climb -700 --observer front_right -o huey_slap.wav --plot huey_slap.png
pywopwop render bo105 --speed 100 --cockpit -o bo105_cruise_cockpit.wav
pywopwop bank uh60a -o out/uh60a
pywopwop bank uh60a -o out/uh60a --conditions hover_ige --update
pywopwop trim mv22b --speed 220 --nacelle 0
pywopwop bank mv22b --mode airplane -o out/mv22b_airplane
python tools/slap_map.py mv22b -o out/mv22b/slap_map.json
pywopwop wind mv22b -o out/mv22b_wind --gain-from out/mv22b
pywopwop flyover uh1h --speed 80 --climb -500 --height 60 --offset 30 -o huey_flyby.wav
pywopwop play out/uh60a/bvi_front_right_rotor.wav out/uh60a/bvi_front_right_slap.wav --gain 1 0.5
pywopwop validate
pywopwop bench
```

`play` loops WAV files gaplessly, several at once (a bank's layers, with `--gain` per file), the way a simulator
does: one audio stream stays open and each file wraps around inside it, so there is no click at the loop point.
Media players such as Audacity and VLC restart playback at the loop point, which can click. It needs the
`sounddevice` package (`pip install 'pywopwop[play]'`).

`render` writes one loop; `--hub-agl` puts the main rotor hub that high above the ground (ground effect, by
Cheeseman and Bennett, and the recirculating downwash's turbulence). Listening positions: `front`, `front_right`, `side`, `rear`, `below`, `left`
(100 m from the main rotor hub, riding along with the aircraft), `--az --el --dist` for any other, or
`--cockpit` (the preset's pilot position, heard through the cabin wall). `--stems` also writes every rotor and
component.

`bank` renders every flight condition (hover out of and in ground effect, translational lift, slow and fast
cruise, climb, the approach descent with blade slap and a steep descent whose wake passes above the disc) at four outside positions and the cockpit. For each it writes the mix and four layers that
add up to it:

| Layer | What it is |
|---|---|
| `rotor` | main rotor thickness and loading noise, without the blade slap |
| `slap` | the blade-vortex interaction alone (only where there is any) |
| `swish` | the main rotor's broadband noise, modulated by the blade passage |
| `tail` | the tail rotor |

The outside loops share one gain and the cockpit loops another, so within each set the relative loudness is the
physics' (the cockpit's near-field rotor thump is far louder than anything 100 m outside, and a simulator plays the
two sets separately anyway); `manifest.json` gives both gains. `manifest.json` lists each file's
condition, position and level (dB and dBA re 20 µPa). `pywopwop-sound.xml` is a FlightGear `<fx>` section that
plays the layers: the level-flight loops cross-fade with airspeed, the descent loops come in with the rate of
descent, the hover in ground effect takes over from the hover below about one rotor radius above the ground
(`position/altitude-agl-ft`), and every loop's pitch follows its rotor's rpm (`rotors/main/rpm`, `rotors/tail/rpm`). The loops are
band limited for pitch factors up to 1.25 (`--headroom`), so pitching up never aliases. Each loop has one `<volume>`,
an expression that multiplies its fades: a `<volume>`'s own factor, offset, minimum and maximum can't make a fade in
FlightGear, which clips the factor times the property before adding the offsets and reads at most five of them.

Tiltrotors: `--mode airplane` renders the airplane mode's conditions (nacelles 0: 180, 220 and 260 kt, a climb and a
descent) from the preset's `airplane_trim` and `airplane_rpm`; a simulator fades the two banks into each other with
the nacelle angle.

`tools/slap_map.py` maps the blade slap's level over the rotor's advance ratio and its inflow through the tip-path
plane (trims over a grid of airspeed and vertical speed): a flight model that computes these for its rotors can
drive a bank's slap loops in any manoeuvre (a flare, a turn), not only in the bank's steady descents. The
FlightGear V-22 does.

`wind` renders the cockpit's wind: the windscreen's turbulent boundary layer (a flat plate's from the nose,
`windscreen_m` aft of it), its wall pressure by Goody's spectral shape at Farabee and Casarella's measured level (Bies's
flight data within about 1 dB), through the cabin wall as Levine and DeFelice predict a helicopter's cabin noise
(NASA, 1978). Loops at sea level from 30 to 350 kt, at a bank's cockpit gain (`--gain-from`), and a FlightGear section
that cross-fades them by the equivalent sea level airspeed V_eq = V (nu/nu_0)^-1/6 and scales them by
sqrt(sigma mu/mu_0) for the altitude, from FlightGear's equivalent airspeed and the air's density and temperature: the
same spectrum as the full model's within 0.5 dB per third octave to 12 km, 1.2 dB at 21 km.

## Gliders

A glider has no engine; what is heard is the air. `pywopwop glider` and `glider-bank` compute it from the glider's
geometry and sections (preset: `ask21`, from the FlightGear ASK 21 JSBSim model's geometry):

| Source | Model |
|---|---|
| Wing and tail | BPM self-noise on the real sections' boundary layers: NeuralFoil gives each strip's trailing-edge displacement thicknesses, transition points and separated share of the chord (FX S 02-196, FX 60-126, FX 71-L-150) at its Reynolds number and angle of attack; lift along the span by Schrenk |
| Turbulence | Amiet's inflow noise from the air's (or a thermal's) turbulence, and slow level changes as gusts move the airspeed |
| Canopy wind | the fuselage's potential flow (a panel method on its contours, with the wing's lifting line) at the body's angle of attack and sideslip; each panel of the glass gets the boundary layer of the surface streamline that reaches it, traced back to the nose: laminar to the canopy frame and turbulent from there, thickening where the streamlines converge (the leeward side), separating where it can't stand the pressure rise; Goody's wall pressure on each panel, heard by the solid angle it fills at the ear, through the acrylic (mass law), less what a panel misses below hydrodynamic coincidence |
| Airbrakes | each plate's vortex shedding and the spoiled section's unsteady lift (Curle dipoles) and its wake at the trailing edge (BPM with the plate's height as the displacement thickness): broadband, growing as the sixth power of the speed, as spoiler studies find |
| Manoeuvres | a steady pull-up at the load factor, a steady roll at the aileron, a steady sideslip at the rudder: the ailerons (with the FDM's differential), elevator and rudder deflect the sections in NeuralFoil; the roll rate balances the wing's rolling moment, the tailplane the pitching moments about the centre of gravity (the elevator finding its lift, to its stops), the fin the fuselage's yawing moment; each deflected aileron's ends shed vortices (BPM's tip noise at the lift the deflection adds) |
| Separation and stall | each section's suction side separates from the trailing edge forward as its angle grows, and BPM's separation noise follows that share instead of a NACA 0012's angle: the attached layer's hiss goes, the separated flow's rumble comes, and past the maximum lift BPM's deep stall |

```bash
pywopwop glider ask21 --speed 55 --cockpit -o thermalling.wav
pywopwop glider ask21 --speed 60 --brakes 1 --cockpit -o approach_brakes.wav
pywopwop glider ask21 --speed 64 --load-factor 3.5 -o accelerated_stall.wav
pywopwop glider ask21 --speed 60 --aileron 1 -o roll.wav
pywopwop glider-bank ask21 -o out/ask21
```

The bank has 8 airspeeds from 40 to 150 kt with the brakes fully out and full aileron in a steady roll, and the
stall at 1, 2, 3.5 and 5 g (an accelerated stall: at n g the wing stalls at the square root of n times the 1 g
speed, and its separated flow is louder for the speed), and inverted at -1 g (from 1.1 times the inverted stall's
speed, and that stall), at both seats and 4 positions outside (30 m). Layers:
`airframe`, `wind` (inside), `brakes` (what the brakes add), `ailerons` (what full aileron and its roll add), and
`mix` (airframe and wind); the accelerated stalls have the airframe only. Slips (full rudder, about 17 degrees of
sideslip) have the cockpit's loops. `pywopwop-glider.xml` cross-fades the
airspeed loops by equal power on `velocities/airspeed-kt`: the canopy's wind by airspeed alone (the 1 g stall's
loop the lowest), the wing's sound turning to the stall loops' (themselves cross-faded by airspeed) on
`orientation/alpha-deg` from where its flow begins to separate to the stall; it plays the brakes' layer with the
square root of `surface-positions/speedbrake-pos-norm` and the ailerons' with
`surface-positions/left-aileron-pos-norm`; the inverted loops come in for the upright ones as
`accelerations/pilot-g` goes from +1 to -1, and the slips' wind for the straight flight's as `orientation/side-slip-deg`
grows to the full-rudder slip's.

The computed stall speed of the ASK 21 at 540 kg is 34.7 kt, the flight manual's 35 kt. In the cockpit the canopy's
boundary layer dominates: 76 dBA at the stall, 78 at 40 kt, 91 at 150 kt (a study of the K7's cockpit found over 75
dBA). Toward the stall the wing's root separates from the trailing edge (from 4.3 degrees body angle of attack at
1 g), its hiss turning to rumble. The canopy's flow stays attached: its boundary layer thickens on the leeward line,
but the wing's circulation keeps the flow over the canopy speeding up to the wing root, so the cockpit grows quieter
toward the stall by the airspeed it loses. The brakes add a low rumble, clear outside and at speed, faint under the canopy's hiss at approach speeds:
how much of their buffet reaches the cockpit through the structure is not modelled, so their layer is separate.
The airbrakes' force coefficients are assumed values from the bluff-body literature.

In aerobatics the computed sound comes mostly from the airspeed and from the wing's flow separating. A pull at
3.5 g at 70 kt changes the airframe's sound by about 1 dB; stalled there it is 13 dB louder than the 1 g stall, 17
dB at 5 g. Full aileron adds about 1 dB outside: the roll rate keeps the ailerons' sections near their angles, and
what they add comes mostly from the vortices at their ends. The elevator and rudder span their surfaces and add
little. A slip turns the flow across the canopy: on the windward side its boundary layer is thin and the flow
slower, on the leeward side the layer gathers, two to four times thicker, and stays attached on the glass (the
leeward separation runs below the sills); in the cockpit the hiss grows by about 1 dB, brighter on one side and
duller on the other. Air leaking through the canopy's seals and vents, which a slip's pressure across the cockpit
drives, is not modelled. Inverted, the cambered wing needs a large negative angle and stalls at 47 kt (35 upright); the canopy
faces the flow instead of trailing it, its boundary layer thinner and the air over it slower, so its hiss loses
3-5 dB of its lows at 60 kt and keeps its highs: thinner, brighter. Under the canopy all of the wing's sound lies
some 40 dB below the canopy's wind: what a pilot feels and hears of
buffet through the structure is not in the model.

## Your own rotor

Copy a preset from `pywopwop/presets/` and edit it: everything is SI (m, kg) except angles (deg) and rotor speed
(rpm). A rotor needs at least `radius`, `blades`, `rpm`, `chord`, `twist`, `root_cutout`, `position` and `axis`
(thrust direction, aircraft frame: x forward, y left, z up). The elastic blade uses `hub` (articulated, hingeless,
teetering), `flap_frequency`, `flap2_frequency`, `torsion_frequency`, `elastic_axis`, `cg_axis` and the
`lock_number`; set `elastic = false` for stiff blades. The section is `airfoil = { table = "...", shape = "..." }`:
a table in `pywopwop/presets/airfoils/` (JSON: Cl, Cd and optionally Cm against angle and Mach) and a NACA
designation or a coordinate file there. `tools/import_fdm_airfoils.py` makes them from the C-81 and GenHel tables
of the FlightGear Bo 105 and UH-60 JSBSim models and the V-22 model's sections in C-81's equation form; the tables
shipped came from those. Without a table, a smooth
model from `lift_slope`, `alpha0_deg`, `cl_max`, `cd0`, `mach_dd` and `cm0` stands in. Then
`pywopwop render my_heli.toml ...`. A tiltrotor's rotors take a `pivot` (the conversion axis, about which the hub and
thrust axis given for helicopter mode turn forward), and the aircraft its airplane mode's level-flight trims,
`airplane_trim = [[true airspeed kt, angle of attack deg, thrust N per rotor], ...]` at its mass, and `airplane_rpm`
(the MV-22B's from its FlightGear flight model).

## Checks

`pywopwop validate` and `pytest`:

- BPM against the output of NASA's own code (RP-1218 tables D2 and D3): within 0.02 dB.
- F1A loading against Gutin's closed-form hover harmonics: within 0.3 dB.
- F1A against Formulation 1 (the same physics by a numerical time derivative) with unsteady loads, elastic
  heave and forward flight: within 0.1 % (hover) and 0.5 % (fly-by).
- Reception times close the retarded-time equation to round-off.
- ISO 9613-1 absorption against ISO 9613-2's table.
- Hover figure of merit of the UH-60A in the expected range; the Bo 105's second flap frequency predicted from its
  first within 2 % of the published value.
- Loops: nothing above the band limit (to numerical precision) and no step at the loop point.
- The FlightGear section's volumes, evaluated as SimGear does: the loops add up to one at any airspeed, vertical
  speed and height.
- Wind: the wall pressure within Bies's range of the dynamic pressure; the FlightGear section's loops add up to the
  altitude's amplitude around the equivalent airspeed.
- Airplane mode: the trim follows the preset's trims; the MV-22B's rotor power within 1.2 % of its flight model's
  at the same thrust and rotor speed (160-270 kt).

## Limits

The blade-vortex interaction is calibrated to HART II's airloads, the blade-wake interaction's turbulence and the
unsteadiness to plausible values, and the loops were checked against recordings for their texture (how alike
successive blade passages are, tones against noise); absolute levels are not calibrated against flight tests.
Linear acoustics (no transonic delocalisation above about Mach 0.9 at the advancing tip), a prescribed rather
than free wake, modal blades with uniform properties and no lag motion, no fuselage scattering, and no engine,
gearbox or main rotor wake on the tail rotor. One section per rotor (the UH-60's SC1095 tables with SC1094R8
coordinates for the whole blade; the MV-22B's XN12, the 0.75 R section, with a 12 % thickness for the whole blade,
whose tip is 9 %); the XV-15 and the quadcopter, whose section tables aren't public, use the smooth section model.
The tiltrotors fly with their nacelles vertical and their rotors carrying all the weight (no wing lift), or in
airplane mode from given trims; the conversion between them is not modelled. The cockpit's wind is an attached
boundary layer's through a limp wall (the cabin's 8 kg/m^2 by default): no separated flow round the windscreen's frame,
no compressibility (past about Mach 0.5), and a boundary layer drives a panel less efficiently than sound does below
hydrodynamic coincidence, so its low end is an upper bound. Exported loops are high-passed at 25 Hz (`--highpass`), as a recording chain is: a main rotor's
blade-passing fundamental (10 to 30 Hz) is felt more than heard.

## Licence

MIT (see [LICENSE](LICENSE)).
