# PyWopWop

Rotor sounds for flight simulators, computed from the physics of the rotor: the blades' aerodynamics, their
elastic response, the tip-vortex wake that makes the blade slap, and the Ffowcs Williams-Hawkings acoustics. The
output is seamless, alias-free WAV loops, with banks of them per flight condition, for the cockpit and for
outside, and a FlightGear sound configuration that blends them. Also gliders' airframe sound and turbine engines'
(compressor, combustion, turbine, starter) through starts and shutdowns.

You give it a rotor's published geometry: radius, blades, chord, twist, airfoil, rpm, and a few structural
frequencies. No CFD and no external comprehensive analysis are needed.

How it works, and the comparisons with closed forms, rotor tests, published levels and recordings test by test, are
in the report: [PyWopWop-report.pdf](PyWopWop-report.pdf).

## What it computes

| Stage | Model |
|---|---|
| Sections | Lift, drag and pitching moment tables against angle of attack and Mach number (C-81 decks, GenHel), smoothed for acoustics; real section shapes (coordinate files, NACA four- and five-digit with camber) for the blade surface |
| Trim | Aircraft weight and drag balanced by the rotor; collective and cyclic for the thrust with no flapping relative to the tip-path plane; tail rotor carries the main rotor's torque |
| Inflow | Blade-element momentum theory with Prandtl tip loss in the hover, Glauert and Drees in forward flight, an empirical bridge through the vortex ring state; a rotor may give its annuli loaded downward (a twisted proprotor's tips at flat pitch) the mirrored momentum balance, the annulus hovering upside down (`signed_hover_inflow`: the MV-22B's, whose run-up then makes its flight model's thrust) |
| Elastic blades | Rotating-beam flap modes (hinged or clamped root; a hingeless rotor cones by bending) and a torsion mode, coupled by the section CG's offset and the aerodynamics, loaded by the section's moment and the propeller moment, solved by harmonic balance |
| Blade-vortex interaction | Beddoes' prescribed tip-vortex wake released from the deflected tip, Vatistas cores growing with age, Biot-Savart velocity sampled over the chord, Wagner's indicial lift response; the effective core calibrated so HART II's measured noise map comes out at its level |
| Unsteadiness | Revolutions computed with their own turbulent gusts (frozen von Kármán field) and tip-vortex wander and strength; each blade morphs between them from one passage to the next, so no two slaps are alike |
| Blade-wake interaction | The turbulence around the older tip vortices met by the following blades (Landgrebe's hover wake, Beddoes' in forward flight), through Amiet's inflow noise: the swish and the crackle around the slap |
| Tonal noise | Farassat's Formulation 1A: thickness from the real blade surface, loading over the chord; closed-form reception times, analytic kinematics including the elastic motion |
| Wing behind the discs | A tiltrotor in airplane mode (`wing`): each blade's bound vortex, the tip and root vortices it trails (the prescribed wake's, with HART II's measured core) and its viscous wake (Lakshminarayana and Davino's deficit) sweep over the wing's leading edge; each strip answers its upwash by Sears' function and its unsteady lift radiates as a dipole, part of the blade's loading harmonics. Where the tip circle crosses the wing (the MV-22B's roots) the vortices run into its leading edge 3 ms old, and the wakes' pulses there are a fraction of a millisecond wide: harmonics that fall slowly. On the airframe's own skin only its sound counts: the pressure hugging the wing is no sound, and the skin under the wing is out of the outside air. Where the wing is bolted to the fuselage its loads also reach the cabin through the structure (statistical energy analysis: the power fed in through the structure's input mobility, stored against its loss factor, radiated by the panels between frames and stringers by their edge modes, Maidanik), which passes those harmonics as the walls don't |
| Wing under the discs | A tiltrotor in a hover (`wing`): the wing blocks part of each rotor's downwash, which meets the other's over the fuselage and rises back through the discs. The inflow over the wing falls by 20 % (Rutledge, Coffen and George's model of the XV-15), a jump in each blade's load that radiates spikes behind the aircraft, of a strength of their own each revolution; and the fountain's turbulence (from hot wires on a 1/12 scale XV-15: 25 % of the induced velocity over the disc's outer third, from the wing rearward, the disc's height over the wing its length scale, half the time) goes through Amiet's inflow noise. Gone with the download by 15 m/s |
| Broadband noise | BPM airfoil self-noise (ported from the NASA RP-1218 Fortran) and Amiet turbulent inflow noise (the air's turbulence fading with height, from the surface layer's to a quarter of it above 1000 m), per segment and azimuth, Doppler shifted: a spectral history over the revolution |
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
`--cockpit` (the preset's pilot position, heard through the cabin's walls: the preset's `[cabin]`, skin and windows
with their coincidence dips, leaks, and with `--doors-open` every door and window it lists open; without one a limp
8 kg/m^2 wall).
`--stems` also writes every rotor and component.

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

Tiltrotors: `--mode airplane` renders the airplane mode's conditions, in the thin, smooth air of a cruise at 3048 m
unless `--altitude` says otherwise (nacelles 0: 180, 220 and 260 kt, a climb and a
descent) from the preset's `airplane_trim` and `airplane_rpm`; a simulator fades the two banks into each other with
the nacelle angle. A bank's airspeeds are calibrated, as the simulator's fades are: at altitude a condition flies the
true airspeed with sea level's dynamic pressure at that speed (220 kt at 3048 m is 256 kt true), which holds the
cockpit's level within 0.2 dB of the same speed at sea level (at 220 kt true it fell 4 dB short).

`tools/slap_map.py` maps the blade slap's level over the rotor's advance ratio and its inflow through the tip-path
plane (trims over a grid of airspeed and vertical speed): a flight model that computes these for its rotors can
drive a bank's slap loops in any manoeuvre (a flare, a turn), not only in the bank's steady descents. The
FlightGear V-22 does.

`wind` renders the cockpit's wind: the windscreen's turbulent boundary layer, or with the fuselage described the
boundary layer over all its skin round the cockpit and cabin, each patch at its run from the nose (a flat plate's;
the windscreen `windscreen_m` aft of it), its wall pressure by Goody's spectral shape at Farabee and Casarella's
measured level (Bies's flight data within about 1 dB), through the walls as Levine and DeFelice predict a helicopter's
cabin noise (NASA, 1978): the windows and the skin by the mass law, each coupled poorly to the convected eddies below
its hydrodynamic coincidence (as a glider's canopy is), not through the cabin's leaks (the eddies don't pass a gap as
sound does), and built up inside as the cockpit's absorption lets it. Loops at sea level from 30 to 350 kt, at a bank's cockpit gain (`--gain-from`), and a FlightGear section
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
| Cockpit wind | the fuselage's potential flow (a panel method on its contours, with the wing's lifting line) at the body's angle of attack and sideslip; each panel of the glass and of the shell below the sills gets the boundary layer of the surface streamline that reaches it, traced back to the nose: laminar to the canopy frame and turbulent from there, thickening where the streamlines converge (the leeward side), separating where it can't stand the pressure rise or lies past the crossflow's separation line (Lee's lines for slender bodies); Goody's wall pressure on each panel, heard by the solid angle it fills at the ear, through its wall (mass law), less what a panel misses below hydrodynamic coincidence; the left and right halves apart |
| Airbrakes | each plate's vortex shedding and the spoiled section's unsteady lift (Curle dipoles) and its wake at the trailing edge (BPM with the plate's height as the displacement thickness): broadband, growing as the sixth power of the speed, as spoiler studies find |
| Manoeuvres | a steady pull-up at the load factor, a steady roll at the aileron, a steady sideslip at the rudder: the ailerons (with the FDM's differential), elevator and rudder deflect the sections in NeuralFoil; the roll rate balances the wing's rolling moment, the tailplane the pitching moments about the centre of gravity (the elevator finding its lift, to its stops), the fin the fuselage's yawing moment; each deflected aileron's ends shed vortices (BPM's tip noise at the lift the deflection adds) |
| Sideslip | set by its angle (the flight model's full rudder goes far past the fin's own balance); the wing and tail at the speed across their span, V cos(sideslip); the fuselage across the crossflow as Fink's strut, length by length; its wake on the inboard leeward wing and the stalled fin's on the leeward tailplane through Amiet's inflow noise. An estimate: no recording of a full slip has tested it |
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
`airframe` (with the wheels that don't retract), `windward` and `leeward` (inside: the flow on the cockpit's right and left halves, in straight flight two equal halves that aren't correlated), `brakes` (what the brakes add), `ailerons` (what full aileron and its roll add), and
`mix` (airframe and both halves); the accelerated stalls have the airframe only. Slips at 15, 25, 35 and 50 degrees at
each airspeed (`slip25_60kt`) have the wind's halves inside and the airframe outside; the manifest lists the angles in
`slip_angles_deg` and on each slip's files in `sideslip_deg`. `pywopwop-glider.xml` cross-fades the
airspeed loops by equal power on `velocities/airspeed-kt`: the cockpit's wind by airspeed alone (the 1 g stall's
loop the lowest), below the lowest loop each layer falling as the speed to its own exponent (`low_speed_power` in the
manifest), the wing's sound turning to the stall loops' (themselves cross-faded by airspeed) on
`orientation/alpha-deg` from where its flow begins to separate to the stall; it plays the brakes' layer with the
square root of `surface-positions/speedbrake-pos-norm` and the ailerons' with
`surface-positions/left-aileron-pos-norm`; the inverted loops come in for the upright ones as
`accelerations/pilot-g` goes from +1 to -1. The slips' loops come in over their angles on the size of
`orientation/side-slip-deg`, the wind's halves playing left and right of the ear (`cockpit_sound_position`), windward on
the side the sign says: positive is the wind from the right. Each loop's condition is only true inside its own fades, since
FlightGear has 128 sound sources for everything and a looped sound holds one while its condition is true, however silent:
at most 23 of the section's 262 loops run at once inside and 13 outside (`max_sources` in the manifest).

The computed stall speed of the ASK 21 at 540 kg is 34.7 kt, the flight manual's 35 kt. In the cockpit the canopy's
boundary layer dominates: 76 dBA at the stall, 77 at 40 kt, 89 at 150 kt (a study of the K7's cockpit found over 75
dBA). Under it lies a low roar measured from recordings, 50 to 316 Hz, growing as the cube of the speed, its source unknown. Toward the stall the wing's root separates from the trailing edge (from 4.3 degrees body angle of attack at
1 g), its hiss turning to rumble. The canopy's flow stays attached: its boundary layer thickens on the leeward line,
but the wing's circulation keeps the flow over the canopy speeding up to the wing root, so the cockpit grows quieter
toward the stall by the airspeed it loses. The brakes add a low rumble, clear outside and at speed, faint under the canopy's hiss at approach speeds:
how much of their buffet reaches the cockpit through the structure is not modelled, so their layer is separate.
The airbrakes' force coefficients are assumed values from the bluff-body literature.

In aerobatics the computed sound comes mostly from the airspeed and from the wing's flow separating. A pull at
3.5 g at 70 kt changes the airframe's sound by about 1 dB; stalled there it is 13 dB louder than the 1 g stall, 17
dB at 5 g. Full aileron adds about 1 dB outside: the roll rate keeps the ailerons' sections near their angles, and
what they add comes mostly from the vortices at their ends. The elevator and rudder span their surfaces and add
little. A slip turns the flow across the canopy, and a big one makes the glider fly sideways: a pilot hears it turn
into a flying airbrake. On the windward side the boundary layer is thin, on the leeward side it gathers, and past
the crossflow's separation line, from 14 degrees of sideslip at the rear seat and 23 at the front, the
leeward walls lose their hiss to a roar. At the flight model's full-rudder slip, 46 degrees at 99 km/h, the front seat gains
8 dB(A), the leeward half 9 dB(A) over the windward, most of it from 25 to 250 Hz; 30 m to the side the airframe
gains about 1 dB(A) and 6 dB overall, underneath 5 dB(A) and 13 dB, what it adds to the side
close to what the brakes add from 25 to 250 Hz. These levels are estimates: no recording of a full slip has tested them. Air
leaking through the canopy's seals and vents, which a slip's pressure across the cockpit drives, is not modelled. Inverted, the cambered wing needs a large negative angle and stalls at 47 kt (35 upright); the canopy
faces the flow instead of trailing it, its boundary layer thinner and the air over it slower, so its hiss loses
3-5 dB of its lows at 60 kt and keeps its highs: thinner, brighter. Under the canopy all of the wing's sound lies
some 40 dB below the canopy's wind: what a pilot feels and hears of
buffet through the structure is not in the model.

## Propeller aircraft

`pywopwop/propeller.py` renders aircraft whose discs have speeds and loads of their own: each group of discs, its motors and
its spinners separately, so a stopped motor or a failed contactor drops out of the mix. The preset `x57` (in
`presets/propellers`) is NASA's X-57 Mod III and IV, from its OpenVSP model: two wingtip cruise propellers and twelve folding
high-lift propellers in four contactor groups.

```bash
python -m pywopwop.propeller x57 -o out/x57
```

| Source | Model |
|---|---|
| Blades | the rotor model, rigid blades in the flow along the shaft, sections by NeuralFoil; fixed pitch finds its thrust, the cruise propellers' thrust is the condition's. BPM's angle runs from zero lift, for the high-lift blades' strong camber |
| Discs ahead of the wing | the wing's bound vortex tilts the flow into each disc; its unsteady lift under their vortices is the tiltrotors' wing model, untested here |
| Unsynchronised motors | each disc of a group a revolution per loop off the next, so their tones beat |
| Motors | the electrical frequency's harmonics and the slots' passing, at an assumed level: nothing measured. The inverters switch near 40 kHz, out of hearing. At that level they stand some 40 dB(A) under the propellers in the cockpit, so their layer can be raised without touching the rest |
| Spinners | the two tones NASA's array placed on the high-lift spinner, as measured |
| Cockpit | at the left seat's head through the P2006T's cabin walls, estimated, not through a fuselage's wall patches |

The bank covers each group over airspeed and rotor speed, the cruise propellers also over a thrust coefficient from
regenerating to climbing. `pywopwop-props.xml` cross-fades them by equal power, each pitched by its disc's speed, and each loop's
condition is only true inside its fades: at most 42 run at once. Its properties (`PROPERTIES` in `propeller.py`) are proposals for the flight model:
`/fdm/jsbsim/x57/cruise[n]/rpm`, `ct` and `torque-norm`, `/fdm/jsbsim/x57/hl[n]/rpm` and `torque-norm`.

Mods II to IV never flew. What could be checked: NASA's schedule and CFD for the loads, and one isolated high-lift propeller in
NASA's tunnel. What couldn't: the cruise propeller's noise, the motors, the cabin, and the discs on the wing together.

## Tyres

`pywopwop/tyre.py` makes the sound of an aircraft's tyres rolling on a paved runway: each wheel's loops by ground speed,
in the cockpit and 30 m to the side, and a FlightGear section keyed to each wheel's own rolling speed and load. It has no
command yet. From Python, the X-57's three tyres at its propeller bank's gains:

```python
import json
from pywopwop import tyre
from pywopwop.propeller import load_prop_aircraft

tyres = [dict(name="nose", gear=0, position=(2.463, 0.0, -1.172), width_m=0.1257, diameter_m=0.3607, static_compression_ft=0.244),
         dict(name="left_main", gear=1, position=(-0.437, 1.0, -1.172), width_m=0.16, diameter_m=0.4445, static_compression_ft=0.244),
         dict(name="right_main", gear=2, position=(-0.437, -1.0, -1.172), width_m=0.16, diameter_m=0.4445, static_compression_ft=0.244)]
gain = json.load(open("out/x57/manifest.json"))["gain_db"]
tyre.make_tyre_bank(load_prop_aircraft("x57"), "out/x57_tyres", tyres, gain, "p2006t")
```

| Part | Model |
|---|---|
| Rolling noise | a measured truck tyre's (Hoever 2014), scaled to the tyre's width and diameter, to the speed (the tread's vibration as its square, the air pumped in and out of the contact as its fourth power) and to the runway's texture |
| Outside | each wheel from its contact, over a hard ground |
| Cockpit | through the cabin's walls, and through the structure, which is most of it: a transfer taken from a recorded landing in the same airframe (`STRUCTURES`: an ASK 21's and a P2006T's, the X-57's airframe) |
| Loops | stationary Gaussian noise: a rolling tyre has no tone and no thump once a turn, in the recordings as in the loops |

`pywopwop-tyres.xml` cross-fades each wheel's loops by equal power over `/gear/gear[n]/rollspeed-ms`, each loop's condition
true only inside its speeds and with weight on the wheel (`wow`): at most two loops per wheel and view. A wheel fades out as
`compression-norm` (JSBSim's, in feet) falls under a quarter of its static compression.

What could be checked: outside, the one light aircraft measurement found (three towed 5.00-5s, 30 m to the side, 60 dB(A);
the model 60.5). In a cockpit, the camera's gain controls in the recordings leave only the moment of touchdown to read, so
each airframe's transfer rests on one landing, and the P2006T's touchdown speed isn't in the picture. How the cockpit's level
grows with speed is the model's alone. Not modelled: grass, concrete, wet runways, brakes, a runway's joints.
`tools/tyre_recordings.py` (the report's stage `tyres`) reproduces every number from the recordings.

At touchdown a wheel meets the runway not turning and slides until friction spins it up. The slide loses
I (V/r)²/2 whatever the load does, so a firm touchdown gives a short loud squeal and a gentle one a long quiet squeal of the
same energy. `make_touchdown_bank` writes one-shots of each wheel group's squeal by slip speed (the ground speed less the
wheel's rim speed) and sink rate, in the cockpit and outside, and its strut's thump by sink rate, in the cockpit:

```python
groups = [dict(name="main", tyre_type="6.00-6", damping=8071.0, spring=76296.0, static_load_n=5671.0, mass_kg=617.1,
               wheels=[dict(gear=1, position=(-0.437, 1.0, -1.172)), dict(gear=2, position=(-0.437, -1.0, -1.172))]),
          dict(name="nose", tyre_type="5.00-5", damping=2846.0, spring=26926.0, static_load_n=2002.0, mass_kg=320.7,
               wheels=[dict(gear=0, position=(2.463, 0.0, -1.172))])]
tyre.make_touchdown_bank(load_prop_aircraft("x57"), "out/x57_touchdown", groups, gain, "p2006t")
```

| Part | Model |
|---|---|
| Spin-up | the wheel's inertia (the tyre's weight, its radius) against the friction on the load: the strut's damping on the rate it compresses, rising as the wheel squashes its tyre, and its spring, stopping the airframe's mass at the wheel (`strut_mass`, from the aircraft's mass and pitch inertia) as the wing sheds the weight onto it over 0.3 s |
| Squeal | a share of the spin-up's loss, -48 dB, from a P2006T's landing; an ASK 21's comes within 3 dB. It follows the wheel's load, holding its level to the end of the slide as the recorded one does. Its spectrum and its line (wandering or gliding, a half to a tenth of the energy) from the same two recordings, per tyre size |
| Thump | the strut's damping force on the airframe, its level from the P2006T's touchdown, at the edge of what the recording shows |

`pywopwop-touchdown.xml` plays them once per wheel and touchdown, chosen by the sink rate and slip the flight model holds
for that wheel from the moment its weight comes on until it lifts off: `/fdm/jsbsim/gear/unit[n]/touchdown-sink-fps` and
`touchdown-slip-kt`, which the flight model has to provide. The sounds cross-fade: each wheel's files come from one seed
of its own, so on its grid they are one waveform under different envelopes and levels, starting together, and the files at
the grid's nodes around a touchdown blend by amplitude, with tent weights that sum to one, into one event (up to four
squeals and two thumps a wheel, for a few tenths of a second). Uncorrelated files blended so would lose up to 3 dB between
nodes, and blended by equal power they keep the level but sound like two tyres. Each node's amplitude follows the event's
energy, as the slip squared for a squeal (the sink rate only sets how long it lasts) and as the sink rate squared for a
thump, its file's own difference from that law taken out. Neighbours' envelopes differ (a firm touchdown's squeal is
front-loaded, a gentle one's held), which would cost a blend up to 0.9 dB between nodes; each squeal's volume puts that
back from the neighbours' correlation (the manifest's `coherence`), and the blends stay within 0.3 dB of the law
anywhere on the grid. The volumes use SimGear's `div` and `sqrt`.
Both mains touching together add 3 dB, as two tyres do. Past 105 kt of slip and 4 fps the last files play at full volume. The squeal is checked against the
P2006T's in level (0.2 dB(A) over), in how its energy falls through the slide (by 10 ms within 1.6 dB rms, 54 % in the
first half against 53 %), in how much of it is one line (44 % against 47 %, seed to seed ±6) and in its bands (within
2.5 dB from 630 Hz to 6.3 kHz). The slides are checked against the X-57's flight model in JSBSim, which slows the sink
as the strut does: trimmed at 75 kt with the controls held, 66 and 43 ms against its 65 and 39 at 2.6 and 4 fps; dropped
untrimmed at 85 kt, the wing carrying three quarters of the weight, within 7 % from 1.1 to 4.2 fps (161 against 156 ms
at 1.1). How fast the wing sheds the weight is the pilot's, and no flown landing has been timed: with the controls held
JSBSim's mains slide 357 ms at 0.95 fps, against 159 here. The thumps play at their estimate. Where one would pass full scale at a gain the propellers set, it goes
through a soft knee 6 dB under full scale, as a limiter would, and past 6 dB over full scale it is driven in at half its
growth: an X-57 main thump plays 0.5 dB under its estimate at 1 fps, 3.0 at 2 and 6.9 at 4, still louder each time, with
the clipper's distortion at 1, 6 and 11 % of its energy.

Legs with twin tyres, as the MV-22B's (8.50-10 mains and 18x5.7-8 noses, two on every leg), are one source each: a
rolling tyre's `count`, a touchdown group's `tyres`. In a touchdown the tyres share the leg's strut and slide together,
each from a seed of its own, so a leg's squeal carries twice one tyre's energy and its thump is the strut's. An airframe
with no cockpit recording of its own takes a recorded one's transfer scaled to its mass (`airframe_kg`): its structure
is taken to answer a force at the gear as the force over its mass, so the transfer falls by 20 log10 of the masses'
ratio and a thump grows as the force over the mass, squared. That is an estimate; a heavier airframe's stiffer
structure, its larger cabin and where its gear meets it aren't in it. For the MV-22B (21.5 t) the P2006T's transfer
comes 25 dB down, and its squeal, which no recording gives, is the P2006T's 6.00-6's moved down in frequency for the
larger tyres as the rolling noise's features move. At the MV-22B rotor bank's gains a main leg rolling at 200 kt comes
to 85 dB(A) 30 m to the side and 78 in the cockpit, and its thump at 16 fps to 99 dB(A), at its estimate and under full
scale. Its squeal blends stay within 0.7 dB of the law: its shortest slides, 6 to 25 ms at low slip and a firm sink,
are clicks whose neighbours differ more than the cross-fade's average correlation along each axis says.

## Turbine engines

`pywopwop engine` and `engine-bank` compute a turboshaft's sound from its thermodynamic state: starts, shutdowns
and everything between. An aircraft preset lists its engines (`[[engine]]`: the type, the inlet's and exhaust's
positions and directions, the FlightGear properties that drive it); the engine is a preset in
`pywopwop/presets/engines/` (`t700`: the UH-60A's T700-GE-700; `ae1107c`: the MV-22B's AE 1107C). The sources are the ones NASA's helicopter noise
system ROTONET uses for a helicopter's engines (ANOPP's fan, combustion and turbine modules):

| Part | Model |
|---|---|
| Cycle | along the operating line by gas generator speed Ng: the compressor's temperature rise as a power of Ng through its published delivery pressures; the flight model's fuel flow (steady, or a start's acceleration schedule below idle); the turbines passing the flow as Stodola's ellipse law says, calibrated on the published power turbine inlet temperature; the shaft power of the flight model's torque line. Cold (cranked, or spooling down): the same air without fuel |
| Compressor | the first two stages' whine, radiated forward from the inlet: Heidmann's method (NASA TM X-71763) in Hough and Weir's revision for small engines (NASA CR-198300): broadband, the blade-passing tones (cut off at low speed), the inlet distortion's tones, and once the first stage's relative tip speed is supersonic the buzz-saw's tones at the shaft's orders; the second stage heard through the first |
| Combustion | GE's (Matta's) core noise method in SAE ARP 876, through the turbines (GE's transmission loss), radiated aft from the exhaust |
| Power turbine | its last stage by GE's turbine method (FAA-RD-77-4), from the exhaust, where it works at its governed speed. Off it (the rotor brake holding it, a start bringing it up behind the gas generator) FlightGear plays that loop slowed to its speed and with its level by Smith and Bushell's turbine noise (ANOPP section 8.9: the tip Mach number cubed at the gas generator's flow), off at 1 %: GE's method holds the stage's work and grows without bound as the speed falls, and no measurement of a free turbine's noise against its speed was found |
| Jet | the exhaust's mixing noise, SAE ARP 876's acoustic power |
| Starter | the air turbine starter while it cranks: its turbine's tones and broadband (GE's method) through the engine's cowling (mass law), its exhaust ducted overboard; the bleed air reaches it choked, so its loading holds as it speeds up; a preset may leave its tones out (`tones = false`: the AE 1107C's, whose starter's blades and gearing have no source and whose recorded start shows no such tone) |
| APU | by its recorded signature (`presets/apu/`): its shaft's orders and its broadband spectrum moving with its speed, their power with the speed to the 5th |
| Transmission | the main gearbox's mesh tones at their tooth counts' frequencies, with harmonics and the planets' or shafts' sidebands, heard inside through the roof (doors open or shut alike), their amplitude with the engines' torque; where it has been measured (the MV-22B, the UH-60A), the drive's and engines' structure-borne roar, their vibration reaching the cabin's frames through the airframe, as third octaves at the pilot moving with the rotor speed and the torque, and swelling and fading with its load's ripple at the blade rate |
| Cabin | the walls between the outside and the pilot by area: skin and windows as panels with their coincidence dips (Sharp's transmission loss), a trimmed sidewall as a double wall (`trim_mass`, `trim_gap`: the skin and a trim panel with blankets between, by Fahy's mass-air-mass law, far tighter than one wall above its resonance), leaks, and what opens (doors, windows) by its share of the area (composite transmission); inside, the transmitted sound built up in a diffuse field as the interior's losses let it (1/alpha: a trimmed cabin absorbs little below 250 Hz and most above 1 kHz, so its lows build up and its highs don't, though the lowest leak back out through the walls, the room's loss too), from the cockpit's lowest mode up. With the fuselage described (`fuselage`: its sections round the cockpit and the cabin open to it, each a rounded rectangle of a given squareness; `glazing`: where its windows are, each patch passing through the windows' panel or the opaque wall's), the rotors are heard through the pressure on its walls rather than at the listener's head: below the lowest mode their area-weighted mean (one pressure chamber), above it their powers (a diffuse field, each harmonic a draw of the cabin's modal response, Schroeder's complex Gaussian: a random phase, levels scattered by 5.6 dB rms, so a few lines stand on resonances), each patch also heard directly by the listener (Sabine's direct field, A / 8 pi r^2 against A / S alpha). Near a blade most of that pressure is its hydrodynamic footprint (the radiation formula's terms falling as 1/r^2), which sweeps along the skin at the tip's speed and drives it as a boundary layer's eddies do, poorly below its hydrodynamic coincidence; only the radiating terms pass as sound |
| Synthesis | tones scattered into narrow bands by the turbulence in the inlet and the wakes, the shafts' speeds wandering, the combustion's roar swelling and fading; absorption and the ground as for the rotors |

```bash
pywopwop engine uh60a --ng 65 -o idle.wav
pywopwop engine uh60a --ng 24 --cold --starter --cockpit -o cranking.wav
pywopwop engine-bank uh60a -o out/uh60a_engines --like out/uh60a
pywopwop bank uh60a -o out/uh60a --runup --update
pywopwop engine-bank mv22b -o out/mv22b_engines --like out/mv22b
pywopwop engine-bank mv22b -o out/mv22b_engines_airplane --nacelle 0 --like out/mv22b_airplane
```

A tiltrotor's engines turn with its nacelles: an `[[engine]]` with a `pivot` gives its positions and directions with
the nacelles vertical, as the rotors do, and `--nacelle 0` renders the airplane mode's bank (its files and sounds named
`airplane_...` and `pww-airplane-...`), to be faded into the helicopter mode's by the nacelle angle as the rotor banks
are. `n2_property` names the power turbine's speed where the flight model has none of its own: the MV-22B's turns
with the proprotors (`/fdm/jsbsim/drive/nr-pct`). The AE 1107C's cycle is the published one at its rating (pressure
ratio 16.7; 15.9 kg/s and 1,208 C at the turbine inlet, the published 1,209) and below it follows the MV-22B flight
model's own line, n1 = 60 + 45 sqrt(P / 6150) % and its fuel flow, which its loops are played by. Its first stage's 32
blades put the whine at 7.08 kHz at 90.5 % n1, as recorded beside a nacelle in airplane-mode cruise; the paths from
the nacelles forward to the cockpit lose 18 dB in airplane mode that the model doesn't resolve (`cockpit_db`,
measured). Outside, on the ground 50 m to the side, one engine makes 75 dBA at 65 % n1 and 91 at the rating. An installation's `bands_db` corrects the engine's broadband by the spool's speed and frequency for every listener (the cockpit hears it through its walls), where a recording says what the sources miss: the MV-22B's is fitted to a start recorded outside, at 14-37 % Ng, and is nil at ground idle, where model and recording already agreed.

The bank has each engine lit at 14 gas generator speeds (20-104 %: below idle a start's acceleration, from idle the
operating line), cold at 13 (3-100 %: cranking, and spooling down after a shutdown or a flameout) and its starter
at 11 (3-62 %), the APU at 9 speeds (5-100 %) and the gearbox at full torque, in the cockpit with everything shut and
everything open and outside (on the ground 50 m to the right). Layers: `compressor`, `core` (combustion and jet), `turbine`
(the power turbine, at 100 % Np; played with (Np / 100)^1.5), `starter`, `apu` and `gearbox`. `--like` takes a rotor bank's gains,
so the engines' loudness against its loops is the physics'. `pywopwop-engines.xml` plays each engine's loops by its
own properties: cross-faded by equal power on `n1-pct` and pitched by n1 over the loop's own Ng (the turbine's by
`n2-pct`; the core's not, the combustor sets its spectrum), the lit loops while `running` is set and the cold ones
while it isn't, the core's with `fuel-flow-pph` over its loop's, the starter's while the starter's property is set
(the UH-60's `/fdm/jsbsim/eng[n]/starter-on`); the APU's by its speed (a property the aircraft preset names: the
UH-60 flight model has no APU yet), the gearbox's with the rotor speed and the engines' mean `torque-pct`; inside, the
shut and the all-open cockpit's loops by equal power on the share of the openings' area that is open: the
preset's `openings`, each with its property (the UH-60's sliding cabin doors, cockpit doors and gunner's windows,
`sim/model/door-positions/{bayL,bayR,passenger,crew,windowsL,windowsR}/position-norm`). Every frequency of a turbomachine scales with its shaft speed, so a
start or a shutdown is the loops following the flight model's Ng: a hung start, a hot start, an aborted start or a
slow rundown sounds as the engine behaves, not as a recording of someone else's start did.

`bank --runup` adds the rotor's run-up: on the ground with the collective fully down (`flat_pitch_deg`: the UH-60's
0.25 degrees at 0.75 R, GenHel's 9.9 at the root less the twist), at 20-100 % rotor speed. FlightGear plays them by
`rotors/main/rpm` while `gear/gear[0]/wow` is set, the flight conditions' loops otherwise.

The T700's cycle follows GE's simulation (NASA TM-105805) at 87.5-99.5 % Ng: delivery pressure within 1.5 %, power
turbine inlet temperature within 72 K (hotter; GenHel's fuel flow, which the flight model uses, runs about 5 % above
GE's). Outside, on the ground 50 m to the right, one engine makes 69 dBA at idle, 84 at 87.5 % Ng and 89 at full
power, the combustion's roar and the power turbine's hiss leading; the APU 83. Measured round a UH-60 on the ground,
the 85 dBA contour reaches past 100 ft and the crew's positions have 105 dBA or more, the worst near the APU (Jones et
al. 2018). In the cockpit with the doors shut: on the ground the APU leads at 78-80 dBA (an idling engine 62-66); at
FLY on the ground the drive leads; in the hover the drive (its mesh tones and its vibration through the airframe) 103
dBA over the rotor's 83 and the engines' 80, between NASA's measured 97 and the Army's 106. With the cabin doors open the
outside sources come in 10-12 dB louder (the APU about 90 dBA, the idling engines 85). On a start the APU's whistle and roar come first, then the
compressor's two-tone whine while the starter cranks, rising through the kilohertz to idle; after a shutdown the whine
falls with Ng.

Measured from recordings (each marked in its preset): the first two compressor stages' blade counts (20 and 22), the
second stage's level through the first, the whine's level at low speed (Heidmann's method makes it about 20 dB too
quiet while cranking, far from the compressor's design), the APU's tones, spectrum and level, and the gearbox's tones
relative to each other, and their level and the drive's bands from a calibrated spectrum at the pilot's seat at 120 kt. Estimated, as no
public source gives them: the vane counts, the stages' diameters and pressure ratios, the rotor-stator spacing, the
power turbine's last stage, the exhaust's area, the air turbine starter, the cabin's walls (`presets/engines/t700.toml`
and `presets/uh60a.toml` say which). GE's and Heidmann's methods are empirical, good to about 5 dB in level. Not
modelled: the igniters, the light-off itself (the heat release's pulse lies near 10 Hz), the gas generator turbine and
the centrifugal stage (the blade rows around them block them), the power turbine's own spectrum while it runs the rotor up (its governed speed's, slowed), the
cabin's own systems (fans, avionics), and the airframe's structure-borne paths but the gearbox's. FlightGear clips a
sound's volume at one, so the core's loops follow a fuel flow below their own but not one above it. The UH-60's flight
model cranks the gas generator to 24 % on the starter; the recorded start lights off from about 30 %.

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
`airplane_trim = [[true airspeed kt, angle of attack deg, thrust N per rotor], ...]` at its mass and `airplane_trim_m`
(their altitude), and `airplane_rpm` (the MV-22B's from its FlightGear flight model). A rotor tied to the first by a
shaft gives its blades' azimuth behind the first's as `index_deg`.

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
- Tyres: the truck tyre they're scaled from comes back within 1 dB at 1 m, and three towed 5.00-5s 30 m to the side
  within 3 dB of the 60 dB(A) measured (the model 60.5). A wheel's spin-up loses I (V/r)²/2 to 1 %, its squeal carries
  its share of that, and the P2006T's recorded touchdown, 0.101 s at 58 kt, comes back from 1.28 fps on the X-57's struts.
  The X-57's slides within 10 % of JSBSim's at 2.6 and 4 fps (`tools/jsbsim_touchdown.py`, on the X-57's flight
  model, trimmed with the controls held, or dropped untrimmed at 85 kt with `--drop`), and within 7 % in the drop from
  1.1 to 4.2 fps.
- Wind: the wall pressure within Bies's range of the dynamic pressure; the FlightGear section's loops add up to the
  altitude's amplitude around the equivalent airspeed.
- Airplane mode: the trim follows the preset's trims; the MV-22B's rotor power within 1.2 % of its flight model's
  at the same thrust and rotor speed (160-270 kt).
- Engines: the T700's cycle against GE's nonlinear simulation (NASA TM-105805) at 87.5-99.5 % Ng: compressor
  delivery pressure within 2 %, power turbine inlet temperature within 80 K. Heidmann's levels in decibels and
  ANOPP's power form of the same method agree within 0.1 dB at 1 m.
- Run-up: at flat pitch the rotor makes next to no thrust and its power grows as the cube of its speed; on the
  ground the run-up loops add up to one at any rotor speed, and off it the flight conditions' do.
- Against UH-60 recordings made in the cabin (videos; their camera's automatic gain cancels in the ratios used): the
  T700's whine settles at idle on two tones at exactly 10:11, 10.06 and 11.06 kHz, which 20 and 22 blades give at
  67.5 % Ng, and sits at 12.9 kHz at FLY on the ground and 14.04 kHz in the hover (87 and 94 % Ng). The gearbox's
  planetary mesh stands at 993 Hz (980 at 100 % rotor speed), its line growing about 22 dB over the floor beside it
  from flat pitch to the hover, more than its amplitude with the torque gives. No recording shows the cabin's doors'
  effect (an earlier check here read one narrow window as the whine drifted out of it): that stays a prediction.
- Against ASK 21 cockpits recorded in aerobatics (two videos; their cameras' gain is unknown, so only spectral shapes
  and growth with airspeed): in the first, its airspeed read off the ASI's needle frame by frame, the canopy's hiss
  keeps the model's spectrum within 1.5 dB from 315 Hz to 6.3 kHz at 50 to 115 kt (a 30 dB cap on the canopy's
  transmission loss for leaks had left it flat from 2 kHz up, 6 to 8 dB too bright), and from 400 Hz to 8 kHz its
  power grows as the airspeed to the power 2.2, the model's 1.9, though from 3 to 8 kHz faster than the model's. The second, without the first's whistle, keeps the model's
  shape within 2 dB from 400 Hz to 4 kHz. Above 5 kHz both fall faster than the model, 5 to 10 dB at 8 kHz and 10 to
  25 dB above 10 kHz: how much of that their cameras' microphones and mounts make can't be told from the cockpit's. The
  first glider's whistle, at about 25 Hz per knot
  from 85 kt, is a leak of its own and not in the model.
- Against a roll in an ASK 21's front cockpit (a video from a camera on the pilot's head; airspeed, load factor and
  sideslip read off the ASI, the g-meter and the yaw string): the roll is barrelled, from a pull-up at +1.5 to +3.5 g
  through 0 g at 90 degrees of bank and +1 g inverted to a pull-out at +4.5 g, at 67 to 93 kt. The canopy's hiss from
  1.6 to 6.3 kHz moves the way the model's does at the same airspeed and load factor, but twice as far, 8 dB against
  4: in the pull-up it grows by 2 to 3 dB per g, the model's by 0.2; with the yaw string far to one side it falls 2 to
  4 dB below the model, far to the other 2 to 4 dB above (the model's slip adds 0.7 dB there), and in both the ASI
  misreads by 10 to 20 kt. The model's hiss hardly depends on where the ear is (0.3 dB over 10 cm). Leaks through the
  canopy's seals and vents, driven by the lift and the slip, would grow so; so might the camera's microphone, and one
  recording can't tell them apart. Nothing in it starts or stops with the ailerons, as in the model.
- Against one X-57 high-lift propeller in NASA's Low Speed Aeroacoustic Wind Tunnel (4800 rpm, 75 KEAS, an array 3.54 m away,
  read off the published figure; nothing in the noise model fitted to it): the blade passing tone, 76.4 dB in the propeller's
  plane, lies between the test's two batches of blades, 79.4 and 71.7 dB: 3.4 dB under the left-handed ones, 4.6 dB at most from 85
  to 115 degrees, 5.4 dB rms from 57 to 120. The broadband stands 2 to 14 dB under from 2 to 20 kHz, and 16 to 23 dB under at
  1 kHz and below, where NASA's array finds the blade roots and the test stand's strut. The harmonics above the first are far too
  low: a rigid disc in steady flow makes almost none. At one pitch fitted to NASA's final tables (20.0 degrees; 27.35 as drawn, the
  hinge explaining 2.5 to 4.5) its power lands 10 % under to 4 % over them and its thrust 24 % under to 15 % over, the largest
  errors where it is small; the cruise propeller's thrust 2 to 15 % over NASA's CFD up to 22 degrees of pitch, 29 % at worst at 26.
- Against an MV-22B recorded in airplane-mode cruise (a video: behind the pilots facing the instrument panel, and at a
  cabin window beside a nacelle; the camera's gain and response unknown, so the checks are ratios at one frequency:
  tones over the broadband beside them, and how the sound pulses with the blades): the proprotors pass their blades at
  16.61 Hz (332 rpm, 83.7 %, the preset's airplane mode). At the window the first compressor stage's whine sits at
  7.08 kHz and falls to 6.72 kHz as the power comes off (90.5 to 85.9 % n1 in the flight model), chopped by the
  proprotor's blades (its envelope 30 dB over its floor at the blade-passing frequency). In the cockpit a tone at 8.17
  kHz holds while the whine moves (39 times the power turbine's shaft rate: a gear's mesh), and one at 2.03 kHz;
  neither the chopping nor the drive's tones are in the model. Heard at the listener's head through limp walls with a
  30 dB leak, the model's cockpit was the proprotors' first harmonics with little else. Heard through the pressure on
  the fuselage's skin (its blades' hydrodynamic footprint coupled as a convected field), with the boundary layer over
  the whole skin and the walls a trimmed sidewall (a double wall) round the windscreen where the 3-D model has it,
  what the recording has besides the proprotors and the wind is a steady roar from 40 Hz to 12 kHz that hardly pulses
  with the blades and is strongest by the wing's root (the window's camera): the drive's and engines' vibration
  through the wing and the cabin's frames. Measured from it (its spectrum the recording's floor between the lines, its
  level so that the harmonics up to the 10th stand out of it as there: medians 13 and 15 dB, the room's modes
  scattering the recording's by 8 dB from one to the next), with the two gear meshes at their prominence, the model
  makes 99 dBA at the camera's place, and the whine, the 2.03 kHz and the 8.17 kHz meshes stand 0.8, 5.6 and 5.4 dB
  out of the broadband (the recording's 0.3, 5.4 and 4.7). From the 6th to the 30th harmonic (100 to 500 Hz) the
  recording's fall 0.75 dB a harmonic, standing 2 to 15 dB out of the broadband, where the proprotors' fall 2.4 dB a
  harmonic and fade into it from the 14th: even 5 cm from the tip path their pressure falls 36 dB from the 6th to the
  30th, following the blade's 2.8 ms passage (finer patches, the radiation formula's resolution, the section's shape
  and the coupling move that by a few dB), and the recording's lines are all at multiples of the blade-passing
  frequency. The wing behind the discs makes them: each tip vortex (13 m^2/s) runs into its leading edge just outboard
  of the root 2.7 ms after it is shed, and the blades' viscous wakes sweep it there in pulses a fraction of a
  millisecond wide. Through the walls as sound its highest harmonics still lose 25 dB more from the 10th to the 30th
  than the recording's, which lose about what the pressure outside does: the wing is bolted to the cabin's roof frames
  where the vortices strike it, and through the structure (statistical energy analysis) its loads' harmonics fall 0.9
  dB a harmonic, as the recording's do. Heard at the cruise's height and speed (10,000 ft; level trims hold by dynamic
  pressure, so they are looked up at the speed with the same one at the trims' altitude; in sea-level air every rotor
  pressure was 2.6 dB too strong), with the air's turbulence there a quarter of the surface layer's (near the ground's
  0.8 m/s made an inflow rumble as strong as the harmonics under 250 Hz; now the 2nd to the 5th stand 19 to 23 dB out
  of it, the recording's 12 to 25), the two rotors' blades 26 degrees out of step (measured: below the cockpit's
  lowest mode their pressures add coherently, and in step the 2nd harmonic stands 12 dB over the recording's; only
  this index fits it and the 1st together), the structure's input mobility measured from the recording (5.6e-4 s/kg, 9
  dB under the cabin skin's: a wing box's thick skins and spars), what the walls let out counted as the room's loss
  (at 66 Hz more than the trim absorbs), the skin under the wing out of the outside air and only the wing's sound on
  the skin (the pressure hugging it is no sound), the walls' mass from the lines' stand out of the broadband from 50
  to 200 Hz (7 and 4 kg/m^2), and each harmonic a draw of the cabin's modal response (a complex Gaussian: with equal
  levels neighbouring lines beat, and the 100 to 200 Hz band pulsed 14 dB a blade pass where the recording's, one line
  standing on a resonance, pulses 5), the blade lines hold 74 % of the power from 100 to 200 Hz (the recording's 69),
  25 % from 200 to 300 Hz (20), and that band's 100 to 200 Hz pulsing is 5.6 dB, its 90 to 600 Hz periodic part 10.5
  dB under the band's rms (10.2). The harmonics from the 14th to the 30th stand up to 11 dB out of the broadband (mean
  4.5; the recording's 2 to 12, mean 5.6), from the 6th to the 13th 16 dB (its 13). Which harmonics stand on
  resonances isn't known: of 64 draws the recording's statistics lie within the spread (the band under 250 Hz stands
  6.5 dB over the rest in the median, the recording's 5.7), and the preset's is the one showing the rotors least, 6.8
  dB. Loudness-matched (ITU-R BS.1770) the model is within 2.4 dB of the recording in every band from 50 Hz to 12 kHz
  (9 dB under it from 25 to 50 Hz, where the camera's response is unknown), the noise between the lines within 2 dB
  from 39 to 500 Hz but for the third octave at 200 Hz (3.7 dB under). The recording's broadband from 500 Hz to 12 kHz
  also swells and fades with the blades, by 0.8 to 1.3 dB in every band and at the same point of each pass, as deep in
  the passes with the weakest third of the low pulses as in those with the strongest (2 dB apart; the estimate's noise
  0.2 dB): not the camera's gain, which would duck deeper under the stronger pulses. With the drive's load rippling 6
  % at the blade rate (the proprotors' torque as each blade passes the wing), measured from it, the model's swells and
  fades by 0.6 to 1.4 dB from 500 Hz up over a 10 s stretch (the recording's by 0.5 to 1.9). The wing's own flow
  through the discs, as a 2-D source and bound vortex at the 3-D model's 0.2 to 0.6 m from them, swings the blades'
  angle of attack 5 degrees a revolution and lifts their harmonics from the 6th to the 16th 13 dB further out of the
  broadband than the recording's, without their slope changing: not modelled, the real clearance and the wing's
  three-dimensional relief unknown (so the torque's ripple is measured, not computed).
- Against an XV-15 hovering over microphones on ground boards (Rutledge, Coffen and George 1991) and a 1/12 scale
  model with fences on its wings (George 1996): at 57 rotor radii the wave ahead of the aircraft is 2 Pa either way with
  no spikes, as measured, and behind it the fountain's spikes reach 10 Pa 46 degrees below the disc (measured about the
  same, three times the shallower angles') and 6 to 10 Pa at 13 and 23 degrees (5). The fountain adds 3.8 dBA behind
  and nothing ahead; the fences, which removed it, took 4.1 dBA off behind.
- Against a published table of the MV-22's noise outside (Japan's Ministry of Defense, 2012). Directly below in airplane
  mode at 220 kt: maximum level 88 dBA at 500 ft, 81 at 1000 ft and 78 at 1500 ft; the model at sea level, nothing
  fitted to it, makes 90.1 dBA at 100 m, which spreading alone takes to 86.4, 80.4 and 76.9 (77 at 100 m before the
  wing's source). In the hover the table gives 105.3 dBA at 50 m, 100.4 at 100 m, 95.5 at 200 m and 84.5 at 500 m, not
  saying the hover's height or the listener's direction. Behind the aircraft, 30 degrees below the disc, over hard
  ground and taking the maximum at fast response, the rotors alone make 104.5, 98.2, 92.3 and 84.1: within 3.2 dB. To
  the side, 2.5 dB less. A low hover heard along the ground (the hub 30 m up) falls away faster, 103.4, 95.9, 85.2 and
  73.9, its listeners sinking towards the disc's plane, which the table's 4.9 dB a doubling doesn't fit. Its engine
  test (100.4 dBA at 50 m) can't be compared: the MV-22B has no ground run here.

- Against NASA's XV-15 rotor on its outdoor stand (CR-177436), with nothing fitted: the overall level from 200 to
  10,500 lb of thrust 2 ft from the tip path, the dip where the twisted blade's tip unloads included, and at 76 m.
- Against the Bo 105's measured torsion moment (TM-103867) and HART II's tip twist, and HART II's noise map and airloads.
- Against the FAA's calibrated UH-60A flights (FAA-EE-81-16), nothing fitted: under the path, eight level flights and
  four 6 degree approaches, +1.8 dB on average and 3.0 dB rms, the approaches 4 dB high. Two real misses: from 100 to 152 kt
  the test rises 6 dB and the model doesn't, and 150 to 284 m beside the path the model is 5 to 7 dB too quiet.
- Against a UH-60A's cockpit: calibrated on the octave bands at the pilot's seat at 120 kt, the model follows NASA's levels
  with speed (TM-84664) within 1.2 dB, 5.4 dB over them.
- Against NASA's flyovers of a Prue-2 glider (CR-2377): the ASK 21 model, its wheels left out, keeps the spectrum's shape
  within 1.9 dB rms at each speed, 3 to 6 dB low.
- A fixed wheel by Fink's method (ANOPP): on the same test's Cessna 150, with bare wheels, it stays 2 to 7 dB under the whole
  aircraft, as it should. On the ASK 21 the wheels add 8 to 11 dB outside from 400 Hz to 1 kHz, an estimate no measurement
  tests; in the cockpit, through the air, nothing to speak of.
- The cockpit's low roar, measured on one recording: on another, a phone in another glider, it takes the error from 126 to
  316 Hz from 9-12 dB to 4-6 dB rms. A floor under 130 Hz that stays the same from the stall to 115 kt is left out as not
  the flow's.
- Against NASA's YF-102 (TM-73748, TM-88971): GE's combustion noise keeps the shape within 3.5 dB rms and the directivity
  within 1.9. Flight tests put it 10 to 21 dB over modern turbofans; 10 dB off a T700's would take under 2 dB off the engine.
- Against wall pressure measured in flight on the Tu-144LL (NASA CR-2005-213938), Mach 0.74: the spectrum's shape within
  0.7 dB rms from 100 Hz to 10 kHz, its level 2.4 dB under.
  `report/mkresults.py` runs all of these; the report has the numbers.

## Limits

The blade-vortex interaction's effective core is calibrated to HART II's noise, the blade-wake interaction's
turbulence and the unsteadiness set to plausible values, and the loops were checked against recordings for their
texture (how alike successive blade passages are, tones against noise). A rotor's absolute level is checked against
NASA's stand, HART II and the FAA's UH-60A flyovers (the report); a cockpit's against calibrated UH-60A surveys that
differ from each other by 8 dB, and the model is 8 to 10 dB high there below 100 Hz.
Linear acoustics (no transonic delocalisation above about Mach 0.9 at the advancing tip), a prescribed rather
than free wake, modal blades with uniform properties and no lag motion, no fuselage scattering, and no gearbox
or main rotor wake on the tail rotor. One section per rotor (the UH-60's SC1095 tables with SC1094R8
coordinates for the whole blade; the MV-22B's XN12, the 0.75 R section, with a 12 % thickness for the whole blade,
whose tip is 9 %, and the XV-15's 64-(1.5)12 by the same table); the quadcopter uses the smooth section model, whose
abrupt stall limits a rotor's thrust early. A trim that can't reach its thrust stops with an error.
The tiltrotors fly with their nacelles vertical and their rotors carrying all the weight (no wing lift), or in
airplane mode from given trims; the conversion between them is not modelled. Their wing is a source with the
nacelles below 30 degrees only, by strip theory, the blades' vortices passing through it uncut; in a hover its
fountain is the XV-15's inflow deficit scaled by the chord (a model's probes found the load falling, jumping and
falling again over the wing, which isn't modelled), and its switching from rotor to rotor only each revolution's
strength. A propeller aircraft's blades are rigid and each disc takes the wing's upwash as one angle, with no slipstream from its neighbours,
no fuselage's wall patches in its cockpit, and its motors' level assumed. A tyre's level in a cockpit rests on one recorded
landing per airframe and grows with speed as the model says; tyres roll on asphalt only. A touchdown's thump rests on
one event at the edge of its recording, and the flight model has to hold each wheel's touchdown for the one-shots. A glider's slip is an estimate throughout: Fink's strut stands in for a fuselage across the flow, a far wake's
laws for the near wakes on the leeward wing and tail, and the shell's mass is assumed; the slip's rolling moment, held by
the ailerons, isn't modelled, and neither is the wing's lift falling as the glider banks. A rotorcraft's cockpit
wind is an attached boundary layer's: no separated flow round the windscreen's frame, no leaks' own flow noise, no
compressibility (past about Mach 0.5). The cabin's interior absorption is a trimmed cabin's, estimated, for every
aircraft that doesn't give its own. Exported loops are high-passed at 25 Hz (`--highpass`), as a recording chain is: a main rotor's
blade-passing fundamental (10 to 30 Hz) is felt more than heard.

## Rebuilding the report

The [`report`](report) folder holds what produced [the report](PyWopWop-report.pdf): `mkresults.py` runs the test
programme into `report/results/` (some stages read recordings kept out of the repository and the FlightGear aircrafts'
sounds; their results are kept, so the report rebuilds without them), `refdata.py` and `report/data/` hold the
measurements compared with and their sources, and `build.sh` makes every table, number and figure from the results and
builds the PDF with [tectonic](https://tectonic-typesetting.github.io).

```bash
python report/mkresults.py
report/build.sh
```

## Licence

MIT (see [LICENSE](LICENSE)).
