# PyWopWop

Rotor sounds for flight simulators, computed from the physics of the rotor: the blades' aerodynamics, their
elastic response, the tip-vortex wake that makes the blade slap, and the Ffowcs Williams-Hawkings acoustics. The
output is seamless, alias-free WAV loops, with banks of them per flight condition, for the cockpit and for
outside, and a FlightGear sound configuration that blends them. Also gliders' airframe sound and turbine engines'
(compressor, combustion, turbine, starter) through starts and shutdowns.

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

## Turbine engines

`pywopwop engine` and `engine-bank` compute a turboshaft's sound from its thermodynamic state: starts, shutdowns
and everything between. An aircraft preset lists its engines (`[[engine]]`: the type, the inlet's and exhaust's
positions and directions, the FlightGear properties that drive it); the engine is a preset in
`pywopwop/presets/engines/` (`t700`: the UH-60A's T700-GE-700). The sources are the ones NASA's helicopter noise
system ROTONET uses for a helicopter's engines (ANOPP's fan, combustion and turbine modules):

| Part | Model |
|---|---|
| Cycle | along the operating line by gas generator speed Ng: the compressor's temperature rise as a power of Ng through its published delivery pressures; the flight model's fuel flow (steady, or a start's acceleration schedule below idle); the turbines passing the flow as Stodola's ellipse law says, calibrated on the published power turbine inlet temperature; the shaft power of the flight model's torque line. Cold (cranked, or spooling down): the same air without fuel |
| Compressor | the first two stages' whine, radiated forward from the inlet: Heidmann's method (NASA TM X-71763) in Hough and Weir's revision for small engines (NASA CR-198300): broadband, the blade-passing tones (cut off at low speed), the inlet distortion's tones, and once the first stage's relative tip speed is supersonic the buzz-saw's tones at the shaft's orders; the second stage heard through the first |
| Combustion | GE's (Matta's) core noise method in SAE ARP 876, through the turbines (GE's transmission loss), radiated aft from the exhaust |
| Power turbine | its last stage by GE's turbine method (FAA-RD-77-4), from the exhaust, where it works at its governed speed |
| Jet | the exhaust's mixing noise, SAE ARP 876's acoustic power |
| Starter | the air turbine starter while it cranks: its turbine's tones and broadband (GE's method) through the engine's cowling (mass law), its exhaust ducted overboard; the bleed air reaches it choked, so its loading holds as it speeds up |
| APU | by its recorded signature (`presets/apu/`): its shaft's orders and its broadband spectrum moving with its speed, their power with the speed to the 5th |
| Transmission | the main gearbox's mesh tones at their tooth counts' frequencies, with harmonics and the planets' or shafts' sidebands, heard inside through the roof (doors open or shut alike), their amplitude with the engines' torque |
| Cabin | the walls between the outside and the pilot by area: skin and windows as panels with their coincidence dips (Sharp's transmission loss), leaks, and what opens (doors, windows) by its share of the area (composite transmission) |
| Synthesis | tones scattered into narrow bands by the turbulence in the inlet and the wakes, the shafts' speeds wandering, the combustion's roar swelling and fading; absorption and the ground as for the rotors |

```bash
pywopwop engine uh60a --ng 65 -o idle.wav
pywopwop engine uh60a --ng 24 --cold --starter --cockpit -o cranking.wav
pywopwop engine-bank uh60a -o out/uh60a_engines --like out/uh60a
pywopwop bank uh60a -o out/uh60a --runup --update
```

The bank has each engine lit at 14 gas generator speeds (20-104 %: below idle a start's acceleration, from idle the
operating line), cold at 13 (3-100 %: cranking, and spooling down after a shutdown or a flameout) and its starter
at 11 (3-62 %), the APU at 9 speeds (5-100 %) and the gearbox at full torque, in the cockpit with everything shut and
everything open and outside (on the ground 50 m to the right). Layers: `compressor`, `core` (combustion and jet), `turbine`
(the power turbine, at 100 % Np), `starter`, `apu` and `gearbox`. `--like` takes a rotor bank's gains,
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
FLY on the ground the gearbox 87, the engines 72 and the rotor 70; in the hover the gearbox 102 dBA over the rotor's
85 and the engines' 74, about the 100-105 dBA reported in such cabins. With the cabin doors open the outside sources come in
10-12 dB louder (the APU about 90 dBA, the idling engines 85). On a start the APU's whistle and roar come first, then the
compressor's two-tone whine while the starter cranks, rising through the kilohertz to idle; after a shutdown the whine
falls with Ng.

Measured from recordings (each marked in its preset): the first two compressor stages' blade counts (20 and 22), the
second stage's level through the first, the whine's level at low speed (Heidmann's method makes it about 20 dB too
quiet while cranking, far from the compressor's design), the APU's tones, spectrum and level, and the gearbox's tones
relative to each other (their level from NASA's review: about 103 dB in a cabin of this class). Estimated, as no
public source gives them: the vane counts, the stages' diameters and pressure ratios, the rotor-stator spacing, the
power turbine's last stage, the exhaust's area, the air turbine starter, the cabin's walls (`presets/engines/t700.toml`
and `presets/uh60a.toml` say which). GE's and Heidmann's methods are empirical, good to about 5 dB in level. Not
modelled: the igniters, the light-off itself (the heat release's pulse lies near 10 Hz), the gas generator turbine and
the centrifugal stage (the blade rows around them block them), the power turbine while it runs the rotor up, the
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
- Engines: the T700's cycle against GE's nonlinear simulation (NASA TM-105805) at 87.5-99.5 % Ng: compressor
  delivery pressure within 2 %, power turbine inlet temperature within 80 K. Heidmann's levels in decibels and
  ANOPP's power form of the same method agree within 0.1 dB at 1 m.
- Run-up: at flat pitch the rotor makes next to no thrust and its power grows as the cube of its speed; on the
  ground the run-up loops add up to one at any rotor speed, and off it the flight conditions' do.
- Against UH-60 recordings made in the cabin (videos; their camera's automatic gain cancels in the ratios used): the
  T700's whine settles at idle on two tones at exactly 10:11, 10.06 and 11.06 kHz, which 20 and 22 blades give at
  67.5 % Ng, and sits at 13.0 kHz at FLY on the ground and 14.05 kHz in the hover (87 and 94 % Ng). Closing the cabin's
  doors takes 18.3 dB off the whine at 10 kHz while the cabin's own sounds stay put; the model's walls take 18.8 dB.
  The gearbox's tones sit at the planetary mesh's 980 Hz and its harmonics and grow by about 12 dB from flat pitch to
  the hover, as their amplitude with the torque gives.

## Limits

The blade-vortex interaction is calibrated to HART II's airloads, the blade-wake interaction's turbulence and the
unsteadiness to plausible values, and the loops were checked against recordings for their texture (how alike
successive blade passages are, tones against noise); absolute levels are not calibrated against flight tests.
Linear acoustics (no transonic delocalisation above about Mach 0.9 at the advancing tip), a prescribed rather
than free wake, modal blades with uniform properties and no lag motion, no fuselage scattering, and no gearbox
or main rotor wake on the tail rotor. One section per rotor (the UH-60's SC1095 tables with SC1094R8
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
