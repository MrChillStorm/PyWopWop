"""The measurements the report compares with, transcribed or digitised, each with its source. The tables digitised from
figures are in report/data/*.json.

Rotors
  XV15_HOVER  NASA TM-86833 (Felker, Betzina and Signor, 1985), fig. 14(c): the full-scale XV-15 rotor, metal blades,
              isolated, out of ground effect, tip Mach number 0.69: the report's curve fit through the wind-corrected
              data; sigma 0.0891 (traced from the scan, about +-0.0001 in CP/sigma).
  BO105_HOVER NASA TM-85892 (Peterson and Warmbrodt, 1984), appendix B: the full-scale Bo 105 rotor at 425 rpm on NASA
              Ames' Rotor Test Apparatus, shaft 10 deg forward, flapping nulled, with the tunnel's air speed at each point
              (appendix A: the rotor drove air through the test section, its doors open). sigma 0.07.
  UH60_*      USAAEFA 83-24 app. E fig. 3 (the hover fit) and USAAEFA 77-17 app. E figs. 15-26 (level flight: faired C_P
              against advance ratio and C_T): both engines' output power, so main and tail rotor, transmission and
              accessories, on the main rotor's disc.

HART II (the 40 % Mach scaled Bo 105 rotor in the DNW's open jet, 2001)
  Lim et al., 29th European Rotorcraft Forum, 2003, paper 18, table 1 and fig. 2, fig. 7, table 3. report/data/:
  hart2_bl_loads.json  fig. 2, BL5.3, the measured M^2 c_n at r/R = 0.87 (the curve's centre line, every 2 deg);
  hart2_bl_noise.json  fig. 7, BL5.3, measured, the 6-40 BPF level (dB) on the microphones' plane, 2.215 m under the
                       hub, read off the map's colours at the 0.45 m microphone spacing and every 0.5 m along the tunnel
                       (each colour a 2 dB band; 104 and 106 share one). The map's "-x" axis runs downstream: its top is
                       upstream, by the paper's own words (the minimum noise case's advancing lobe "moved forward", towards
                       the top, the minimum vibration case's "moved back", and the baseline "quiet in the front").
  Blade torsion and flap frequencies per rev from the same paper (the prediction teams' 1.11 and 3.65). The elastic and
  mass axes and the Lock number, not given there, are the quarter chord and the Bo 105's.

XV-15 rotor on NASA's outdoor stand
  report/data/xv15_stand_noise.json: NASA CR-177436 (Bartie, Alexander, McVeigh et al., 1986) figs. 9.1 and 9.2: overall level
  against thrust, metal blades, tip Mach number 0.69, traced from the scan: near field 2 ft outboard of the tip path and 8 ft under the
  disc (a tiltrotor's fuselage side in airplane mode), far field 250 ft out, 15 deg below the disc plane.

Outside levels
  MV22_*      Japan's Ministry of Defense, MV-22 pamphlet, 2012, pp. 12-13: maximum levels, dBA (their hover's height and
              the listener's direction aren't stated).
  XV15_*      Rutledge, Coffen and George, 1991 (NTRS 19930045417), figs. 12 and 15: hover over ground boards at 57.2
              rotor radii: pressure swings read off the time histories. George, 1996, NASA CR-203040: fences on a 1/12
              scale model's wings took 4.1 dBA off behind the aircraft.

Engines
  T700_GE     GE's nonlinear simulation, NASA TM-105805, table 1: Ng %, compressor delivery psi, power turbine inlet R.
  AE1107C     published design point: pressure ratio 16.7, 15.9 kg/s, turbine inlet 1209 C.
"""

HART2 = dict(radius=2.0, blades=4, rpm=1041.0, chord=0.121, root_cutout=0.22, twist=-8.0, precone_deg=2.5, flap_frequency=1.11,
             torsion_frequency=3.65, thrust_n=3250.0, speed_ms=33.0, alpha_deg=5.3, mic_below_hub_m=2.215)

XV15_HOVER = dict(sigma=0.0891, mtip=0.69,
                  ct_s=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
                  cp_s=[0.00256, 0.00211, 0.00218, 0.00262, 0.00322, 0.00385, 0.00464, 0.00549, 0.00644, 0.00754, 0.00865,
                        0.00982, 0.01117, 0.01269, 0.01419, 0.01607, 0.01824, 0.02039, 0.02364])

# NASA TM-103867 (Peterson and Johnson, 1991) fig. 5: the steady torsion moment at r/R = 0.34 (in lb) against CT/sigma
BO105_TORSION_034 = [(0.010, -320.0), (0.025, -425.0), (0.035, -450.0), (0.060, -480.0), (0.065, -490.0), (0.075, -470.0), (0.10, -500.0)]

# (theta at 0.7 R deg, CT/sigma, CP/sigma, tunnel kt)
BO105_HOVER = dict(sigma=0.07, rpm=425.0, rho=1.2, shaft_deg=10.0,
                   points=[(3.3, -0.000044, 0.001087, 3.9), (6.7, 0.019913, 0.001842, 4.3), (8.9, 0.041104, 0.003093, 0.0),
                           (11.2, 0.061961, 0.004791, 6.3), (13.0, 0.078788, 0.006412, 0.0), (14.9, 0.104688, 0.009573, 11.0),
                           (16.5, 0.119362, 0.012141, 6.7)])


# NASA TM-110356 (Peterson, 1995) fig. 21(d): the full-scale Bo 105 rotor in the 40 x 80 ft tunnel, 425 rpm, shaft 10 deg forward, flapping
# minimised, CT/sigma 0.07: (advance ratio, CP/sigma), read from the symbols
BO105_FORWARD = dict(sigma=0.07, ct_s=0.07, shaft_deg=10.0, rho=1.2, rpm=425.0,
                     points=[(0.0714, 0.00410), (0.1417, 0.00416), (0.1767, 0.00443), (0.2120, 0.00484), (0.2823, 0.00589), (0.2823, 0.00599),
                             (0.3289, 0.00669)])


def uh60_hover_cp(ct):
    """USAAEFA 83-24's fit for the first-year aircraft, tethered at 100 ft"""
    if ct < 0.007:
        return 8.06470e-5 + 0.99309 * ct ** 1.5
    return 2.55577e-4 + 0.395715 * ct ** 1.5 + 510.0 * ct ** 3


# C_P x 1e5 against advance ratio, by C_T x 1e4
UH60_LEVEL = {0.12: {65: 33.78, 80: 45.50}, 0.14: {65: 32.89, 80: 42.79}, 0.16: {65: 32.32, 80: 40.91},
              0.18: {65: 32.05, 80: 39.92}, 0.20: {65: 32.09, 80: 39.68}, 0.22: {65: 33.14, 80: 40.44},
              0.24: {65: 34.70, 80: 41.94}, 0.26: {65: 38.01, 80: 45.47}, 0.28: {65: 41.86, 80: 49.54},
              0.30: {65: 46.87, 80: 55.17}, 0.32: {65: 53.60, 80: 62.16}}

# NASA/CR-2005-213938 (Rackl and Weston, 2005) fig. 5: the Tu-144LL in flight at Mach 0.74, 16,400 ft: wall pressure in a window blank in the
# middle of the fuselage, about 30 m from the nose of a 65.7 m aircraft: Hz, dB re 20 uPa in 1 Hz, the sensors' middle read from the figure
TU144 = dict(mach=0.74, alt_m=16400 * 0.3048, x_m=30.0, f=[100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 8000, 10000],
             level=[98.3, 98.0, 98.6, 98.8, 98.3, 97.8, 96.8, 95.8, 94.2, 93.0, 92.0, 90.6, 90.0])

# calibrated cockpit measurements of the UH-60A. Mace et al. 1981, in USAARL report 2000-19 table 1 (Gordon and Ahroon, 2000): 120 kt, doors shut,
# octave bands dB and dBA; the calibration of the gearbox and the drive's bands
UH60_OCTAVES = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
UH60_PILOT = dict(bands=[108, 100, 102, 107, 96, 91, 88, 85, 85], dba=101.0)
UH60_COPILOT = dict(bands=[112, 101, 101, 100, 94, 89, 88, 84, 83], dba=97.0)
# NASA TM-84664 (Clevenson, Leatherwood and Hollenbaugh, 1983) table V, aircraft H-5, microphones by the pilots' heads, pistonphone calibrated:
# independent of that calibration. kt: dBA (OGE hover at 0; 145 kt the cruise)
UH60_NASA_DBA = {0: 97.0, 60: 94.0, 80: 93.0, 100: 94.0, 120: 93.0, 145: 99.0, 165: 101.0}

# FAA-EE-81-16 (Newman, Rickley and Ford, 1981) appendix A, the UH-60A over the centreline's centre microphone, 1.2 m up, each an average of several
# runs: (kt indicated, ft/min, height m, maximum dBA, OASPL dB). Speeds and rates from appendix D's cockpit photographs: the approaches' 6 deg glide
# slope, closest point 120 m, on June 22 at about 70 kt and on June 26 at the best rate of climb speed, 10 kt over and 10 kt under it, read 76, 59 and
# 56.5 kt. Three of the June 22 approaches read -1900 ft/min, a moment's reading: their levels are the others'. The 100 kt level is a single run
UH60_FAA = [(100.0, 0.0, 152.4, 80.3, 89.4), (115.0, 0.0, 152.4, 81.8, 90.9), (132.0, 0.0, 152.4, 84.0, 93.9), (152.0, 0.0, 152.4, 86.1, 97.8),
            (165.0, 0.0, 152.4, 85.2, 99.8), (150.0, 0.0, 91.4, 89.2, 101.4), (150.0, 0.0, 213.4, 81.8, 93.8), (150.0, 0.0, 304.8, 79.2, 92.0),
            (70.0, -800.0, 120.0, 87.2, 98.6), (76.0, -875.0, 120.0, 86.9, 97.4), (59.0, -733.0, 120.0, 86.2, 96.6), (56.5, -650.0, 120.0, 85.5, 95.0)]

# the same report, June 26, a 150 kt flyover at 152 m, most runs flying east: beside the path (m, side, microphone height m, maximum dBA); the centre's
# 84.2 stands 1.9 dB under June 22's 86.1 at much the same condition
UH60_FAA_SITES = [(0.0, "right", 1.2, 84.2), (150.0, "right", 1.2, 84.9), (164.0, "left", 1.2, 85.6), (284.0, "left", 1.2, 82.7), (0.0, "right", 0.0, 87.7),
                  (0.0, "right", 10.0, 83.8)]

# overall level less A-weighted level under the path and 150 m beside it, those runs (dB): the side's sound is the less low-pitched
UH60_FAA_GAP = (97.3 - 84.2, 95.0 - 84.9)

# appendix D's torque, %, of level flights by indicated airspeed: 20 kt read 75-77 near the hover, 160 kt 99-101
UH60_FAA_TORQUE = {100.0: 49.0, 115.0: 59.0, 132.0: 69.5, 152.0: 88.5, 165.0: 100.0}

# NASA CR-2377 (Healy, 1974) fig. 20(B), digitised: the Prue-2 glider at 152.4 m over the microphones, heavy, the runs' spectra as Healy smoothed them,
# without their tones and the ambient below about 300 Hz (kt: {Hz: dB}); table III their overall levels, which add 3 dB for the part below 300 Hz;
# eq. 5a his law for an aircraft's overall level
_PF = (400, 500, 630, 800, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300)
PRUE2 = {59.1: dict(zip(_PF, (24.0, 24.0, 23.7, 23.1, 21.3, 19.5, 18.1, 16.0, 13.1, 10.0, 6.3, 2.0))),
         81.9: dict(zip(_PF, (33.0, 33.1, 33.0, 32.6, 30.5, 29.2, 28.0, 26.2, 24.2, 21.4, 18.0, 13.6))),
         99.9: dict(zip(_PF, (35.8, 35.8, 35.8, 35.6, 34.8, 34.1, 33.1, 31.9, 30.1, 28.3, 25.9, 22.5)))}
PRUE2_OASPL = {59.1: 35.3, 81.9: 44.6, 99.9: 48.3}

# the Cessna 150 of the same test: fixed gear, no fairings, fig. 21 its smoothed spectra (kt: {Hz: dB}, digitised); the tyres about 0.44 m (two) and 0.36 m
CESSNA150 = {58.1: {200: 40.6, 250: 39.7, 315: 39.0, 400: 38.1, 500: 37.2, 630: 36.2, 800: 34.7},
             90.7: {200: 50.6, 315: 49.4, 400: 48.7, 500: 47.9, 630: 47.0, 800: 46.0}}
CESSNA150_TYRES = ((0.44, 2), (0.36, 1))

# a YF-102's combustion noise, NASA TM-73748 (Karchmer, Reshotko and Montegani, 1977): table I, overall power dB re 1e-13 W, by fan speed %: GE's method
# (its ref. 9, FAA-RD-77-4), the far field's 50-200 Hz bands, an upper bound, and the part coherent with one combustor probe, which NASA TM-88971
# (Krejsa, 1987) puts 3 to 10 dB low. Its fig. 10, digitised: directivity index, dB, by angle from the inlet. TM-88971 fig. 2: the core noise by three
# signal coherence at 95 % speed, 120 deg, third octaves, dB; its axis prints 80 on the first tick where the spacing makes it 50 Hz
YF102_POWER = {30: (127.0, 125.0, 120.0), 43: (131.5, 128.2, 123.5), 60: (136.0, 134.5, 126.2)}
YF102_DI = {40: -5.9, 50: -3.9, 60: -3.9, 70: -2.8, 80: -1.9, 90: -0.5, 100: 1.5, 110: 2.2, 120: 2.3, 130: 1.5, 140: 1.1, 150: 1.0, 160: -1.5}
YF102_CORE = {50: 78.9, 63: 80.7, 80: 80.2, 100: 81.9, 125: 84.8, 160: 87.7, 200: 87.3, 250: 87.0, 315: 91.3, 400: 92.2, 500: 97.6, 630: 92.8, 800: 91.4,
              1000: 90.2}
# GE's method less the data for a GEnx-1B and a GE90-115B in flight (Nesbitt, Clark, Guo and Thomas, AIAA 2025-3896): dB, low to high power, read off
GE_MODERN = (10.0, 21.0)
# NACA 1329 (Callaghan and Coles, 1957): sound power over Lighthill's parameter rho A V^8 / c^5, cold jets and engines alike
NACA1329_K = 2.7e-5

# Noise and Military Service (National Academies, 2006) table 3-1, from the US Army Center for Health Promotion and Preventive Medicine (2004):
# in flight, the condition not stated, dBA
ARMY_COCKPIT_DBA = dict(uh60a_pilot=106.0, uh60a_copilot=106.0, ch47d=103.0, yah64_pilot=104.0, yah64_copilot=101.0)

MV22_HOVER_DBA = {50.0: 105.3, 100.0: 100.4, 200.0: 95.5, 500.0: 84.5}
MV22_AIRPLANE_DBA = {152.4: 88.0, 304.8: 81.0, 457.2: 78.0}         # directly below at 220 kt: 500, 1000, 1500 ft

XV15_BOARD = dict(radii=57.2, ahead_pa=2.0, behind_23_pa=5.0, behind_46_pa=15.0, fences_dba=4.1)

T700_GE = [(87.5, 132.0, 1472.0), (91.4, 161.7, 1578.0), (94.5, 188.9, 1675.0), (96.7, 212.3, 1778.0), (98.5, 231.5, 1896.0),
           (99.5, 240.5, 1954.0)]
AE1107C = dict(pressure_ratio=16.7, flow_kg_s=15.9, t4_c=1209.0)

# read off an ASK 21's instruments in a roll (a video, about 19 s): t s, ASI km/h; t s, load factor. Where the ASI misreads (in the
# slips) it is interpolated
ROLL_ASI = [(0, 136), (0.5, 138), (1, 144), (1.5, 147), (2, 150), (2.5, 154), (3, 157), (3.5, 160), (4, 163), (4.5, 160), (5, 157),
            (5.5, 157), (6, 150), (6.5, 150), (7, 142), (7.5, 146), (8, 141), (8.5, 130), (9, 128), (9.5, 125), (12, 150), (12.4, 155),
            (13, 155), (13.6, 155), (14, 158), (14.4, 160), (14.8, 160), (17.2, 160), (17.4, 165), (17.6, 168), (18, 166), (18.5, 171),
            (19, 173)]
ROLL_G = [(0, 1.4), (0.5, 1.6), (1, 1.8), (1.5, 2.3), (2, 2.8), (2.5, 2.9), (3, 3.0), (3.5, 3.3), (4, 3.5), (4.5, 3.5), (5, 3.5),
          (5.5, 3.5), (6, 3.4), (6.5, 3.3), (7, 2.7), (7.5, 2.4), (8, 1.7), (8.5, 1.2), (9, 0.1), (9.5, 0.0), (10, 0.0), (10.5, 0.4),
          (11, 1.0), (11.5, 1.5), (12, 1.6), (12.5, 2.8), (13, 3.3), (13.5, 3.8), (14, 4.0), (14.5, 4.0), (15, 3.0), (15.5, 1.8),
          (16, 1.6), (16.5, 2.6), (17, 3.5), (17.5, 4.3), (18, 4.5), (18.5, 4.5), (19, 4.6)]
ROLL_GAPS = [(9.5, 12.0), (14.8, 17.2)]


# X-57 high-lift propeller in NASA's Low Speed Aeroacoustic Wind Tunnel (NTRS 20210016834, fig. 16), 4800 rpm, 75 KEAS, read off the
# figure to about 1 dB. The array runs along the tunnel, 3.54 m from the hub at its nearest. The left-handed blades are the
# heavier batch, which NASA's blade element model matches; the right-handed made 16 % less torque
X57_ANGLE = [53, 57, 61.5, 66, 70.5, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 128, 131, 133, 136, 138.5, 141, 144, 147, 149.5, 152]
X57_BPF_LH = [66.4, 71.3, 74.6, 77.0, 79.0, 80.0, 80.4, 80.0, 79.3, 79.4, 79.7, 79.3, 78.0, 75.6, 72.4, 66.0, 63.5, 65.2, 67.2, 70.4,
              69.6, 71.2, 71.1, 70.8, 70.5, 68.8]
X57_BPF_RH = [58.8, 61.0, 67.0, 70.1, 73.4, 75.0, 75.3, 74.5, 73.0, 71.7, 72.1, 73.5, 74.4, 73.3, 70.7, 64.0, 53.6, 52.3, 60.9, 65.4,
              64.0, 65.6, 65.7, 64.0, 64.5, 61.5]
# the same figure's spectrum at 95 deg, in 10 Hz bands, between its tones: the propeller and the empty tunnel at 75 KEAS
X57_BROADBAND_HZ = [500, 1000, 2000, 3000, 5000, 10000, 20000]
X57_BROADBAND_DB = [51, 48, 47, 44, 41, 36, 31]
X57_BACKGROUND_DB = [46, 40, 36, 35, 33, 26, 19]

# NASA's final high-lift tables (ANLYS-CEPT-023, last updated 8 October 2021): KEAS, rpm, Nm, kW, lbf, pressure altitude ft. Airspeed
# mode (appendix B tables 1 to 4) at 2289 ft, Fixed mode (appendix C tables 14 to 16) at sea level
X57_SCHEDULE = [[35, 3744, 11.7, 4.6, 30.8, 2289], [50, 4944, 20.5, 10.6, 52.6, 2289], [58, 5020, 20.5, 10.8, 50.4, 2289],
                [65, 4704, 16.6, 8.2, 38.0, 2289], [75, 4284, 11.1, 5.0, 22.2, 2289], [85, 3925, 5.5, 2.3, 8.9, 2289],
                [35, 4800, 21.4, 10.8, 60.2, 0], [58, 4800, 19.9, 10.0, 48.6, 0], [75, 4800, 17.5, 8.8, 38.2, 0], [90, 4800, 14.2, 7.1, 27.9, 0]]
# NASA's cruise propeller CFD, 2500 ft, 2250 rpm (Mod-III Power On CFD Results): ft/s, blade pitch deg, lbf, kW
X57_CRUISE = [[75.0, 18, 266.0, 45.0], [75.0, 26, 360.3, 73.4], [112.5, 18, 194.2, 38.9], [112.5, 26, 338.4, 77.9],
              [150.0, 18, 96.8, 24.0], [150.0, 22, 194.6, 48.3], [150.0, 26, 284.5, 74.7]]
