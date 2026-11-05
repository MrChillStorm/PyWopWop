#!/usr/bin/env python3
"""The report's tables (tab/tab_*.tex) and the numbers quoted in its text (tab/numbers.tex, as \\N... macros), from report/results/*.json
(mkresults.py) and the data they are compared with."""
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refdata as D

TAB = os.path.join(HERE, "tab")
NUM = {}


def R(name):
    return json.load(open(os.path.join(HERE, "results", name + ".json")))


def num(name, value, fmt="%.1f"):
    assert re.fullmatch("[A-Za-z]+", name), name
    s = (fmt % value) if not isinstance(value, str) else value
    NUM[name] = "$-$" + s[1:] if s.startswith("-") else s


def write(name, body):
    open(os.path.join(TAB, "tab_%s.tex" % name), "w").write(body)


def f(x, fmt="%+.1f"):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else (fmt % x).replace("-", "$-$")


def minus(r):
    """a table cell's minus signs, as a typesetter wants them"""
    return re.sub(r"(?<![\w$.-])-(?=\d)", "$-$", r)


def tabular(cols, head, rows, foot=""):
    return ("\\begin{tabular}{@{}%s@{}}\n\\toprule\n%s \\\\\n\\midrule\n" % (cols, head) + "".join(minus(r) + " \\\\\n" for r in rows) + foot + "\\bottomrule\n\\end{tabular}\n")


def verify():
    v = R("verify")
    gut = max(abs(g["got"] - g["ref"]) for g in v["gutin"])
    num("NBpmDtwo", v["bpm_d2_err"], "%.3f")
    num("NBpmDthree", v["bpm_d3_err"], "%.3f")
    num("NBpmTip", v["bpm_d3_tip_err"], "%.2f")
    num("NGutin", gut, "%.2f")
    num("NFoneHover", 100 * max(v["f1_hover"].values()), "%.3f")
    num("NFoneFly", 100 * max(v["f1_flyby"].values()), "%.2f")
    num("NClosure", "%.0e" % max(v["closure"], 1e-16))
    num("NLoopAbove", abs(v["loop_above_db"]), "%.0f")
    num("NBoFlapTwo", v["bo105_flap"][1], "%.2f")
    num("NTests", v["pytest"].split()[0], "%s")
    rows = ["BPM self-noise, untripped section & NASA RP-1218 table D2 & %.3f\\,dB" % v["bpm_d2_err"],
            "BPM, tripped, ten segments & table D3 (tip column %.2f\\,dB) & %.3f\\,dB" % (v["bpm_d3_tip_err"], v["bpm_d3_err"]),
            "F1A loading, hover harmonics & Gutin's closed form & %.2f\\,dB" % gut,
            "F1A, hover, unsteady loads and heave & Formulation 1, numerical derivative & %.3f\\,\\%%" % (100 * max(v["f1_hover"].values())),
            "F1A, a fly-by past a fixed listener & the same & %.2f\\,\\%%" % (100 * max(v["f1_flyby"].values())),
            "Reception time, closed form & the retarded-time equation & round-off",
            "Absorption & ISO 9613-2 table 2, as printed & none",
            "Bo 105 second flap frequency from the first & published 2.78/rev & %.2f/rev" % v["bo105_flap"][1],
            "Loops: above the band limit & none & $-$%.0f\\,dB" % abs(v["loop_above_db"])]
    write("verify", tabular("l@{\\hspace{3pt}}l@{\\hspace{3pt}}r", "Check & Against & Error", [r for r in rows]))


def rotors():
    r = R("rotors")
    b = R("before_rotors")
    t = r["xv15_test"]
    ratio = [a["cp_s"] / np.interp(a["ct_s"], t["ct_s"], t["cp_s"]) for a in r["xv15"]]
    num("NXvCpMin", 100 * (min(ratio) - 1), "%+.0f")
    num("NXvCpMax", 100 * (max(ratio) - 1), "%+.0f")
    num("NXvFmPeak", max(a["fm"] for a in r["xv15"]), "%.2f")
    rb = [a["cp_s"] / np.interp(a["ct_s"], t["ct_s"], t["cp_s"]) for a in b["xv15"]]
    num("NXvFalse", max(rb), "%.1f")
    bo = r["bo105"]
    q = [a["cp_s"] / a["cp_s_test"] for a in bo]
    num("NBoCpMin", 100 * (min(q) - 1), "%+.0f")
    num("NBoCpMax", 100 * (max(q) - 1), "%+.0f")
    off = [a["theta_test"] - a["theta"] for a in bo]
    num("NBoThMin", min(off))
    num("NBoThMax", max(off))
    ob = [a["theta_test"] - a["theta"] for a in b["bo105"]]
    num("NBoThBeforeMin", min(ob))
    num("NBoThBeforeMax", max(ob))
    tt = np.array(D.BO105_TORSION_034)
    e = [a["torsion_034"] / np.interp(a["ct_s"], tt[:, 0], tt[:, 1]) - 1 for a in bo]
    num("NBoTorsMax", 100 * max(abs(x) for x in e), "%.0f")
    num("NBoTwistHi", bo[-1]["twist_tip"])
    num("NBoTwistBefore", b["bo105"][-1]["twist_tip"], "%+.1f")
    ff = r["bo105_forward"]
    lo = [a["cp_s"] / a["cp_s_test"] for a in ff if a["mu"] < 0.2]
    hi = [a["cp_s"] / a["cp_s_test"] for a in ff if a["mu"] > 0.27]
    num("NBoFwdLo", 100 * (np.mean(lo) - 1), "%+.0f")
    num("NBoFwdHi", 100 * (np.mean(hi) - 1), "%+.0f")
    uh = r["uh60_hover"]
    u = [a["cp"] / a["cp_test"] for a in uh]
    num("NUhHoverMin", 100 * (min(u) - 1), "%+.0f")
    num("NUhHoverMax", 100 * (max(u) - 1), "%+.0f")
    for k, n in (("65", "A"), ("80", "B")):
        rows = r["uh60_level"][k]
        lo = [a["cp"] / a["cp_test"] for a in rows if a["mu"] <= 0.24]
        hi = [a["cp"] / a["cp_test"] for a in rows if a["mu"] > 0.28]
        num("NUhLevel%sLo" % n, 100 * (np.mean(lo) - 1), "%+.0f")
        num("NUhLevel%sHi" % n, 100 * (np.mean(hi) - 1), "%+.0f")


def stand():
    s = R("stand")
    b = R("before_stand")
    t = s["test"]
    rows = s["rows"]
    dn = np.array([q["near"]["total"] - np.interp(q["lb"], t["thrust_lb"], t["near_db"]) for q in rows])
    df = np.array([q["far"]["total"] - np.interp(q["lb"], t["far_thrust_lb"], t["far_db"]) for q in rows])
    num("NStNearMean", dn.mean(), "%+.1f")
    num("NStNearRms", float(np.sqrt(np.mean(dn ** 2))))
    num("NStNearMax", float(np.abs(dn).max()))
    num("NStFarMean", df.mean(), "%+.1f")
    num("NStFarRms", float(np.sqrt(np.mean(df ** 2))))
    q = min(rows, key=lambda a: a["near"]["total"])
    num("NStDipLb", q["lb"], "%.0f")
    num("NStDipMod", q["near"]["total"])
    tn = np.array(t["near_db"])
    num("NStDipTest", tn.min())
    num("NStDipTestLb", t["thrust_lb"][int(np.argmin(tn))], "%.0f")
    tl = np.array([a["tip_lift"] for a in rows])
    lb = np.array([a["lb"] for a in rows], float)
    i = int(np.argmax(tl > 0))
    num("NStTipZero", lb[i - 1] + (lb[i] - lb[i - 1]) * (-tl[i - 1]) / (tl[i] - tl[i - 1]), "%.0f")
    hi = next(a for a in b["rows"] if a["lb"] == 10000)
    num("NStBeforeHi", hi["near"]["total"])
    num("NStBeforeFar", hi["far"]["total"])
    num("NStTestHi", float(np.interp(10000, t["thrust_lb"], t["near_db"])))
    num("NStTestFarHi", float(np.interp(10000, t["far_thrust_lb"], t["far_db"])))


def hart2():
    h = R("hart2")
    b = R("before_hart2")
    m = h["measured_stats"]
    num("NHtMean", m["mean"], "%.3f")
    num("NHtRms", m["rms"], "%.4f")
    G = np.array(h["measured"])
    y = np.array(h["y"])
    num("NHtAdv", G[:, y > 0].max(), "%.0f")
    num("NHtRet", G[:, y < 0].max(), "%.0f")
    rows = []
    for c in h["cases"]:
        L = np.array(c["L"])
        cc = np.corrcoef(L.ravel(), G.ravel())[0, 1]
        rows.append("%.1f & %.2f & %.1f & %.3f & %.4f & %.1f & %.1f & %s & %.1f & %.2f" % (c["alpha"], c["core"], c["rollup"], c["stats"]["mean"], c["stats"]["rms"], c["max_adv"], c["max_ret"], f(c["mean_err"]), c["rms_err"], cc))
    rows.append("\\multicolumn{3}{@{}l}{measured} & %.3f & %.4f & %.0f & %.0f & & & " % (m["mean"], m["rms"], G[:, y > 0].max(), G[:, y < 0].max()))
    write("hart2", tabular("rrrrrrrrrr", "$\\alpha_s$ & core & roll & $\\overline{M^2 c_n}$ & rms & adv. & ret. & mean & rms & $r$", rows))
    c = h["cases"][0]
    o = b["cases"][0]
    L = np.array(c["L"])
    num("NHtCore", c["core"], "%.1f")
    num("NHtModMean", c["stats"]["mean"], "%.3f")
    num("NHtModRms", c["stats"]["rms"], "%.4f")
    num("NHtRmsRatio", c["stats"]["rms"] / m["rms"])
    num("NHtBeforeRatio", o["stats"]["rms"] / m["rms"])
    num("NHtAdvMod", c["max_adv"])
    num("NHtRetMod", c["max_ret"])
    num("NHtRetDiff", G[:, y < 0].max() - c["max_ret"], "%.0f")
    num("NHtMapMean", c["mean_err"], "%+.1f")
    num("NHtMapRms", c["rms_err"])
    num("NHtMapR", np.corrcoef(L.ravel(), G.ravel())[0, 1], "%.2f")
    num("NHtNoBvi", c["no_bvi_max"], "%.0f")
    num("NHtBvidB", c["max_adv"] - c["no_bvi_max"], "%.0f")
    num("NHtBeforeMean", o["mean_err"], "%+.1f")
    num("NHtBeforeAdv", o["max_adv"])
    num("NHtPower", c["power_kw"])
    t = np.array(c["tip_twist"])
    num("NHtTwist", t.mean())
    num("NHtTwistBefore", float(np.mean(o["tip_twist"])), "%.2f" if "tip_twist" in o else "%s")
    fl = np.array(c["tip_flap"])
    num("NHtFlapLo", fl.min(), "%.3f")
    num("NHtFlapHi", fl.max(), "%.3f")
    meas = next((q for q in h["cases"] if abs(q["core"] - 0.12) < 1e-6), None)
    if meas:
        num("NHtMeasCoreMean", meas["mean_err"], "%+.1f")
        num("NHtMeasCoreAdv", meas["max_adv"])
        num("NHtMeasCoreRatio", meas["stats"]["rms"] / m["rms"])


def bvi():
    r = R("bvi")
    a = r["alpha"]
    p = max(a, key=lambda q: q["peak"])
    num("NBvAlphaPeak", p["alpha"])
    num("NBvAlphaLevel", p["peak"])
    d = r["descent"]
    for nm, n in (("uh60a", "Uh"), ("uh1h", "Hu")):
        rows = [q for q in d if q["aircraft"] == nm]
        p = max(rows, key=lambda q: q["slap"])
        lvl = next(q for q in rows if q["fpm"] == 0.0)
        num("NBv%sFpm" % n, -p["fpm"], "%.0f")
        num("NBv%sTotal" % n, p["total"] - lvl["total"])
        num("NBv%sSlap" % n, p["slap"])


def texture():
    t = R("texture")
    rec = t["recordings"]
    rows = ["%s & %.2f & %+.0f & %.0f" % (q["tag"], q["corr"], q["tones_db"], q["crest_db"]) for q in rec]
    rows = [r.replace("-", "$-$") if False else r for r in rows]
    for q in [a for a in t["model"] if a["tag"] != "cruise_ahead"]:
        v = q["varied"]
        s = q["steady"]
        rows.append("%s %s (steady: %.2f) & %.2f & %+.0f & %.0f" % ("UH-60A" if q["aircraft"] == "uh60a" else "Bo 105", q["tag"], s["corr"], v["corr"], v["tones_db"], v["crest_db"]))
    write("texture", tabular("lrrr", "Sound & corr. & tones, dB & crest, dB", [x.replace("+", "$+$").replace("-", "$-$") if False else x for x in rows]))
    c = [q["corr"] for q in rec]
    num("NTxRecLo", min(c), "%.2f")
    num("NTxRecHi", max(c), "%.2f")
    mm = [q["varied"]["corr"] for q in t["model"]]
    num("NTxModLo", min(mm), "%.2f")
    num("NTxModHi", max(mm), "%.2f")
    num("NTxSteady", max(q["steady"]["corr"] for q in t["model"]), "%.2f")
    u = {q["tag"]: q for q in t["model"] if q["aircraft"] == "uh60a"}
    o = {q["tag"]: q for q in R("before_texture")["model"] if q["aircraft"] == "uh60a"}
    num("NTxApproach", u["approach"]["dba"] - u["cruise_ahead"]["dba"], "%+.0f")
    num("NTxApproachBefore", o["approach"]["dba"] - o["cruise_ahead"]["dba"], "%+.0f")


def fountain():
    rows = R("fountain")["rows"]
    num("NFoFences", D.XV15_BOARD["fences_dba"])
    get = lambda az, el, on: next(q for q in rows if q["az"] == az and abs(q["el"] - el) < 0.1 and q["on"] == on)
    els = (-12.6, -23.0, -45.7)
    num("NFoBehind", float(np.mean([get(180, e, True)["dba"] - get(180, e, False)["dba"] for e in els])))
    num("NFoAhead", float(np.mean([get(0, e, True)["dba"] - get(0, e, False)["dba"] for e in els])), "%.1f")
    swing = lambda q: max(abs(q["whole_hi"]), abs(q["whole_lo"]))
    num("NFoSpikeA", swing(get(180, -12.6, True)), "%.0f")
    num("NFoSpikeB", swing(get(180, -23.0, True)), "%.0f")
    num("NFoSpikeC", swing(get(180, -45.7, True)), "%.0f")
    num("NFoAheadPa", max(max(abs(get(0, e, True)["mean_hi"]), abs(get(0, e, True)["mean_lo"])) for e in els), "%.0f")    # the averaged pass


def outside():
    o = R("outside")
    h = o["hover"]
    d = sorted(D.MV22_HOVER_DBA)
    sel = lambda c: {q["slant"]: q for q in h if c(q)}
    behind = sel(lambda q: q["az"] == 180 and q["on"] and not q["low"])
    side = sel(lambda q: q["az"] == 90 and q["on"] and not q["low"])
    nof = sel(lambda q: q["az"] == 180 and not q["on"])
    low = sel(lambda q: q["low"])
    rows = []
    for k in d:
        rows.append("%.0f & %.1f & %.1f & %.1f & %.1f & %.1f" % (k, D.MV22_HOVER_DBA[k], behind[k]["fast"], side[k]["fast"], nof[k]["fast"], low[k]["fast"]))
    write("outside", tabular("rrrrrr", "m & table & behind & side & no fountain & hub 30 m", rows))
    num("NOuBehindMax", max(abs(behind[k]["fast"] - D.MV22_HOVER_DBA[k]) for k in d))
    num("NOuFountain", float(np.mean([behind[k]["fast"] - nof[k]["fast"] for k in d])))
    b = {q["wing"]: q for q in o["below"]}
    num("NOuBelow", b[True]["dba"])
    num("NOuBelowNoWing", b[False]["dba"], "%.0f")
    for n, k in (("A", 152.4), ("B", 304.8), ("C", 457.2)):
        num("NOuSpread%s" % n, b[True]["dba"] - 20 * math.log10(k / 100.0))


def engines():
    e = R("engines")
    t = R("tones")
    c = e["t700"]
    num("NEnPr", 100 * max(abs(a["pr"] / a["pr_test"] - 1) for a in c))
    num("NEnTemp", max(abs(a["t45"] - a["t45_test"]) for a in c), "%.0f")
    a = e["ae1107c"]
    num("NEnAeFlow", a["W"])
    num("NEnAeTemp", a["t4_c"], "%.0f")
    L = {q["tag"]: q for q in e["levels"]}
    num("NEnIdle", L["idle"]["total"], "%.0f")
    num("NEnFly", L["FLY"]["total"], "%.0f")
    num("NEnFull", L["full power"]["total"], "%.0f")
    num("NEnApu", e["apu_dba"], "%.0f")
    num("NEnMvLo", e["mv22"][0]["total"], "%.0f")
    num("NEnMvHi", e["mv22"][-1]["total"], "%.0f")
    i = t["idle"]
    num("NToIdleA", i["f1"] / 1000, "%.2f")
    num("NToIdleB", i["f2"] / 1000, "%.2f")
    num("NToRatio", i["ratio"], "%.3f")
    num("NToFly", t["fly"][0] / 1000, "%.2f")
    num("NToHover", t["hover"][0] / 1000, "%.2f")
    k = e["ng_rpm"] / 60 / 100 * 20
    num("NToNgIdle", i["f1"] / k, "%.1f")
    num("NToNgFly", t["fly"][0] / k, "%.1f")
    num("NToNgHover", t["hover"][0] / k, "%.1f")
    d = t["doors"]
    num("NDoNarrow", d["narrow_before"] - d["narrow_after"])
    num("NDoBand", d["band_before"] - d["band_after"])
    tr = d["track"]
    num("NDoFromA", tr[3]["lines"][0][0] / 1000, "%.2f")
    num("NDoToA", tr[-1]["lines"][0][0] / 1000, "%.2f")
    g = t["gear"]
    num("NGeHz", g["hover"][0], "%.0f")
    num("NGeRise", g["hover"][2] - g["fly"][2], "%.0f")
    if "gear_rise_model" in e:
        num("NGeModel", e["gear_rise_model"], "%.0f")


def cabin():
    c = R("cabin")
    num("NCaDoor", c["uh60a"]["door_db"], "%.0f")
    num("NCaResonance", c["mv22b"]["f0"], "%.0f")
    w = R("wind")
    r = w["rows"]
    num("NWiLo", min(q["p_rms_q"] for q in r), "%.4f")
    num("NWiHi", max(q["p_rms_q"] for q in r), "%.4f")
    t = w["tu144"]
    d = np.array(t["model"]) - np.array(t["test"])
    num("NWiTuMean", d.mean(), "%+.1f")
    num("NWiTuShape", d.std())
    num("NWiTuQ", t["p_rms_q"], "%.4f")
    a = {q["h"]: q for q in w["altitude"]}
    num("NWiAltA", a[12000.0]["err"])
    num("NWiAltB", a[21000.0]["err"])
    num("NWiAltTop", a[21000.0]["err_all"])


def cockpit():
    c = R("cockpit")
    r = {q["tag"]: q for q in c["rows"]}
    num("NCkHover", r["hover"]["total"])
    num("NCkCruise", r["cruise"]["total"])
    num("NCkTest", c["test"]["uh60a_pilot"], "%.0f")
    num("NCkLo", c["test"]["uh60a_pilot"] - r["hover"]["total"], "%.0f")
    num("NCkHi", c["test"]["uh60a_pilot"] - r["cruise"]["total"], "%.0f")
    num("NCkGear", r["hover"]["gearbox"], "%.0f")
    num("NCkRotHover", r["hover"]["rotors"], "%.0f")
    num("NCkRotCruise", r["cruise"]["rotors"], "%.0f")
    num("NCkEng", r["hover"]["engines"], "%.0f")


def mv22():
    m = R("mv22")
    fc = np.array(m["fc"])
    d = np.array(m["model"]) - np.array(m["recording"])
    do = np.array(m["oct_mod"]) - np.array(m["oct_rec"])
    num("NMvBand", float(np.abs(do[1:]).max()))
    num("NMvLow", float(abs(do[0])), "%.0f")
    k = (fc >= 200) & (fc <= 5000)
    num("NMvThirdMid", float(np.abs(d[k]).max()))
    k = (fc >= 45) & (fc <= 200)                    # from the 50 Hz third octave, centred on 49.6
    num("NMvThirdLow", float(np.abs(d[k]).max()), "%.0f")
    fl = np.array(m["floor_mod"]) - np.array(m["floor_rec"])
    num("NMvFloor", float(np.sort(np.abs(fl))[-2]))
    num("NMvBpf", m["bpf_rec"], "%.2f")
    num("NMvRpm", m["bpf_rec"] / 3 * 60, "%.0f")
    P = m["parts"]
    num("NMvRotors", P["proprotors"], "%.1f")
    num("NMvDrive", P["drive"], "%.1f")
    num("NMvWind", P["wind"], "%.1f")
    num("NMvEngines", P["engines"], "%.1f")
    num("NMvAll", m["all_dba"], "%.1f")
    H = np.array(m["harmonics"])
    orr, om = np.array(m["over_rec"]), np.array(m["over_mod"])
    for n, lo, hi in (("A", 2, 5), ("B", 6, 13), ("C", 14, 30)):
        s = (H >= lo) & (H <= hi)
        num("NMvOverRec%s" % n, float(orr[s].mean()))
        num("NMvOverMod%s" % n, float(om[s].mean()))
    t = m["tones"]
    rows = []
    for key, lab in (("7080.0", "whine, 7.08 kHz"), ("2030.0", "mesh, 2.03 kHz"), ("8170.0", "mesh, 8.17 kHz")):
        rows.append("%s, over its neighbours, dB & %.1f & %.1f" % (lab, t[key]["rec"], t[key]["mod"]))
    for s, q in zip(m["stats_rec"][:4], m["stats_mod"][:4]):
        rows.append("%d--%d\\,Hz: blade lines' share, \\%% & %.0f & %.0f" % (s["lo"], s["hi"], s["share"], q["share"]))
    for s, q in zip(m["stats_rec"][:4], m["stats_mod"][:4]):
        rows.append("%d--%d\\,Hz: pulsing per pass, dB & %.1f & %.1f" % (s["lo"], s["hi"], s["pulse"], q["pulse"]))
    for s, q in zip(m["stats_rec"][-4:], m["stats_mod"][-4:]):
        rows.append("%d--%d\\,Hz: pulsing per pass, dB & %.1f & %.1f" % (s["lo"], s["hi"], s["pulse"], q["pulse"]))
    for n, lab in (("A", "2--5"), ("B", "6--13"), ("C", "14--30")):
        s = {"A": (2, 5), "B": (6, 13), "C": (14, 30)}[n]
        k = (H >= s[0]) & (H <= s[1])
        rows.append("harmonics %s over the floor, dB & %.1f & %.1f" % (lab, orr[k].mean(), om[k].mean()))
    write("mv22", tabular("lrr", "Measure & recording & model", rows))
    w = m["window"]
    num("NMvWhine", w["cruise"][0] / 1000, "%.2f")
    num("NMvWhineOff", w["power_off"][0] / 1000, "%.2f")
    c = R("chop")
    num("NMvChop", c["line"], "%.0f")
    num("NMvChopBeside", c["beside"], "%.0f")
    try:
        t = R("mv22_terms")
        names = dict(full="the model", no_wing="no wing source", in_step="rotors in step", sea_level="sea-level air", single_wall="single wall, no trim")
        rows = ["%s & %.1f & %.1f & %.1f & %.1f & %.1f & %.0f" % (names[q["name"]], q["rotors_dba"], q["h2_5"], q["h6_13"], q["h14_30"], q["pulse_100"], q["share_100"]) for q in t["rows"]]
        rc = t["rec"]
        rows.append("recording & & %.1f & %.1f & %.1f & %.1f & %.0f" % (rc["h2_5"], rc["h6_13"], rc["h14_30"], m["stats_rec"][1]["pulse"], m["stats_rec"][1]["share"]))
        write("terms", tabular("lrrrrrr", "Case & dBA & h2--5 & h6--13 & h14--30 & pulse & share", rows))
    except FileNotFoundError:
        write("terms", "")


def ask21():
    a = R("ask21")
    num("NAsStall", a["stall_kt"])
    fc = np.array(a["fc"])
    s = (fc >= 315) & (fc <= 6300)
    e = [float(np.sqrt(np.mean((np.array(q["rec"])[s] - np.array(q["mod"])[s]) ** 2))) for q in a["shapes"]]
    num("NAsShapeLo", min(e))
    num("NAsShapeHi", max(e))
    num("NAsSpeedLo", a["shapes"][0]["kt"], "%.0f")
    num("NAsSpeedHi", a["shapes"][-1]["kt"], "%.0f")
    num("NAsExpRec", a["exponent_rec"])
    num("NAsExpMod", a["exponent_mod"])
    num("NAsHighRec", a["high_rec"], "%.0f")
    num("NAsHighMod", a["high_mod"], "%.0f")
    s2 = (fc >= 400) & (fc <= 4000)
    num("NAsSecond", float(np.max(np.abs(np.array(a["second"]["rec"])[s2] - np.array(a["second"]["mod"])[s2]))))
    num("NAsSecondSec", a["second"]["seconds"], "%.0f")
    r = a["roll"]
    num("NAsRollRec", r["swing_rec"], "%.0f")
    num("NAsRollMod", r["swing_mod"], "%.0f")
    num("NAsPerGRec", r["per_g_rec"])
    num("NAsPerGMod", r["per_g_mod"])
    d = {float(k): v for k, v in a["dba"].items()}
    num("NAsLevelLo", d[50.0], "%.0f")
    num("NAsLevelHi", d[120.0], "%.0f")


def bench():
    b = R("bench")
    txt = "\n".join(b["lines"])
    m = re.search(r"retarded-time.*?:\s+([\d.]+) ms", txt)
    s = re.search(r"blade symmetry:\s+([\d.]+) ms \((\d+)x", txt)
    num("NBeRet", float(m.group(1)), "%.0f")
    num("NBeSrc", float(s.group(1)), "%.0f")
    num("NBeRatio", float(s.group(2)), "%.0f")
    num("NBeTrim", float(re.search(r"trim, elastic response and wake:\s+([\d.]+) s", txt).group(1)))
    num("NBeFa", float(re.search(r"F1A at one observer:\s+([\d.]+) s", txt).group(1)), "%.2f")
    q = re.search(r"\((\d+) load points and (\d+) panels, (\d+) M", txt)
    num("NBeRate", float(q.group(3)), "%.0f")
    num("NBeLoop", b["loop_15s"], "%.0f")
    num("NBeLoad", b["load"])


def before_after():
    """the corrections the rotor tests forced, each measure before and after"""
    rb, ra = R("before_rotors"), R("rotors")
    hb, ha = R("before_hart2"), R("hart2")
    sb, sa = R("before_stand"), R("stand")
    t = sa["test"]
    near = lambda s, lb: next(a for a in s["rows"] if a["lb"] == lb)["near"]["total"]
    rows = ["Bo 105 tip twist, $C_T/\\sigma$ 0.12, deg & %+.1f & %+.1f & nose down" % (rb["bo105"][-1]["twist_tip"], ra["bo105"][-1]["twist_tip"]),
            "Bo 105 torsion at 0.34\\,$R$, in\\,lb & %.0f & %.0f & $-$500" % (rb["bo105"][-1]["torsion_034"], ra["bo105"][-1]["torsion_034"]),
            "Bo 105 collective, test less model, deg & %.1f--%.1f & %.1f--%.1f & (FDM 2.5--3.1)" % (min(a["theta_test"] - a["theta"] for a in rb["bo105"]), max(a["theta_test"] - a["theta"] for a in rb["bo105"]),
                                                                                          min(a["theta_test"] - a["theta"] for a in ra["bo105"]), max(a["theta_test"] - a["theta"] for a in ra["bo105"])),
            "HART II tip twist, mean, deg & %.1f & %.1f & about $-$1" % (np.mean(hb["cases"][0].get("tip_twist", [float("nan")])), np.mean(ha["cases"][0]["tip_twist"])),
            "HART II BVI load, rms, ratio to test & %.1f & %.1f & 1" % (hb["cases"][0]["stats"]["rms"] / hb["measured_stats"]["rms"], ha["cases"][0]["stats"]["rms"] / ha["measured_stats"]["rms"]),
            "HART II noise map, mean error, dB & %+.1f & %+.1f & 0" % (hb["cases"][0]["mean_err"], ha["cases"][0]["mean_err"]),
            "XV-15 near field, 10\\,000\\,lb, dB & %.1f & %.1f & %.1f" % (near(sb, 10000), near(sa, 10000), np.interp(10000, t["thrust_lb"], t["near_db"]))]
    write("before", tabular("lrrr", "Measure & before & after & test", [r.replace("-", "$-$") if False else r for r in rows]))


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    for fn in (verify, rotors, stand, hart2, bvi, texture, fountain, outside, engines, cabin, cockpit, mv22, ask21, bench, before_after):
        try:
            fn()
        except (FileNotFoundError, KeyError) as e:
            print("skipped", fn.__name__, repr(e))
    with open(os.path.join(TAB, "numbers.tex"), "w") as out:
        for k in sorted(NUM):
            out.write("\\newcommand{\\%s}{%s}\n" % (k, NUM[k]))
    print(len(NUM), "numbers")
