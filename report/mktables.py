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
    rows = {q["kt"]: q for q in c["rows"]}
    fc = np.array(c["fc"])
    r = rows[120]
    d = np.array(r["octaves"]) - np.array(c["pilot"]["bands"])
    num("NCkBandMax", float(np.abs(d[fc >= 125]).max()))
    num("NCkLowHi", float(d[fc < 100].max()), "%.0f")
    num("NCkLowLo", float(d[fc < 100].min()), "%.0f")
    num("NCkCruise", r["dba"])
    num("NCkPilot", c["pilot"]["dba"], "%.0f")
    num("NCkCopilot", c["copilot"]["dba"], "%.0f")
    num("NCkNasaCruise", c["nasa"]["120"], "%.0f")
    num("NCkHover", rows[0]["dba"])
    num("NCkNasaHover", c["nasa"]["0"], "%.0f")
    num("NCkArmy", c["army"]["uh60a_pilot"], "%.0f")
    e = np.array([q["dba"] - c["nasa"][str(q["kt"])] for q in c["rows"]])
    num("NCkNasaMean", e.mean(), "%+.1f")
    num("NCkNasaSd", e.std())
    P = rows[0]["parts"]
    num("NCkDrive", P["drive"], "%.0f")
    num("NCkRotHover", P["rotors"], "%.0f")
    num("NCkEng", P["engines"], "%.0f")
    num("NCkWind", r["parts"]["wind"], "%.0f")
    o = next(q for q in R("before_cockpit")["rows"] if q["kt"] == 120)
    d0 = np.array(o["octaves"]) - np.array(c["pilot"]["bands"])
    under = -d0[(fc >= 250) & (fc != 1000)]
    num("NCkOldCruise", o["dba"])
    num("NCkOldTone", float(d0[fc == 1000][0]), "%.0f")
    num("NCkOldUnderLo", float(under.min()), "%.0f")
    num("NCkOldUnderHi", float(under.max()), "%.0f")


def f_(x):
    return "0.0" if abs(x) < 0.05 else ("%+.1f" % x).replace("-", "$-$")


def flight():
    f = R("flight")["rows"]
    d = np.array([q["dba"] - q["test_dba"] for q in f])
    lvl = np.array([q["fpm"] == 0 for q in f])
    num("NFlMean", d.mean(), "%+.1f")
    num("NFlRms", float(np.sqrt(np.mean(d ** 2))))
    num("NFlLevelMean", d[lvl].mean(), "%+.1f")
    num("NFlLevelSd", d[lvl].std())
    a = [q for q, l in zip(f, lvl) if not l]
    da = d[~lvl]
    num("NFlAppMean", da.mean(), "%+.1f")
    num("NFlAppMin", float(da.min()), "%+.1f")
    num("NFlAppMax", float(da.max()), "%+.1f")
    num("NFlAppTestLo", min(q["test_dba"] for q in a))
    num("NFlAppTestHi", max(q["test_dba"] for q in a))
    num("NFlAppLo", min(q["dba"] for q in a))
    num("NFlAppHi", max(q["dba"] for q in a))
    num("NFlAppSlap", float(np.mean([q["slap"] for q in a])), "%.0f")
    num("NFlAppRest", float(np.mean([10 * np.log10(10 ** (q["dba"] / 10) - 10 ** (q["slap"] / 10)) for q in a])), "%.0f")
    num("NFlLevelSlap", max(q["slap"] for q, l in zip(f, lvl) if l), "%.0f")
    s = [q for q in f if q["height"] == 152.4]
    num("NFlSlowDiff", s[1]["dba"] - s[1]["test_dba"], "%+.1f")
    num("NFlFastDiff", s[3]["dba"] - s[3]["test_dba"], "%+.1f")
    num("NFlEngines", s[0]["engines"], "%.0f")
    num("NFlOaRise", s[3]["test_oaspl"] - s[0]["test_oaspl"], "%.0f")
    num("NFlTestRise", s[3]["test_dba"] - s[0]["test_dba"], "%.0f")
    num("NFlModRise", s[3]["dba"] - s[0]["dba"], "%+.1f")
    o = np.array([q["oaspl"] - q["test_oaspl"] for q in f])
    num("NFlOaLevel", o[lvl].mean(), "%+.1f")
    num("NFlOaApp", o[~lvl].mean(), "%+.1f")
    T = R("flight")["torque"]
    e = np.array([100 * (q["torque"] / T[str(q["kt"])] - 1) for q in s])
    num("NFlTorqueLo", float(e.max()), "%+.0f")
    num("NFlTorqueHi", float(e.min()), "%+.0f")
    rows = ["%s & %.0f & %.1f & %.1f & %s" % ("6$^\\circ$ approach" if q["fpm"] else "level", q["height"], q["test_dba"], q["dba"], f_(q["dba"] - q["test_dba"])) for q in f]
    rows = ["%.0f & %s" % (q["kt"], x) for q, x in zip(f, rows)]
    write("flight", tabular("rlrrrr", "kt & & m & test & model & diff.", rows))
    w = R("sites")["rows"]
    c = next(q for q in w if q["y"] == 0 and q["mic"] == 1.2)
    side = [q for q in w if q["y"] > 0]
    ds = np.array([q["dba"] - q["test"] for q in side])
    num("NFlSideMean", ds.mean(), "%+.1f")
    num("NFlSideLo", float(np.abs(ds).min()), "%.0f")
    num("NFlSideHi", float(np.abs(ds).max()), "%.0f")
    num("NFlCentre", c["dba"])
    num("NFlCentreTest", c["test"])
    near = [q for q in side if q["y"] < 200]
    far = [q for q in side if q["y"] > 200][0]
    num("NFlNearTest", float(np.mean([q["test"] for q in near])) - c["test"], "%+.1f")
    num("NFlNearMod", float(np.mean([q["dba"] for q in near])) - c["dba"], "%+.1f")
    num("NFlFarTest", far["test"] - c["test"], "%+.1f")
    num("NFlFarMod", far["dba"] - c["dba"], "%+.1f")
    gap = R("sites")["gap"]
    num("NFlGapCentre", gap[0], "%.0f")
    num("NFlGapSide", gap[1], "%.0f")
    g = next(q for q in w if q["mic"] == 0.0)
    h = next(q for q in w if q["mic"] == 10.0)
    num("NFlFlushTest", g["test"] - c["test"], "%+.1f")
    num("NFlFlushMod", g["dba"] - c["dba"], "%+.1f")
    num("NFlTallTest", h["test"] - c["test"], "%+.1f")
    num("NFlTallMod", h["dba"] - c["dba"], "%+.1f")


def prue():
    p = R("prue")["rows"]
    d = [np.array(q["at_test"]) - np.array(q["test"]) for q in p]
    a = np.concatenate(d)
    num("NPrMean", a.mean(), "%+.1f")
    num("NPrRms", float(np.sqrt(np.mean(a ** 2))))
    num("NPrShape", max(float(x.std()) for x in d))
    num("NPrLo", min(float(-x.mean()) for x in d))
    num("NPrHi", max(float(-x.mean()) for x in d))
    kt = np.log10([q["kt"] for q in p])
    num("NPrExpModel", np.polyfit(kt, [q["oaspl"] for q in p], 1)[0] / 10)
    num("NPrExpTest", np.polyfit(kt, [q["test_oaspl"] for q in p], 1)[0] / 10)
    e = np.array([q["oaspl"] - q["test_oaspl"] for q in p])
    num("NPrOaMean", e.mean(), "%+.1f")
    l = np.array([q["law"] - q["test_oaspl"] for q in p])
    num("NPrLawMean", l.mean(), "%+.1f")
    m = np.array([q["oaspl"] - q["law"] for q in p])
    num("NPrLawModel", m.mean(), "%+.1f")
    P = R("prue")
    f = np.array(p[0]["f"])
    b = (f >= 400) & (f <= 1000)
    up = [float(np.mean(np.array(q["total"])[b] - np.array(q["model"])[b])) for q in p]
    num("NWhLo", min(up), "%.0f")
    num("NWhHi", max(up), "%.0f")
    num("NWhFLo", round(float(f[int(np.argmax(p[0]["wheels"]))]), -2), "%.0f")
    num("NWhFHi", round(float(f[int(np.argmax(p[-1]["wheels"]))]), -2), "%.0f")
    e = np.concatenate([np.array(q["wheels"]) - np.array(q["test"]) for q in P["cessna"]])
    num("NWhCeLo", float(-e.max()), "%.0f")
    num("NWhCeHi", float(-e.min()), "%.0f")
    num("NWhInLo", float(-P["inside"][0]["under"]), "%.0f")
    num("NWhInHi", float(-P["inside"][1]["under"]), "%.0f")
    q0, q2 = p[0], p[-1]
    i400 = int(np.argmin(np.abs(f - 400)))
    t400 = float(np.interp(np.log(400), np.log(q0["test_f"]), q0["test"]))
    num("NWhVsPrue", float(q0["total"][i400]) - t400, "%.0f")
    A = R("ask21")
    k = int(np.argmin(np.abs(np.array(A["fc"]) - 200)))
    low = [float(q["rec"][k] - q["mod"][k]) for q in A["shapes"]]
    num("NWhRecLo", low[0], "%.0f")
    num("NWhRecHi", low[-1], "%.0f")


def core():
    c = R("core")
    d = np.array(c["shape"]) - np.array(c["measured"])
    num("NCoShape", float(d.std()))
    f = np.array(c["f"])
    num("NCoPeak", float(f[int(np.argmax(c["measured"]))]), "%.0f")
    e = np.array(c["di"]) - np.array(c["di_test"])
    num("NCoDi", float(np.sqrt(np.mean(e ** 2))))
    num("NCoDiMax", float(np.abs(e).max()))
    P = np.array(list(c["power"].values()))
    num("NCoDirectLo", float((P[:, 0] - P[:, 1]).min()))
    num("NCoDirectHi", float((P[:, 0] - P[:, 1]).max()))
    num("NCoCohLo", float((P[:, 0] - P[:, 2]).min()), "%.0f")
    num("NCoCohHi", float((P[:, 0] - P[:, 2]).max()), "%.0f")
    num("NCoModLo", c["modern"][0], "%.0f")
    num("NCoModHi", c["modern"][1], "%.0f")
    cut = []
    for q in c["parts"]:
        tot, core_ = 10 ** (q["total"] / 10), 10 ** (q["layer/core"] / 10)
        cut.append(q["total"] - 10 * np.log10(tot - 0.9 * core_))
    num("NCoCutLo", min(cut))
    num("NCoCutHi", max(cut))
    j = c["jets"]
    k = [10 * np.log10(q["k"] / c["k_test"]) for q in j if q["v"] <= 250]
    num("NCoJetLo", min(k))
    num("NCoJetHi", max(k))
    num("NCoJetSt", float(np.mean([q["st"] for q in j])))


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


def roar():
    r = R("roar")
    f = np.array(r["fc"])
    low = f <= 130
    fl = np.array(r["floor"])[low]
    num("NRoFloorLo", float(fl.min()), "%.0f")
    num("NRoFloorHi", float(fl.max()), "%.0f")
    k = np.isin(np.round(f), (32, 40, 79))
    num("NRoFlat", float(np.abs(np.array(r["fast"]) - np.array(r["stall"]))[k].max()), "%.0f")
    o = (f >= 120) & (f <= 260)
    d = (np.array(r["flow"]) - np.array(r["model"]))[o]
    num("NRoOverLo", float(d.min()), "%.0f")
    num("NRoOverHi", float(d.max()), "%.0f")
    a = R("ask21")
    fc = np.array(a["fc"])
    s = (fc >= 120) & (fc <= 330)
    e = [float(np.sqrt(np.mean((np.array(q["rec"])[s] - np.array(q["mod"])[s]) ** 2))) for q in a["shapes"]]
    num("NRoFitLo", min(e))
    num("NRoFitHi", max(e))
    w = [q["rms_without"] for q in r["roll"]]
    v = [q["rms_with"] for q in r["roll"]]
    num("NRoRollWithoutLo", min(w), "%.0f")
    num("NRoRollWithoutHi", max(w), "%.0f")
    num("NRoRollWithLo", min(v), "%.0f")
    num("NRoRollWithHi", max(v), "%.0f")


def slip():
    s = R("slip")
    rows = s["rows"]
    at = lambda kt, b: next(r for r in rows if r["kt"] == kt and r["beta"] == b)
    k, full = s["full_kt"], s["full"]
    z, f = at(k, 0.0), at(k, full)
    num("NSlBalance", s["fin_balance"], "%.0f")
    num("NSlKt", k, "%.1f")
    num("NSlFull", full, "%.0f")
    num("NSlOnFront", s["onset"]["front"], "%.0f")
    num("NSlOnRear", s["onset"]["rear"], "%.0f")
    num("NSlWing", 100 * s["wing_windward"], "%.0f")
    num("NSlFront", f["front"] - z["front"], "%.0f")
    num("NSlFrontFast", at(90.0, full)["front"] - at(90.0, 0.0)["front"], "%.0f")
    num("NSlHalves", f["leeward"] - f["windward"], "%.0f")
    num("NSlShare", 100 * f["share"], "%.0f")
    num("NSlOA", f["front_oa"] - z["front_oa"], "%.0f")
    num("NSlHissFull", f["hiss"] - z["hiss"])
    num("NSlHissFin", at(k, 17.3)["hiss"] - z["hiss"])
    num("NSlFrontFin", at(k, 17.3)["front"] - z["front"])
    num("NSlSide", f["side"] - z["side"])
    num("NSlSideOA", f["side_oa"] - z["side_oa"], "%.0f")
    num("NSlBelow", f["below"] - z["below"], "%.0f")
    num("NSlBelowOA", f["below_oa"] - z["below_oa"], "%.0f")
    fc = np.array(s["fc"])
    b = (fc >= 24) & (fc <= 260)
    d = np.array(s["side_added"])[b] - np.array(s["brakes"])[b]
    num("NSlBrakeMean", float(d.mean()), "%+.0f")
    num("NSlBrakeSd", float(d.std()), "%.0f")
    num("NSlLift", s["lift_rms"], "%.2f")
    num("NSlPowWind", s["power"]["wind"])
    num("NSlPowAir", s["power"]["airframe"])
    num("NSlPowSlip", s["power"]["slip_wind"])


def props():
    p = R("props")
    sc, cr = p["schedule"], p["cruise"]
    err = lambda rows, a, b: [100.0 * (r[a] / r[b] - 1.0) for r in rows if abs(r[b]) > 10]
    num("NXpPitch", p["pitch"])
    pw = [100.0 * (r["model_kw"] / r["kw"] - 1.0) for r in sc]
    th = [100.0 * (r["model_lbf"] / r["lbf"] - 1.0) for r in sc]
    num("NXpPowerLo", min(pw), "%+.0f")
    num("NXpPowerHi", max(pw), "%+.0f")
    num("NXpThrustLo", min(th), "%+.0f")
    num("NXpThrustHi", max(th), "%+.0f")
    good = [r for r in cr if r["beta"] <= 22]
    tight = [100.0 * (r["model_lbf"] / r["lbf"] - 1.0) for r in good]
    num("NXpCruiseLo", min(tight), "%+.0f")
    num("NXpCruiseHi", max(tight), "%+.0f")
    num("NXpCruiseHigh", max(100.0 * (r["model_lbf"] / r["lbf"] - 1.0) for r in cr if r["beta"] > 22), "%+.0f")
    a = np.array(p["angle"])
    d = np.array(p["bpf"]) - np.array(p["lh"])
    plane = (a >= 85) & (a <= 115)
    wide = (a >= 57) & (a <= 120)
    num("NXpPlaneMean", float(d[plane].mean()), "%+.1f")
    num("NXpPlaneMax", float(np.abs(d[plane]).max()))
    num("NXpWideRms", float(np.sqrt(np.mean(d[wide] ** 2))))
    num("NXpBpf", float(p["bpf"][list(p["angle"]).index(95)]))
    num("NXpBpfLh", float(p["lh"][list(p["angle"]).index(95)]))
    num("NXpBpfRh", float(p["rh"][list(p["angle"]).index(95)]))
    num("NXpHarmTwo", p["harmonics"][1], "%.0f")
    num("NXpHarmThree", p["harmonics"][2], "%.0f")
    hz = np.array(p["test_hz"])
    e = np.array(p["model_db"]) - np.array(p["test_db"])
    hi, lo = hz >= 2000, hz <= 1000
    num("NXpHighLo", float(e[hi].min()), "%+.0f")
    num("NXpHighHi", float(e[hi].max()), "%+.0f")
    num("NXpLowLo", float(e[lo].min()), "%+.0f")
    num("NXpLowHi", float(e[lo].max()), "%+.0f")
    num("NXpMotorLo", min(p["under"]), "%.0f")
    num("NXpMotorHi", max(p["under"]), "%.0f")
    num("NXpLoops", p["loops"], "%.0f")
    num("NXpSources", p["sources"], "%.0f")
    num("NXpUpLo", min(p["upwash"]), "%.0f")
    num("NXpUpHi", max(p["upwash"]), "%.0f")


def tyres():
    t = R("tyres")
    o, a, p, x = t["outside"], t["ask21"], t["p2006t"], t["x57"]
    num("NTyRef", o["reference_1m"]["model"])
    num("NTyRefMeas", o["reference_1m"]["measured"])
    num("NTyCar", o["car_80kmh_7m5"]["model"])
    num("NTyTow", o["towed_aircraft_55mph_30m"]["model_smooth"])
    num("NTyTowAsph", o["towed_aircraft_55mph_30m"]["model_asphalt"])
    hz = np.array(a["hz"])
    d = np.abs(np.array(a["touchdown"]) - np.array(a["at_44kt"]))
    num("NTyAskAgree", float(d.mean()))
    num("NTyAskAgreeMax", float(d.max()), "%.0f")
    num("NTyHiStep", abs(a["hi_before_after_db"][1] - a["hi_before_after_db"][0]))
    r = a["roll_over_2_5khz_db_at_44_37_28_18kt"]
    num("NTyRatioSpread", max(r) - min(r))
    d = (np.array(a["module"]) - np.array(p["module"]))[hz > 240]
    num("NTyLessLo", float(d.min()), "%.0f")
    num("NTyLessHi", float(d.max()), "%.0f")
    num("NTyRiseLow", p["rise_vs_low_db"])
    num("NTyRiseHigh", p["rise_vs_high_db"])
    num("NTyThrottles", abs(p["throttles_db"]), "%.0f")
    num("NTyLine", abs(p["line_557hz_change_db"]))
    num("NTyLater", float(np.mean(np.array(p["later_three_wheels_54kt"]) - np.array(p["module"]))))
    num("NTySpeed", float(max(np.max(np.abs(np.array(p[k]) - np.array(p["half_58kt"]))) for k in ("half_55kt", "half_62kt"))))
    num("NTyApproach", p["cabin_dba"]["approach_all"], "%.0f")
    k = p["kurtosis_by_second_380_399"]
    steady = k[11:] + a["kurtosis_44_37_28kt"]
    num("NTyKurtLo", min(steady))
    num("NTyKurtHi", max(steady))
    num("NTyThump", max(k))
    num("NTyKurtLoop", t["loop_kurtosis"])
    at = lambda key, kt: x[key][x["kt"].index(kt)]
    for name, key in (("Cock", "cockpit"), ("CockAsk", "cockpit_ask21"), ("Side", "side"), ("Wind", "wind"), ("Air", "airborne")):
        num("NTy%sThirty" % name, at(key, 30), "%.0f")
        num("NTy%sSixty" % name, at(key, 60), "%.0f")
    num("NTyPow", x["low_speed_power"])
    rows = ["%.0f & %s & %s & %s" % (h, f(u, "%.1f"), f(v, "%.1f"), f(u - v, "%.1f")) for h, u, v in zip(hz, a["module"], p["module"])]
    write("tyres", tabular("rrrr", "Band, Hz & ASK~21 & P2006T & difference", rows))
    d = t["touchdown"]
    num("NTdEffP", d["p2006t"]["efficiency_db"])
    num("NTdEffA", d["ask21"]["efficiency_db"])
    num("NTdEffDiff", d["p2006t"]["efficiency_db"] - d["ask21"]["efficiency_db"])
    num("NTdSlide", d["p2006t"]["seconds"], "%.2f")
    num("NTdSink", d["p2006t"]["sink_fps"], "%.2f")
    num("NTdLost", d["p2006t"]["lost_j"] / 1e3)
    num("NTdToneP", 100 * d["tonal_share"][0], "%.0f")
    num("NTdToneA", 100 * d["tonal_share"][1], "%.0f")
    th = d["thump"]
    rises = [r for h, r in zip(th["hz"], th["rise_db"]) if 75 <= h <= 260 and r > 0]
    num("NTdRiseLo", min(rises), "%.0f")
    num("NTdRiseHi", max(rises), "%.0f")
    x = d["x57"]
    num("NTdSlideSoft", 1e3 * x["main"]["slide_s"][0], "%.0f")
    num("NTdSlideFirm", 1e3 * x["main"]["slide_s"][1], "%.0f")
    num("NTdSqCock", x["main"]["cockpit_dba"], "%.0f")
    num("NTdSqSide", x["main"]["side_dba"], "%.0f")
    num("NTdSqNose", x["nose"]["cockpit_dba"], "%.0f")
    num("NTdThump", x["main"]["thump_1fps_dba"], "%.0f")


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
    for fn in (verify, rotors, stand, hart2, bvi, texture, fountain, outside, engines, cabin, cockpit, flight, prue, core, mv22, ask21, roar, slip, props, tyres, bench, before_after):
        try:
            fn()
        except (FileNotFoundError, KeyError) as e:
            print("skipped", fn.__name__, repr(e))
    with open(os.path.join(TAB, "numbers.tex"), "w") as out:
        for k in sorted(NUM):
            out.write("\\newcommand{\\%s}{%s}\n" % (k, NUM[k]))
    print(len(NUM), "numbers")
