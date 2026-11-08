"""Figures for the report (vector PDF) from report/results/*.json.  python report/figures.py [name ...]"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refdata as D

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.0, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "grid.linewidth": 0.3, "grid.alpha": 0.5,
                     "axes.grid": True, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
C_MOD, C_TEST, C_REF, C_OLD = "#b2182b", "#222222", "#7a7a7a", "#2166ac"
W1, W2 = 3.35, 6.9
OUT = os.path.join(HERE, "fig")


def res(name):
    return json.load(open(os.path.join(HERE, "results", name + ".json")))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def tag(ax, s):
    ax.set_title("(%s)" % s, loc="left", fontsize=8.5, pad=3)


def pts(ax, x, y, label="test", color=C_TEST, marker="o"):
    ax.plot(x, y, marker, mfc="none", mec=color, ms=3.4, mew=0.8, ls="none", label=label)


def verify():
    v = res("verify")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.0))
    ax = axs[0]
    f = np.array(v["bpm_f"])
    t, p = np.array(v["bpm_d2_table"]), np.array(v["bpm_d2_port"])
    for k, (lab, c) in enumerate((("pressure side", "#2166ac"), ("suction side", "#1b7837"), ("laminar shedding", "#ef8a62"), ("total", C_MOD))):
        ax.semilogx(f, p[:, k], "-", color=c, label=lab)
        ax.plot(f, t[:, k], "o", mfc="none", mec=C_TEST, ms=2.2, mew=0.5, ls="none", label="RP-1218 table D2" if k == 3 else None)
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("SPL (1/3 octave), dB")
    ax.set_ylim(0, 70)
    ax.legend(ncol=2, loc="lower right")
    tag(ax, "a")
    ax = axs[1]
    g = v["gutin"]
    for m, mk in ((1, "o"), (2, "s"), (3, "^")):
        x = [a["ref"] for a in g if a["m"] == m]
        y = [a["got"] for a in g if a["m"] == m]
        ax.plot(x, y, mk, mfc="none", mec=C_MOD, ms=3.6, mew=0.7, ls="none", label="harmonic %d" % m)
    lo, hi = min(a["ref"] for a in g) - 3, max(a["ref"] for a in g) + 3
    ax.plot([lo, hi], [lo, hi], "-", color=C_REF, lw=0.7)
    ax.set_xlabel("Gutin's closed form, dB")
    ax.set_ylabel("Formulation 1A, dB")
    ax.legend()
    tag(ax, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, "verify")


def rotors():
    r = res("rotors")
    b = res("before_rotors")
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.95))
    ax = axs[0]
    ax.plot(r["xv15_test"]["ct_s"], r["xv15_test"]["cp_s"], "-", color=C_TEST, lw=0.9, label="test (TM-86833)")
    x = [a["ct_s"] for a in r["xv15"]]
    ax.plot(x, [a["cp_s"] for a in r["xv15"]], "o-", color=C_MOD, ms=2.2, label="model")
    xo = [a["ct_s"] for a in b["xv15"]]
    yo = [a["cp_s"] for a in b["xv15"]]
    ax.plot(xo, yo, "x", color=C_OLD, ms=3.0, mew=0.7, label="before (generic section)")
    ax.set_ylim(0, 0.027)
    ax.set_xlim(0, 0.19)
    ax.set_xlabel(r"$C_T/\sigma$")
    ax.set_ylabel(r"$C_P/\sigma$")
    ax.legend(loc="upper left")
    tag(ax, "a  XV-15, hover")
    ax = axs[1]
    pts(ax, [a["ct_s"] for a in r["bo105"]], [a["cp_s_test"] for a in r["bo105"]], "test (TM-85892)")
    ax.plot([a["ct_s"] for a in r["bo105"]], [a["cp_s"] for a in r["bo105"]], "-", color=C_MOD, label="model")
    ax.set_xlabel(r"$C_T/\sigma$")
    ax.set_ylabel(r"$C_P/\sigma$")
    ax.legend(loc="upper left")
    tag(ax, "b  Bo 105, hover")
    ax = axs[2]
    ff = r["bo105_forward"]
    pts(ax, [a["mu"] for a in ff], [a["cp_s_test"] for a in ff], "test (TM-110356)")
    ax.plot([a["mu"] for a in ff], [a["cp_s"] for a in ff], "-", color=C_MOD, label="model")
    ax.set_xlabel(r"advance ratio $\mu$")
    ax.set_ylabel(r"$C_P/\sigma$")
    ax.legend(loc="upper left")
    tag(ax, "c  Bo 105, forward")
    ax = axs[3]
    for k, c in (("65", "#2166ac"), ("80", C_MOD)):
        rows = r["uh60_level"][k]
        pts(ax, [a["mu"] for a in rows], [a["cp_test"] for a in rows], r"test $C_T$ %s" % ("0.0065" if k == "65" else "0.0080"), color=c)
        ax.plot([a["mu"] for a in rows], [a["cp"] for a in rows], "-", color=c, label=None)
    ax.set_xlabel(r"advance ratio $\mu$")
    ax.set_ylabel(r"$10^5\,C_P$")
    ax.legend(loc="upper left")
    tag(ax, "d  UH-60A, level")
    fig.tight_layout(w_pad=0.4)
    save(fig, "rotors")


def torsion():
    r = res("rotors")
    b = res("before_rotors")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.95))
    ax = axs[0]
    t = np.array(D.BO105_TORSION_034)
    pts(ax, t[:, 0], t[:, 1], "test (TM-103867)")
    ax.plot([a["ct_s"] for a in r["bo105"]], [a["torsion_034"] for a in r["bo105"]], "-", color=C_MOD, label="model")
    ax.plot([a["ct_s"] for a in b["bo105"]], [a["torsion_034"] for a in b["bo105"]], "--", color=C_OLD, label="its moment, before")
    ax.set_xlabel(r"$C_T/\sigma$")
    ax.set_ylabel("torsion moment at 0.34 R, in lb")
    ax.legend()
    tag(ax, "a")
    ax = axs[1]
    ax.plot([a["ct_s"] for a in r["bo105"]], [a["twist_tip"] for a in r["bo105"]], "-", color=C_MOD, label="model")
    ax.plot([a["ct_s"] for a in b["bo105"]], [a["twist_tip"] for a in b["bo105"]], "--", color=C_OLD, label="before")
    ax.axhline(0, color=C_REF, lw=0.5)
    ax.set_xlabel(r"$C_T/\sigma$")
    ax.set_ylabel("elastic twist at the tip, deg")
    ax.legend()
    tag(ax, "b")
    ax = axs[2]
    pts(ax, [a["ct_s"] for a in r["bo105"]], [a["theta_test"] for a in r["bo105"]], "test")
    ax.plot([a["ct_s"] for a in r["bo105"]], [a["theta"] for a in r["bo105"]], "-", color=C_MOD, label="model")
    ax.plot([a["ct_s"] for a in b["bo105"]], [a["theta"] for a in b["bo105"]], "--", color=C_OLD, label="before")
    ax.set_xlabel(r"$C_T/\sigma$")
    ax.set_ylabel(r"collective at 0.7 R, deg")
    ax.legend()
    tag(ax, "c")
    fig.tight_layout(w_pad=0.8)
    save(fig, "torsion")


def stand():
    s = res("stand")
    b = res("before_stand")
    t = s["test"]
    x = [a["lb"] / 1000 for a in s["rows"]]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.25))
    for ax, key, tt, tv, lab in ((axs[0], "near", t["thrust_lb"], t["near_db"], "a  near field: 2 ft out of the tip, 8 ft under the disc"),
                                 (axs[1], "far", t["far_thrust_lb"], t["far_db"], "b  far field: 250 ft, 15 deg below the disc")):
        ax.plot(np.array(tt) / 1000, tv, "o", mfc="none", mec=C_TEST, ms=3.2, mew=0.8, ls="none", label="test (CR-177436)")
        ax.plot(x, [a[key]["total"] for a in s["rows"]], "-", color=C_MOD, lw=1.3, label="model")
        ax.plot(x, [a[key]["thickness"] for a in s["rows"]], ":", color="#2166ac", label="thickness")
        ax.plot(x, [a[key]["loading"] for a in s["rows"]], "--", color="#1b7837", label="loading")
        ax.plot(x, [a[key]["broadband"] for a in s["rows"]], "-.", color="#ef8a62", label="broadband")
        ax.plot([a["lb"] / 1000 for a in b["rows"]], [a[key]["total"] for a in b["rows"]], "x", color=C_OLD, ms=3, mew=0.6, label="before")
        ax.set_xlabel("rotor thrust, 1000 lb")
        ax.set_ylabel("overall level, dB")
        ax.set_ylim(102 if key == "near" else 74, 136 if key == "near" else 110)
        tag(ax, lab)
    axs[0].legend(ncol=2, loc="upper left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "stand")


def _final(h):
    return next(c for c in h["cases"] if c.get("final")) if any(c.get("final") for c in h["cases"]) else h["cases"][0]


def hart2_loads():
    h = res("hart2")
    b = res("before_hart2")
    c = h["cases"][0]
    o = b["cases"][0]
    fig, ax = plt.subplots(figsize=(W1, 1.95))
    ax.plot(h["loads_psi"], h["loads_measured"], "-", color=C_TEST, lw=1.4, label="HART II, measured")
    ax.plot(c["psi"], c["m2cn"], "-", color=C_MOD, lw=0.9, label="model")
    ax.plot(o["psi"], o["m2cn"], "-", color=C_OLD, lw=0.5, label="before")
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel(r"azimuth $\psi$, deg")
    ax.set_ylabel(r"$M^2 c_n$ at $r/R=0.87$")
    ax.legend(ncol=3, loc="lower center")
    ax.set_ylim(-0.02, 0.17)
    save(fig, "hart2_loads")


def hart2_map():
    h = res("hart2")
    b = res("before_hart2")
    y = np.array(h["y"])
    x = np.array(h["minus_x"])
    G = np.array(h["measured"])
    c = h["cases"][0]
    L = np.array(c["L"])
    o = np.array(b["cases"][0]["L"])
    lv = np.arange(88, 125, 2)
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.6), sharey=True, gridspec_kw=dict(width_ratios=(1, 1, 1, 1)))
    for ax, Z, t in ((axs[0], G, "a  measured"), (axs[1], L, "b  model"), (axs[2], o, "c  before")):
        cs = ax.contourf(y, x, Z, levels=lv, cmap="jet", extend="both")
        ax.contour(y, x, Z, levels=lv[::2], colors="k", linewidths=0.2)
        tag(ax, t)
    ax = axs[3]
    d = L - G
    cd = ax.contourf(y, x, d, levels=np.arange(-14, 15, 2), cmap="RdBu_r", extend="both")
    tag(ax, "d  b less a")
    for ax in axs:
        ax.add_patch(plt.Circle((0, 0), 2.0, fill=False, lw=0.5, color="k"))
        ax.set_aspect("equal")
        ax.set_xlabel("y, m (advancing +)")
        ax.grid(False)
        ax.set_xlim(-2.7, 2.7)
        ax.set_ylim(4, -4)
    axs[0].set_ylabel("x, m (downstream +)")
    fig.tight_layout(w_pad=0.3)
    fig.colorbar(cs, ax=axs[:3], fraction=0.018, pad=0.01, label="dB")
    fig.colorbar(cd, ax=axs[3], fraction=0.06, pad=0.02, label="dB")
    save(fig, "hart2_map")


def bvi():
    r = res("bvi")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 1.95))
    ax = axs[0]
    a = r["alpha"]
    ax.plot([q["alpha"] for q in a], [q["peak"] for q in a], "o-", color=C_MOD, ms=2.5, label="peak on the plane")
    ax.plot([q["alpha"] for q in a], [q["peak_no_bvi"] for q in a], "--", color=C_REF, label="without the wake's impulses")
    ax.axvline(5.3, color=C_TEST, lw=0.6, ls=":")
    ax.text(5.5, 118, "test's maximum", fontsize=7)
    ax.set_xlabel("shaft angle, deg aft")
    ax.set_ylabel("6--40 BPF level, dB")
    ax.set_ylim(55, 124)
    ax.legend(loc="lower left")
    tag(ax, "a  HART II")
    ax = axs[1]
    for nm, c, lab in (("uh60a", C_MOD, "UH-60A, 65 kt"), ("uh1h", "#2166ac", "UH-1H, 70 kt")):
        rows = [q for q in r["descent"] if q["aircraft"] == nm]
        ax.plot([-q["fpm"] for q in rows], [q["slap"] for q in rows], "o-", color=c, ms=2.5, label=lab + ": slap layer")
        ax.plot([-q["fpm"] for q in rows], [q["rotor"] for q in rows], "--", color=c, lw=0.7, label="the rest of its rotor")
    ax.set_xlabel("rate of descent, ft/min")
    ax.set_ylabel("dBA at 100 m, ahead and right")
    ax.set_ylim(28, 100)
    ax.legend(ncol=2, loc="lower center")
    tag(ax, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, "bvi")


def texture():
    r = res("texture")
    m = [q for q in r["model"] if q["aircraft"] == "uh60a" and q["tag"] != "cruise_ahead"]
    fig, axs = plt.subplots(3, 1, figsize=(W1, 2.3), sharex=True)
    for ax, q in zip(axs, m):
        y = np.array(q["wave"])
        ax.plot(np.arange(len(y)) / q["fs"], y, "-", color=C_MOD, lw=0.25)
        ax.set_ylabel("Pa")
        ax.text(0.99, 0.9, "%s, %.0f dBA" % (q["tag"], q["dba"]), transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.grid(False)
    axs[-1].set_xlabel("time, s")
    fig.tight_layout(h_pad=0.2)
    save(fig, "texture")


def fountain():
    r = res("fountain")
    rows = r["rows"]
    fs = r["fs"]
    fig, axs = plt.subplots(2, 2, figsize=(W2, 2.2), sharex=True, sharey=True)
    for i, az in enumerate((180, 0)):
        for j, on in enumerate((True, False)):
            q = next(a for a in rows if a["az"] == az and abs(a["el"] + 45.7) < 0.1 and a["on"] == on)
            y = np.array(q["wave"])
            ax = axs[i, j]
            ax.plot(np.arange(len(y)) / fs, y, "-", color=C_MOD if on else C_OLD, lw=0.4)
            ax.text(0.99, 0.92, "%s, fountain %s: %.1f dBA" % ("behind" if az == 180 else "ahead", "on" if on else "off", q["dba"]),
                    transform=ax.transAxes, ha="right", va="top", fontsize=7)
            ax.grid(False)
    axs[0, 0].set_ylabel("Pa")
    axs[1, 0].set_ylabel("Pa")
    axs[1, 0].set_xlabel("time, s")
    axs[1, 1].set_xlabel("time, s")
    fig.tight_layout(w_pad=0.5, h_pad=0.2)
    save(fig, "fountain")


def outside():
    r = res("outside")
    h = r["hover"]
    fig, ax = plt.subplots(figsize=(W1, 2.05))
    d = sorted(D.MV22_HOVER_DBA)
    ax.semilogx(d, [D.MV22_HOVER_DBA[k] for k in d], "o", mfc="none", mec=C_TEST, ms=4, mew=0.9, ls="none", label="published maximum")
    for sel, lab, c, ls in ((lambda q: q["az"] == 180 and q["on"] and not q["low"], "behind, 30 deg below", C_MOD, "-"),
                            (lambda q: q["az"] == 90 and q["on"] and not q["low"], "to the side", "#ef8a62", "-"),
                            (lambda q: q["az"] == 180 and not q["on"], "behind, no fountain", C_OLD, "--"),
                            (lambda q: q["low"], "hub 30 m up, listener on the ground", C_REF, ":")):
        rows = sorted((q for q in h if sel(q)), key=lambda q: q["slant"])
        ax.semilogx([q["slant"] for q in rows], [q["fast"] for q in rows], ls, color=c, label=lab)
    ax.set_xlabel("distance, m")
    ax.set_ylabel("maximum, fast, dBA")
    ax.set_xticks(d)
    ax.set_xticklabels([str(int(k)) for k in d])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="lower left")
    save(fig, "outside")


def engines():
    e = res("engines")
    t = res("tones")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.95))
    ax = axs[0]
    c = e["t700"]
    pts(ax, [a["ng"] for a in c], [a["pr_test"] for a in c], "GE (TM-105805)")
    ax.plot([a["ng"] for a in c], [a["pr"] for a in c], "-", color=C_MOD, label="model")
    ax.set_xlabel(r"$N_g$, %")
    ax.set_ylabel("delivery pressure ratio")
    ax.legend()
    tag(ax, "a")
    ax = axs[1]
    pts(ax, [a["ng"] for a in c], [a["t45_test"] for a in c], "GE")
    ax.plot([a["ng"] for a in c], [a["t45"] for a in c], "-", color=C_MOD, label="model")
    ax.set_xlabel(r"$N_g$, %")
    ax.set_ylabel("power turbine inlet, K")
    tag(ax, "b")
    ax = axs[2]
    tr = t["doors"]["track"]
    tt = np.array([q["t"] + 4 for q in tr])
    for k, mk in ((0, "o"), (1, "s")):
        ax.plot(tt, [q["lines"][k][0] / 1000 for q in tr], mk, mfc="none", mec=C_TEST, ms=3, mew=0.7, ls="none")
    ax.axhspan(10.045, 10.075, color=C_OLD, alpha=0.3, lw=0)
    ax.axvline(420, color=C_REF, lw=0.6, ls=":")
    ax.set_xlabel("recording time, s")
    ax.set_ylabel("the whine's lines, kHz")
    tag(ax, "c")
    fig.tight_layout(w_pad=0.8)
    save(fig, "engines")


def cabin():
    c = res("cabin")
    f = np.array(c["f"])
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=False)
    ax = axs[0]
    for nm, col, lab in (("uh60a", "#2166ac", "UH-60A"), ("mv22b", C_MOD, "MV-22B")):
        d = c[nm]
        ax.semilogx(f, d["opaque"], "-", color=col, label=lab + " wall")
        ax.semilogx(f, d["window"], "--", color=col, lw=0.8, label=lab + " window")
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("transmission loss, dB")
    ax.set_xlim(31, 16000)
    ax.legend(ncol=2, loc="upper left")
    tag(ax, "a")
    ax = axs[1]
    for nm, col, lab in (("uh60a", "#2166ac", "UH-60A, shut"), ("mv22b", C_MOD, "MV-22B")):
        ax.semilogx(f, c[nm]["inside"], "-", color=col, label=lab)
    ax.semilogx(f, c["uh60a"]["inside_open"], "--", color="#2166ac", lw=0.8, label="UH-60A, all open")
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("inside over outside, dB")
    ax.set_xlim(31, 16000)
    ax.legend()
    tag(ax, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, "cabin")


def wind():
    t = res("wind")["tu144"]
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    ax.semilogx(t["f"], t["test"], "o", mfc="none", mec=C_TEST, ms=3.4, mew=0.8, ls="none", label="Tu-144LL in flight, M 0.74")
    ax.semilogx(t["f"], t["model"], "-", color=C_MOD, label="model")
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("wall pressure, dB in 1 Hz")
    ax.legend()
    save(fig, "wind")


def cockpit():
    c = res("cockpit")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.05))
    ax = axs[0]
    fc = c["fc"]
    r = next(q for q in c["rows"] if q["kt"] == 120)
    ax.semilogx(fc, c["pilot"]["bands"], "o-", color=C_TEST, ms=3, lw=0.8, label="pilot (Mace et al.)")
    ax.semilogx(fc, c["copilot"]["bands"], "s:", mfc="none", color=C_REF, ms=3, lw=0.7, label="copilot")
    ax.semilogx(fc, r["octaves"], "-", color=C_MOD, lw=1.3, label="model")
    for k, col in (("rotors", "#2166ac"), ("drive", "#ef8a62"), ("wind", "#1b7837")):
        ax.semilogx(fc, r["part_octaves"][k], "--", color=col, lw=0.7, label=k)
    ax.set_ylim(60, 122)
    ax.set_xlabel("octave band, Hz")
    ax.set_ylabel("level, dB")
    ax.legend(ncol=2, loc="lower left")
    tag(ax, "a")
    ax = axs[1]
    kts = [q["kt"] for q in c["rows"]]
    ax.plot(kts, [q["dba"] for q in c["rows"]], "-", color=C_MOD, label="model")
    ax.plot(kts, [c["nasa"][str(k)] for k in kts], "o", mfc="none", mec=C_TEST, ms=3.6, mew=0.8, ls="none", label="NASA TM-84664")
    ax.plot([120, 120], [c["copilot"]["dba"], c["pilot"]["dba"]], "s", mfc="none", mec=C_OLD, ms=3.6, mew=0.8, ls="none", label="Mace et al.")
    ax.axhline(c["army"]["uh60a_pilot"], color=C_REF, lw=0.7, ls=":", label="US Army")
    ax.set_xlabel("airspeed, kt")
    ax.set_ylabel("level at the pilots, dBA")
    ax.legend(ncol=2, loc="lower right")
    ax.set_ylim(86, 108)
    tag(ax, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, "cockpit")


def flight():
    f = res("flight")["rows"]
    p = res("prue")["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.05))
    ax = axs[0]
    lv = [q for q in f if q["fpm"] == 0]
    ap = [q for q in f if q["fpm"] != 0]
    ax.plot([q["test_dba"] for q in lv], [q["dba"] for q in lv], "o", mfc="none", mec=C_MOD, ms=4, mew=0.9, ls="none", label="level flyovers")
    ax.plot([q["test_dba"] for q in ap], [q["dba"] for q in ap], "^", mfc="none", mec=C_OLD, ms=4.5, mew=0.9, ls="none", label="6 deg approach")
    ax.plot([76, 94], [76, 94], "-", color=C_REF, lw=0.7)
    ax.set_xlabel("measured maximum, dBA")
    ax.set_ylabel("model, dBA")
    ax.legend(loc="upper left")
    tag(ax, "a")
    ax = axs[1]
    for q, c in zip(p, ("#2166ac", "#1b7837", C_MOD)):
        ax.semilogx(q["f"], q["model"], "-", color=c, lw=0.9, label="%.0f kt" % q["kt"])
        ax.semilogx(q["f"], q["total"], "--", color=c, lw=0.7)
        ax.plot(q["test_f"], q["test"], "o", mfc="none", mec=c, ms=3.4, mew=0.8, ls="none")
    ax.set_xlim(150, 7000)
    ax.set_xticks([200, 500, 1000, 2000, 5000], ["200", "500", "1000", "2000", "5000"])
    ax.set_xticks([], minor=True)
    ax.set_ylim(0, 45)
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("third octave, dB")
    ax.legend(loc="lower left")
    tag(ax, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, "flight")


def combustion():
    c = res("core")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.05))
    ax = axs[0]
    f = np.array(c["f"])
    m = np.array(c["measured"])
    s = np.array(c["shape"])
    ax.semilogx(f, m, "o", mfc="none", mec=C_TEST, ms=3.4, mew=0.8, ls="none", label="YF-102, 95 % speed")
    ax.semilogx(f, s + (m - s).mean(), "-", color=C_MOD, lw=0.9, label="GE's spectrum, its level matched")
    ax.set_xticks([50, 100, 200, 500, 1000], ["50", "100", "200", "500", "1000"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("third octave, dB")
    ax.legend(loc="lower right")
    tag(ax, "a")
    ax = axs[1]
    ax.plot(c["th"], c["di_test"], "o", mfc="none", mec=C_TEST, ms=3.4, mew=0.8, ls="none", label="YF-102")
    ax.plot(c["th"], c["di"], "-", color=C_MOD, lw=0.9, label="GE's directivity")
    ax.set_xlabel("angle from the inlet, deg")
    ax.set_ylabel("directivity index, dB")
    ax.legend(loc="lower center")
    tag(ax, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, "combustion")


def slip():
    s = res("slip")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.05))
    f = np.array(s["fc"])
    k = (f >= 24) & (f <= 8000)
    tick = lambda ax: (ax.set_xticks([30, 100, 300, 1000, 3000], ["30", "100", "300", "1k", "3k"]), ax.set_xticks([], minor=True))
    ax = axs[0]
    ax.semilogx(f[k], np.array(s["straight_half"])[k], "-", color=C_REF, lw=0.9, label="straight, each half")
    ax.semilogx(f[k], np.array(s["windward"])[k], "-", color=C_OLD, lw=0.9, label="windward half")
    ax.semilogx(f[k], np.array(s["leeward"])[k], "-", color=C_MOD, lw=0.9, label="leeward half")
    tick(ax)
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("third octave, dB")
    ax.legend(loc="upper right")
    tag(ax, "a")
    ax = axs[1]
    r = [q for q in s["rows"] if q["kt"] == s["full_kt"]]
    b = [q["beta"] for q in r]
    ax.plot(b, [q["front"] for q in r], "-", color=C_MOD, lw=0.9, label="front seat")
    ax.plot(b, [q["rear"] for q in r], "--", color=C_MOD, lw=0.9, label="rear seat")
    ax.plot(b, [q["windward"] for q in r], "-", color=C_OLD, lw=0.9, label="front, windward half")
    ax.set_xlabel("sideslip, deg")
    ax.set_ylabel("dB(A)")
    ax.legend(loc="upper left")
    tag(ax, "b")
    ax = axs[2]
    ax.semilogx(f[k], np.array(s["side_straight"])[k], "-", color=C_REF, lw=0.9, label="straight")
    ax.semilogx(f[k], np.array(s["side_slip"])[k], "-", color=C_MOD, lw=0.9, label="slip")
    ax.semilogx(f[k], np.array(s["brakes"])[k], "--", color=C_TEST, lw=0.9, label="the brakes alone")
    tick(ax)
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("third octave, dB")
    ax.legend(loc="lower left")
    tag(ax, "c")
    fig.tight_layout(w_pad=1.0)
    save(fig, "slip")


def props():
    p = res("props")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.05))
    ax = axs[0]
    ax.plot(p["angle"], p["lh"], "o", mfc="none", mec=C_TEST, ms=3.2, mew=0.8, label="left-handed blades")
    ax.plot(p["angle"], p["rh"], "s", mfc="none", mec=C_REF, ms=2.8, mew=0.7, label="right-handed")
    ax.plot(p["angle"], p["bpf"], "-", color=C_MOD, lw=0.9, label="model")
    ax.set_ylim(40, 90)
    ax.set_xlabel("angle from upstream, deg")
    ax.set_ylabel("blade passing tone, dB")
    ax.legend(loc="lower center", ncol=1)
    tag(ax, "a")
    ax = axs[1]
    f = np.array(p["f"])
    k = (f >= 300) & (f <= 20000)
    ax.semilogx(p["test_hz"], p["test_db"], "o", mfc="none", mec=C_TEST, ms=3.2, mew=0.8, label="tunnel, between the tones")
    ax.semilogx(p["test_hz"], p["background_db"], "x", color=C_REF, ms=3.2, mew=0.7, label="empty tunnel")
    ax.semilogx(f[k], np.array(p["per10"])[k], "-", color=C_MOD, lw=0.9, label="model")
    ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000], ["500", "1k", "2k", "5k", "10k", "20k"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("level in 10 Hz, dB")
    ax.legend(loc="lower left")
    tag(ax, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, "props")


def mv22():
    m = res("mv22")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.05))
    ax = axs[0]
    ax.semilogx(m["fc"], m["recording"], "o-", color=C_TEST, ms=2, lw=0.8, label="recording")
    ax.semilogx(m["fc"], m["model"], "-", color=C_MOD, label="model")
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("third octave, dB (equal loudness)")
    ax.legend()
    tag(ax, "a")
    ax = axs[1]
    ax.plot(m["harmonics"], m["over_rec"], "o-", color=C_TEST, ms=2, lw=0.7, label="recording")
    ax.plot(m["harmonics"], m["over_mod"], "-", color=C_MOD, label="model")
    ax.set_xlabel("harmonic of the blade passage")
    ax.set_ylabel("over the floor beside it, dB")
    ax.legend()
    tag(ax, "b")
    ax = axs[2]
    n = 8
    lab = ["%d" % r["lo"] for r in m["stats_rec"][:n]]
    x = np.arange(n)
    ax.bar(x - 0.2, [r["pulse"] for r in m["stats_rec"][:n]], 0.4, color=C_TEST, label="recording")
    ax.bar(x + 0.2, [r["pulse"] for r in m["stats_mod"][:n]], 0.4, color=C_MOD, label="model")
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=60, fontsize=6.5)
    ax.set_xlabel("band's lower edge, Hz")
    ax.set_ylabel("pulsing per blade pass, dB")
    ax.legend()
    tag(ax, "c")
    fig.tight_layout(w_pad=0.6)
    save(fig, "mv22")


def ask21():
    a = res("ask21")
    fc = a["fc"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.05))
    ax = axs[0]
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(a["shapes"])))
    for s, c in zip(a["shapes"], cols):
        ax.semilogx(fc, s["rec"], "-", color=c, lw=0.9, label="%.0f kt" % s["kt"])
        ax.semilogx(fc, s["mod"], "--", color=c, lw=0.6)
    ax.set_xlim(200, 12500)
    ax.set_ylim(-30, 25)
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("dB over the 1--4 kHz mean")
    ax.legend(ncol=2, loc="lower left", fontsize=6)
    tag(ax, "a  first glider")
    ax = axs[1]
    ax.semilogx(fc, a["second"]["rec"], "-", color=C_TEST, label="recording")
    ax.semilogx(fc, a["second"]["mod"], "--", color=C_MOD, label="model, 80 kt")
    ax.set_xlim(200, 12500)
    ax.set_ylim(-30, 25)
    ax.set_xlabel("frequency, Hz")
    ax.legend(loc="lower left")
    tag(ax, "b  second glider")
    ax = axs[2]
    r = a["roll"]
    t = np.array(r["t"])
    g = np.array(r["gap"], bool)
    ax.plot(t, np.where(g, np.nan, r["rec"]), "-", color=C_TEST, label="recording")
    ax.plot(t, np.where(g, np.nan, r["mod"]), "--", color=C_MOD, label="model")
    ax.set_xlabel("time, s")
    ax.set_ylabel("1.6--6.3 kHz, dB")
    ax.legend(loc="lower left")
    tag(ax, "c  a roll")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ask21")


def tyres():
    t = res("tyres")
    a, p, x = t["ask21"], t["p2006t"], t["x57"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.05))
    ax = axs[0]
    hz = np.array(a["hz"])
    ax.semilogx(hz, a["module"], "-", color=C_REF, lw=0.9, label="ASK 21")
    ax.semilogx(hz, a["touchdown"], "o", mfc="none", mec=C_REF, ms=2.6, mew=0.6, label="its two calibrations")
    ax.semilogx(hz, a["at_44kt"], "o", mfc="none", mec=C_REF, ms=2.6, mew=0.6)
    pm = np.array(p["module"])
    ax.fill_between(hz, pm - 3, pm + 3, color=C_MOD, alpha=0.15, lw=0, label="a quarter to all")
    ax.semilogx(hz, pm, "-", color=C_MOD, lw=1.0, label="P2006T, half")
    ax.set_xticks([125, 250, 500, 1000, 2000], ["125", "250", "500", "1k", "2k"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("cockpit less tyre power, dB")
    ax.legend(loc="lower left")
    tag(ax, "a")
    ax = axs[1]
    tt = np.array(p["balance_t"]) - p["touchdown_s"]
    k = tt >= -7.0
    ax.step(tt[k], np.array(p["balance_db"])[k], where="post", color=C_TEST, lw=0.9)
    ax.axvline(0.0, color=C_MOD, lw=0.8, ls="--")
    ax.text(0.3, 0.92, "touchdown", color=C_MOD, fontsize=7, transform=ax.get_xaxis_transform())
    ax.set_xlabel("time from touchdown, s")
    ax.set_ylabel("150--1700 Hz against the rest, dB")
    tag(ax, "b  P2006T")
    ax = axs[2]
    kt = x["kt"]
    ax.plot(kt, x["cockpit"], "-", color=C_MOD, label="cockpit, P2006T's")
    ax.plot(kt, x["cockpit_ask21"], "--", color=C_REF, label="with the ASK 21's")
    ax.plot(kt, x["wind"], ":", color=C_TEST, label="cockpit wind")
    ax.plot(kt, x["side"], "-", color=C_OLD, label="30 m to the side")
    ax.set_ylim(0, 105)
    ax.set_xlabel("ground speed, kt")
    ax.set_ylabel("three tyres, dB(A)")
    ax.legend(loc="lower right")
    tag(ax, "c  X-57")
    fig.tight_layout(w_pad=0.8)
    save(fig, "tyres")


ALL = dict(verify=verify, rotors=rotors, torsion=torsion, stand=stand, hart2_loads=hart2_loads, hart2_map=hart2_map, bvi=bvi, texture=texture, fountain=fountain,
           outside=outside, engines=engines, cabin=cabin, wind=wind, cockpit=cockpit, flight=flight, combustion=combustion, mv22=mv22, ask21=ask21, slip=slip, props=props, tyres=tyres)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n in sys.argv[1:] or ALL:
        ALL[n]()
