"""Collect measured numbers (outputs/eval/*.json, outputs/pretrain/rig_eval.json, checkpoints/*/) into
docs/RESULTS_TABLES.md so that the documentation quotes exactly what was measured."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(x):
    return f"{100 * x:.0f} %"


def ci(c):
    return f"{100 * c[0]:.0f}-{100 * c[1]:.0f} %"


def main():
    out = ["# Tabel hasil terukur (dibuat otomatis oleh scripts/make_results.py)", ""]
    rig = ROOT / "outputs/pretrain/rig_eval.json"
    if rig.exists():
        s = json.loads(rig.read_text())["summary"]
        out += ["## Pretraining: rig kamera Stargaze", "",
                f"{s['plants']} tanaman (level {s['level']}) x 12 pose kamera = {s['n_views']} pandangan, noise kamera {'aktif' if s['noise'] else 'mati'}.", "",
                "| Metrik | Mean | Median | p90 | Maks |", "|---|---|---|---|---|"]
        for k, lab, sc, u in (("fruit_center_error_m", "Galat pusat buah", 100, "cm"), ("radius_error_m", "Galat radius buah", 100, "cm"),
                              ("reprojection_px", "Galat reproyeksi pusat", 1, "px"),
                              ("fruit_surface_mean_abs_mm", "Galat titik permukaan buah", 1, "mm"),
                              ("line_clearance_perceived_m", "Clearance garis T1-T2 (daun terlihat)", 100, "cm"),
                              ("line_clearance_true_leaves_m", "Clearance garis T1-T2 (daun sebenarnya)", 100, "cm"),
                              ("compute_ms", "Waktu komputasi", 1, "ms")):
            v = s.get(k)
            if v:
                out.append(f"| {lab} ({u}) | {v['mean'] * sc:.1f} | {v['median'] * sc:.1f} | {v['p90'] * sc:.1f} | {v['max'] * sc:.1f} |")
        out += ["", f"Buah terdeteksi: {pct(s['fruit_detected'])}; garis Stargaze ditemukan: {pct(s['line_found'])}; "
                f"garis yang menembus daun sebenarnya: {pct(s['negative_true_clearance_frac'] or 0)}; face terpilih: {s['faces']}.", ""]
    evs = sorted((ROOT / "outputs/eval").glob("*.json"))
    rows = []
    for f in evs:
        if any(t in f.stem for t in ("_a", "_b", "dev")):
            continue
        d = json.loads(f.read_text())["summary"]
        rows.append((f.stem, d))
    if rows:
        out += ["## Misi (episode lengkap, seed held-out)", "",
                "| File | Misi | Policy | n | Sukses episode | CI 95 % | Buah di keranjang | Galat estimasi buah (median) | Kegagalan |",
                "|---|---|---|---|---|---|---|---|---|"]
        for name, d in rows:
            fails = ", ".join(f"{k} {v}" for k, v in sorted(d["failures"].items(), key=lambda kv: -kv[1]))
            est = d.get("est_error_cm", {}).get("median")
            out.append(f"| {name} | {d['mission']} | {Path(str(d['policy'])).name} | {d['n']} | {d['success']}/{d['n']} = {pct(d['success_rate'])} | "
                       f"{ci(d['ci95'])} | {d['fruits_placed']}/{d['fruits_total']} | {'' if est is None else f'{est:.1f} cm'} | {fails} |")
        out.append("")
        # paired comparison reference vs trained policy on identical seeds
        for m in ("mission1", "mission2", "mission3"):
            fr, ft = ROOT / f"outputs/eval/{m}_reference.json", ROOT / f"outputs/eval/{m}_trained.json"
            if not (fr.exists() and ft.exists()):
                continue
            r = {e["seed"]: e for e in json.loads(fr.read_text())["episodes"]}
            t = {e["seed"]: e for e in json.loads(ft.read_text())["episodes"]}
            common = sorted(set(r) & set(t))
            better = [s for s in common if t[s]["n_placed"] > r[s]["n_placed"] or (t[s]["success"] and not r[s]["success"])]
            worse = [s for s in common if t[s]["n_placed"] < r[s]["n_placed"] or (r[s]["success"] and not t[s]["success"])]
            same = sum((r[s]["failure"] or "ok") == (t[s]["failure"] or "ok") and r[s]["n_placed"] == t[s]["n_placed"] for s in common)
            out.append(f"- {m}: policy vs referensi pada {len(common)} seed yang sama: lebih baik {len(better)} {better}, "
                       f"lebih buruk {len(worse)} {worse}, hasil identik {same}.")
            if m == "mission3":
                for tag, d in (("referensi", r), ("policy", t)):
                    dmg = sum(int(e.get("damage_events", 0)) for e in d.values())
                    sk = {}
                    for e in d.values():
                        for k in e.get("skipped", []):
                            sk[k] = sk.get(k, 0) + 1
                    out.append(f"  - Misi 3 {tag}: buah di keranjang {sum(e['n_placed'] for e in d.values())}/"
                               f"{sum(e['n_targets'] for e in d.values())}, kejadian kerusakan {dmg}, buah dilewati {sk}, "
                               f"jarak tempuh kendaraan {[round(e.get('vehicle_travel', 0), 2) for e in d.values()]} m")
        out.append("")
    for m in ("mission1", "mission2", "mission3"):
        info = ROOT / "checkpoints" / m / "init_info.json"
        log = ROOT / "checkpoints" / m / "eval_log.csv"
        if info.exists():
            out += [f"## Training {m}", "", "init: `" + json.loads(info.read_text()).get("kind", "?") + "`"]
            if log.exists():
                out += ["", "```", log.read_text().strip(), "```"]
            out.append("")
    (ROOT / "docs/RESULTS_TABLES.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
