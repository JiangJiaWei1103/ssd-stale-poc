"""Accept-length decay vs staleness lag -> a self-contained SVG line chart,
plus a PNG for the README (inline SVG render on github.com is flaky; PNG is safe).

No matplotlib: emits raw SVG from the real C-repeat results (one glance = "how
accept length drops with lag"). The README embeds the PNG.

    python3 plots.py
    # SVG -> PNG (macOS, zero deps):
    qlmanage -t -s 1440 -o figs figs/accept_vs_lag.svg && mv figs/accept_vs_lag.svg.png figs/accept_vs_lag.png
"""
from __future__ import annotations

import json
from pathlib import Path

LAGS = [0, 1, 2, 3]
YMAX = 5.5  # headroom above the 4.78 point so its label clears the "5" gridline
BLUE, ORANGE, GRID, TEXT, MUTED = "#2563eb", "#ea580c", "#e5e7eb", "#111827", "#6b7280"


def _read(results_dir: str = "results"):
    rows = []
    for lag in LAGS:
        d = json.loads((Path(results_dir) / "C_repeat" / f"lag{lag}.json").read_text())
        rows.append(
            {
                "lag": lag,
                "incl": d["accept_length_incl_bonus"],
                "excl": d["mean_correct_drafts"],
            }
        )
    b = rows[0]["incl"]
    for r in rows:
        r["ratio"] = r["incl"] / b
    return rows


def build_svg(rows) -> str:
    W, H = 720, 720  # square canvas: qlmanage/Quick Look pads thumbnails to square
    L, R, T, B = 66, 158, 54, 80
    x0, x1, y0, y1 = L, W - R, T, H - B
    xmax = max(r["lag"] for r in rows)

    def X(lag: float) -> float:
        return x0 + (lag / xmax) * (x1 - x0)

    def Y(v: float) -> float:
        return y1 - (v / YMAX) * (y1 - y0)

    e: list[str] = []
    e.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">'
    )
    e.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    e.append(
        f'<text x="{L}" y="26" font-size="17" font-weight="700" fill="{TEXT}">'
        f"DSpark accept length vs staleness lag</text>"
    )
    e.append(
        f'<text x="{L}" y="44" font-size="12" fill="{MUTED}">'
        f"C-repeat &#183; N=64 &#183; gsm8k &#183; temp=0 &#183; lag=0 is vanilla</text>"
    )

    # y grid + labels
    for v in range(int(YMAX) + 1):
        y = Y(v)
        e.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}"/>')
        e.append(
            f'<text x="{x0 - 10}" y="{y + 4:.1f}" font-size="12" fill="{MUTED}" '
            f'text-anchor="end">{v}</text>'
        )
    # x ticks + labels
    for r in rows:
        x = X(r["lag"])
        e.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1 + 5}" stroke="{MUTED}"/>')
        e.append(
            f'<text x="{x:.1f}" y="{y1 + 22}" font-size="12" fill="{TEXT}" '
            f'text-anchor="middle">{r["lag"]}</text>'
        )
    # axis titles
    e.append(
        f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 16}" font-size="13" fill="{TEXT}" '
        f'text-anchor="middle">staleness lag (verifier rounds behind)</text>'
    )
    e.append(
        f'<text transform="translate(20,{(y0 + y1) / 2:.0f}) rotate(-90)" font-size="13" '
        f'fill="{TEXT}" text-anchor="middle">accept length (tokens / verify step)</text>'
    )

    def series(key: str, color: str, label_dy: int):
        pts = " ".join(f"{X(r['lag']):.1f},{Y(r[key]):.1f}" for r in rows)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for r in rows:
            x, y = X(r["lag"]), Y(r[key])
            e.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
            e.append(
                f'<text x="{x:.1f}" y="{y + label_dy:.1f}" font-size="11.5" '
                f'fill="{color}" text-anchor="middle" font-weight="600">{r[key]:.2f}</text>'
            )

    series("incl", BLUE, -10)    # incl bonus (throughput-relevant)
    series("excl", ORANGE, 18)   # excl bonus (draft quality)

    # legend (right gutter)
    lx, ly = x1 + 20, y0 + 20
    e.append(f'<rect x="{lx}" y="{ly - 14}" width="12" height="12" fill="{BLUE}" rx="2"/>')
    e.append(f'<text x="{lx + 18}" y="{ly - 4}" font-size="12" fill="{TEXT}">accept_len</text>')
    e.append(f'<text x="{lx + 18}" y="{ly + 10}" font-size="11" fill="{MUTED}">(incl bonus,</text>')
    e.append(f'<text x="{lx + 18}" y="{ly + 23}" font-size="11" fill="{MUTED}">throughput)</text>')
    e.append(f'<rect x="{lx}" y="{ly + 40}" width="12" height="12" fill="{ORANGE}" rx="2"/>')
    e.append(f'<text x="{lx + 18}" y="{ly + 50}" font-size="12" fill="{TEXT}">correct_drafts</text>')
    e.append(f'<text x="{lx + 18}" y="{ly + 64}" font-size="11" fill="{MUTED}">(excl bonus,</text>')
    e.append(f'<text x="{lx + 18}" y="{ly + 77}" font-size="11" fill="{MUTED}">draft quality)</text>')
    # ratio annotation
    e.append(
        f'<text x="{lx}" y="{ly + 108}" font-size="11.5" fill="{TEXT}" font-weight="600">'
        f"incl ratio vs lag0:</text>"
    )
    for i, r in enumerate(rows):
        e.append(
            f'<text x="{lx}" y="{ly + 126 + i * 15:.0f}" font-size="11" fill="{MUTED}">'
            f'lag{r["lag"]}: {r["ratio"]:.3f}</text>'
        )

    e.append("</svg>")
    return "\n".join(e)


def main(results_dir: str = "results", figs_dir: str = "figs") -> None:
    figs = Path(figs_dir)
    figs.mkdir(parents=True, exist_ok=True)
    out = figs / "accept_vs_lag.svg"
    out.write_text(build_svg(_read(results_dir)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
