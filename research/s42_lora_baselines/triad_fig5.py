"""Recover receiver-equal descriptive values from the authors' vector Figure 5."""

import argparse
import csv
from hashlib import file_digest
import json
from pathlib import Path
from statistics import mean

import pymupdf


RX = (
    [f"rtl_{i}" for i in range(1,10)]
    + ["pluto_1","pluto_2","b200_1","b200_2","b200_mini_1","b200_mini_2",
       "b210_1","b210_2","n210_1","n210_2","n210_3"]
)
TRAIN = {
    "homogeneous": {f"rtl_{i}" for i in range(1,6)},
    "heterogeneous": {"rtl_1","pluto_1","b200_1","b200_mini_1","b210_1"},
}
COLORS = {"author_baseline": (31,119,180), "triad_gan": (255,127,14)}


def extract(pdf):
    doc = pymupdf.open(pdf)
    page = doc[4]
    title = " ".join(page.get_text().split())
    assert "Fig. 5: LoRa classification accuracy under receiver change." in title
    drawings = page.get_drawings()
    # Gray horizontal gridlines at 40, 60, 80, and 100 percent on the right.
    grid = sorted(d["rect"].y0 for d in drawings if
                  d["type"]=="s" and d["rect"].x0>350 and d["rect"].width>190
                  and d["rect"].height<.001 and d.get("color") and
                  all(.49 < value < .52 for value in d["color"]))
    assert len(grid) == 8
    output = []
    for v,style in enumerate(TRAIN):
        top,mid_hi,mid_lo,bottom = grid[4*v:4*v+4]
        scale = (bottom-top)/60
        assert abs((mid_hi-top)/scale-20) < .001 and abs((mid_lo-top)/scale-40) < .001
        for arm,color in COLORS.items():
            bars = sorted((d["rect"] for d in drawings if d.get("fill") and
                           tuple(round(c*255) for c in d["fill"])==color and
                           d["rect"].x0>350 and 3 < d["rect"].width < 4 and
                           bottom+49 < d["rect"].y1 < bottom+50), key=lambda r:r.x0)
            assert len(bars)==20
            assert all(abs(b.x0-a.x0-9.117)<.01 for a,b in zip(bars,bars[1:]))
            for rx,bar in zip(RX,bars):
                value = 100-(bar.y0-top)/scale
                assert 40<=value<=100.01
                output.append(dict(scheme=style,method=arm,rx=rx,
                                   source_rx=rx in TRAIN[style],
                                   accuracy_pct=round(min(value,100), 1)))
    return output


def run(pdf,out):
    rows=extract(pdf)
    with pdf.open("rb") as source:
        digest=file_digest(source,"sha256").hexdigest()
    means={}
    for scheme in TRAIN:
        means[scheme]={}
        for method in COLORS:
            by_arm=[r for r in rows if r["scheme"]==scheme and r["method"]==method]
            held=[r["accuracy_pct"] for r in by_arm if not r["source_rx"]]
            assert len(by_arm)==20 and len(held)==15
            means[scheme][method]=dict(unseen_rx_n=len(held),
                                      unseen_rx_mean_pct=round(mean(held),2),
                                      unseen_rx_worst_pct=min(held),
                                      all_20_mean_pct=round(mean(r["accuracy_pct"] for r in by_arm),2))
    info=dict(source_pdf_sha256=digest,figure="5a/5b, PDF page 5",method=means,
              scope="Vector-derived estimates from the authors' protocol; not P-H5x160 scores")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"TRIAD_FIG5.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out/"TRIAD_FIG5.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(info,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("pdf",type=Path)
    parser.add_argument("out",type=Path)
    arg=parser.parse_args()
    run(arg.pdf,arg.out)
