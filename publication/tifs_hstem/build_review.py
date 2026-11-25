"""Render the final paper, its baseline, and the figure review assets."""

from io import BytesIO
import re

import pymupdf
from PIL import Image, ImageDraw, ImageOps

from build_tables import HERE, git_bytes


BASELINE = "bd671cdff36e3546a4217e4e754caf4cd79c420b"
PAGE_DPI = 115
FIGURE_DPI = 220
PDF_DIR = HERE / "output/pdf"
PREVIEW_DIR = HERE / "output/preview"
REVIEW_DIR = HERE / "tmp/review"
MANUSCRIPT = "hstem_tifs_manuscript.pdf"
FIGURES = (
    ("Cross-receiver task", "cross_receiver_system"),
    ("HStem and attention backend", "identification_pipeline"),
    ("Matched convolution-path geometry", "matched_tokenizers"),
    ("Accuracy on ten receivers", "receiver_performance"),
    ("Accuracy and deployment cost", "native_efficiency"),
    ("Six paired tokenizer comparisons", "paired_tokenizers"),
    ("Mechanism controls", "mechanism_ablations"),
    ("Layerwise representation geometry", "representation_geometry"),
)


def render(page, dpi, grayscale=False):
    space = pymupdf.csGRAY if grayscale else pymupdf.csRGB
    pixmap = page.get_pixmap(dpi=dpi, colorspace=space, alpha=False)
    return Image.frombytes("L" if grayscale else "RGB", (pixmap.width, pixmap.height),
                           pixmap.samples)


def page_images(doc, directory):
    directory.mkdir(parents=True, exist_ok=True)
    images = []
    for index, page in enumerate(doc, 1):
        im = render(page, PAGE_DPI)
        im.save(directory / f"page-{index:02}.png")
        images.append(im)
    return images


def compressed(image):
    payload = BytesIO()
    image.save(payload, format="JPEG", quality=94, subsampling=0)
    return payload.getvalue()


def page_comparison(before, after):
    with pymupdf.open() as compare:
        for index, revised in enumerate(after, 1):
            sheet = compare.new_page(width=1020, height=680)
            sheet.insert_text((22, 22), "HStem / IEEE TIFS - baseline and revised page order",
                              fontsize=12, fontname="hebo", color=(.09, .18, .21))
            for x, left, title in [(20, True, f"Baseline {BASELINE[:7]}"),
                                 (520, False, "Revised manuscript")]:
                area = pymupdf.Rect(x, 50, x+480, 669)
                sheet.draw_rect(area, color=(.76, .80, .82), fill=(.96, .97, .98), width=.6)
                sheet.insert_text((x, 42), f"{title}   /   p. {index}", fontsize=9)
                if left and index > len(before):
                    sheet.insert_text((x+18, 78), "The baseline ends on page 8.", fontsize=10)
                    continue
                img = before[index-1] if left else revised
                sheet.insert_image(area, stream=compressed(img), keep_proportion=True)
        compare.set_metadata({"title": "HStem TIFS manuscript - baseline and revision comparison"})
        compare.save(PDF_DIR / "hstem_tifs_page_comparison.pdf", deflate=True)


def find_figures(doc):
    pages = {}
    for page in doc:
        for block in page.get_text("blocks"):
            match = re.match(r"^Fig\.\s+([1-8])\.", block[4].strip())
            if match:
                pages[int(match[1])] = page.number + 1
    if sorted(pages) != list(range(1, 9)):
        raise ValueError(f"Expected all eight numbered figure captions, found {pages}")
    return pages


def figure_preview(pages):
    paths = []
    output = REVIEW_DIR / "figures"
    output.mkdir(exist_ok=True)
    for number, (title, stem) in enumerate(FIGURES, 1):
        with pymupdf.open(HERE / "figures" / f"{stem}.pdf") as doc:
            rgb = render(doc[0], FIGURE_DPI)
            rgb.save(output / f"fig-{number:02}-color.png")
            render(doc[0], FIGURE_DPI, grayscale=True).save(output / f"fig-{number:02}-gray.png")
            paths.append((number, title, rgb))
    for name, selection in [
        ("system_and_tokenizer", paths[:3]),
        ("receiver_and_mechanism", [paths[i] for i in [3, 4, 5, 6, 7]]),
    ]:
        prepared = [(number, title, ImageOps.contain(im, (1160, 840)))
                    for number, title, im in selection]
        width = 1216
        height = 20 + sum(im.height+54 for _, _, im in prepared)
        board = Image.new("RGB", (width, height), "#F1F4F6")
        draw = ImageDraw.Draw(board)
        y = 20
        for number, title, im in prepared:
            draw.text((28, y), f"FIG. {number}  /  {title}  /  MANUSCRIPT P. {pages[number]}",
                      fill="#24323A", font_size=20)
            board.paste(im, (28, y+30))
            y += im.height+54
        board.save(PREVIEW_DIR / f"{name}.png", optimize=True)


def key_pages(pages, renders):
    selected = [pages[number] for number in (1, 2, 3, 4, 7)]
    sheet = Image.new("RGB", (1450, 1280), "#F1F4F6")
    draw = ImageDraw.Draw(sheet)
    for index, page_number in enumerate(selected):
        x, y = 15+(index % 3)*480, 15+(index // 3)*635
        draw.text((x+4, y), f"REVISED MANUSCRIPT  /  P. {page_number}", fill="#24323A", font_size=19)
        view = ImageOps.contain(renders[page_number-1], (456, 590))
        sheet.paste(view, (x, y+27))
    sheet.save(PREVIEW_DIR / "manuscript_key_pages.jpg", quality=94, subsampling=0)


def main():
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = f"publication/tifs_hstem/output/pdf/{MANUSCRIPT}"
    baseline_data = git_bytes("show", f"{BASELINE}:{baseline_path}")
    with pymupdf.open(stream=baseline_data, filetype="pdf") as old, \
            pymupdf.open(PDF_DIR / MANUSCRIPT) as new:
        before = page_images(old, REVIEW_DIR / "baseline")
        after = page_images(new, REVIEW_DIR / "revised")
        if (len(before), len(after)) != (8, 12):
            raise ValueError(f"Expected 8 baseline and 12 revised pages, got {len(before)} and {len(after)}")
        pages = find_figures(new)
        page_comparison(before, after)
        figure_preview(pages)
        key_pages(pages, after)
        print("Figure pages:", pages)


if __name__ == "__main__":
    main()
