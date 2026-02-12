import sys
import time
from pathlib import Path

import cv2
import numpy as np


# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_STEPS = True
WAIT_KEY_MS = 0  # 0 = tunggu tombol, 1 = cepat, 300 = 0.3 detik

# EasyOCR config
OCR_LANGS = ["en"]   # stabil untuk buku english
USE_GPU = False

# Page detection config
MIN_PAGE_AREA_RATIO = 0.18          # kandidat minimal 18% luas image
TEXT_DENSITY_THRESHOLD = 0.010      # minimal density dianggap ada teks
MAX_CANDIDATES = 2                  # cukup top 2 kandidat


# =========================
# DEBUG VIEWER (AUTO CLOSE)
# =========================
_last_window = None

def show(name, img):
    """
    Menampilkan 1 window saja.
    Window sebelumnya otomatis ditutup biar tidak numpuk.
    """
    global _last_window

    if not SHOW_STEPS:
        return

    if _last_window is not None:
        try:
            cv2.destroyWindow(_last_window)
        except:
            pass

    view = img
    if len(img.shape) == 2:
        view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # resize agar muat layar
    h, w = view.shape[:2]
    max_w = 1100
    if w > max_w:
        scale = max_w / w
        view = cv2.resize(view, (int(w * scale), int(h * scale)))

    cv2.imshow(name, view)
    _last_window = name

    print(f"[DEBUG] imshow: {name} (tekan tombol untuk lanjut)")
    cv2.waitKey(WAIT_KEY_MS)


# =========================
# BASIC UTILS
# =========================
def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def crop_rect(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]


# =========================
# PAGE DETECTION (MASK-BASED)
# =========================
def binarize_for_page(gray):
    """
    Mask untuk mendeteksi area halaman/kertas.
    Lebih stabil dibanding canny edges.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 15
    )

    # invert supaya kertas jadi "blob"
    inv = 255 - th

    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


def find_page_rects(img_bgr):
    """
    Cari kandidat halaman berdasarkan area mask.
    Return list rect (x,y,w,h) sorted terbesar.
    """
    h, w = img_bgr.shape[:2]
    img_area = h * w

    gray = to_gray(img_bgr)
    show("01_gray", gray)

    page_mask = binarize_for_page(gray)
    show("02_page_mask", page_mask)

    contours, _ = cv2.findContours(page_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * MIN_PAGE_AREA_RATIO:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        rect_area = bw * bh

        if rect_area < img_area * MIN_PAGE_AREA_RATIO:
            continue

        rects.append((rect_area, (x, y, bw, bh)))

    rects.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in rects[:5]]


# =========================
# OCR PREPROCESSING
# =========================
def preprocess_for_ocr(img_bgr):
    """
    Preprocess OCR:
    - grayscale
    - denoise
    - CLAHE (contrast)
    - adaptive threshold
    """
    gray = to_gray(img_bgr)
    show("OCR_01_gray", gray)

    gray = cv2.fastNlMeansDenoising(gray, h=10)
    show("OCR_02_denoise", gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    show("OCR_03_clahe", gray)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )
    show("OCR_04_threshold", th)

    return th


def estimate_text_density(binary_img):
    """
    teks dianggap pixel hitam
    """
    black = np.sum(binary_img < 128)
    total = binary_img.size
    return black / total


# =========================
# TEXT REGION CROP (REMOVE BORDER)
# =========================
def crop_to_text_region(binary_img, original_bgr):
    """
    Menghilangkan border hitam / margin kosong.
    binary_img = hasil threshold
    original_bgr = gambar asli halaman
    """

    # teks = hitam -> invert supaya teks putih
    inv = 255 - binary_img

    # Gabungkan teks jadi blok besar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    merged = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)

    show("TEXT_01_merged_mask", merged)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[WARN] Tidak ada text contour, skip crop_to_text_region()")
        return original_bgr

    # ambil contour terbesar (blok teks utama)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # padding biar tidak kepotong
    pad = 15
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(original_bgr.shape[1] - x, w + 2 * pad)
    h = min(original_bgr.shape[0] - y, h + 2 * pad)

    cropped = original_bgr[y:y+h, x:x+w]
    show("TEXT_02_cropped_final", cropped)

    return cropped


# =========================
# DECIDE PAGES (YOUR LOGIC)
# =========================
def decide_pages(img_bgr):
    """
    Logic sesuai request kamu:
    - Kalau foto dominan 1 halaman tapi nyempil halaman kedua -> ambil yang text paling banyak
    - Kalau 2 halaman full -> OCR keduanya
    - Kalau 1 kandidat kosong -> skip
    """
    rects = find_page_rects(img_bgr)

    debug = img_bgr.copy()
    for i, r in enumerate(rects[:5]):
        x, y, bw, bh = r
        cv2.rectangle(debug, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
        cv2.putText(debug, f"cand{i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    show("03_page_candidates_rect", debug)

    if len(rects) == 0:
        print("[WARN] Tidak menemukan area halaman. Pakai full image.")
        return [img_bgr]

    rects = rects[:MAX_CANDIDATES]

    pages = []
    densities = []

    for idx, rect in enumerate(rects):
        cropped = crop_rect(img_bgr, rect)

        # hitung text density berdasarkan preprocess
        pre = preprocess_for_ocr(cropped)
        density = estimate_text_density(pre)

        print(f"[DEBUG] Candidate {idx}: rect={rect}, density={density:.4f}")

        pages.append(cropped)
        densities.append(density)

    # hanya 1 kandidat
    if len(pages) == 1:
        return [pages[0]]

    # 2 kandidat
    d1, d2 = densities[0], densities[1]

    # 2 halaman full
    if d1 >= TEXT_DENSITY_THRESHOLD and d2 >= TEXT_DENSITY_THRESHOLD:
        print("[INFO] Terdeteksi 2 halaman full -> OCR dua-duanya")
        return pages

    # hanya 1 halaman dominan (pilih yang density lebih tinggi)
    if d1 >= d2:
        print("[INFO] Terdeteksi 1 halaman dominan -> ambil kandidat 0 saja")
        return [pages[0]]
    else:
        print("[INFO] Terdeteksi 1 halaman dominan -> ambil kandidat 1 saja")
        return [pages[1]]


# =========================
# OCR
# =========================
def init_easyocr():
    print("[DEBUG] Import easyocr...")
    import easyocr

    print("[DEBUG] Init EasyOCR Reader...")
    start = time.time()
    reader = easyocr.Reader(OCR_LANGS, gpu=USE_GPU)
    print(f"[DEBUG] Reader siap. Waktu init: {time.time() - start:.2f} detik")
    return reader


def ocr_page(reader, page_img_bgr, page_index: int) -> str:
    show(f"PAGE_{page_index}_cropped", page_img_bgr)

    # Preprocess 1
    pre = preprocess_for_ocr(page_img_bgr)

    # Crop ulang berdasarkan area teks (buang border hitam kanan)
    page_img_bgr = crop_to_text_region(pre, page_img_bgr)

    # Preprocess ulang setelah crop final
    pre = preprocess_for_ocr(page_img_bgr)

    # Save preprocessed
    pre_path = OUTPUT_DIR / f"page_{page_index}_preprocessed.png"
    cv2.imwrite(str(pre_path), pre)
    print(f"[OK] Preprocessed disimpan: {pre_path}")

    # OCR
    print(f"[DEBUG] OCR readtext page {page_index}...")
    start = time.time()
    lines = reader.readtext(str(pre_path), detail=0)
    print(f"[DEBUG] OCR selesai page {page_index} ({len(lines)} baris) dalam {time.time() - start:.2f}s")

    return "\n".join(lines)


# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) < 2:
        print("Cara pakai:")
        print("  python OCR.py <path_gambar>")
        print("Contoh:")
        print("  python OCR.py sample.jpeg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"[ERROR] File tidak ditemukan: {image_path}")
        sys.exit(1)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print("[ERROR] cv2 gagal membaca image")
        sys.exit(1)

    print(f"[INFO] Input: {image_path.resolve()}")
    show("00_input", img_bgr)

    # Decide pages
    pages = decide_pages(img_bgr)
    print(f"[INFO] Total halaman yang akan di-OCR: {len(pages)}")

    # Init OCR
    reader = init_easyocr()

    # OCR pages
    full_parts = []
    for i, page in enumerate(pages, start=1):
        text = ocr_page(reader, page, i)

        out_txt = OUTPUT_DIR / f"page_{i}_ocr_raw.txt"
        out_txt.write_text(text, encoding="utf-8")
        print(f"[OK] OCR page {i} disimpan: {out_txt}")

        full_parts.append(f"===== PAGE {i} =====\n{text}")

    full_text = "\n\n".join(full_parts)

    out_full = OUTPUT_DIR / "ocr_raw.txt"
    out_full.write_text(full_text, encoding="utf-8")

    print("\n[OK] Semua selesai.")
    print(f"[OK] Output gabungan: {out_full}")
    print("\nPreview 400 char pertama:\n")
    print(full_text[:400])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
