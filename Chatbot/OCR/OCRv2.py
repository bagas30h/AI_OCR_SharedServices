import cv2
import numpy as np
import sys


# =========================
# SHOW STEP (press q)
# =========================
def show_step(title, img, max_w=1200, max_h=900):
    disp = img.copy()
    h, w = disp.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        disp = cv2.resize(disp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cv2.imshow(title, disp)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyWindow(title)


# =========================
# STEP 1-3: Gray -> Shadow Remove -> Denoise
# =========================
def shadow_remove(gray, sigma=25):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.divide(gray, blur, scale=255)


def preprocess_until_denoise(image_bgr, debug=True):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if debug:
        show_step("1) Gray", gray)

    norm = shadow_remove(gray, sigma=25)
    if debug:
        show_step("2) Shadow Removed", norm)

    den = cv2.fastNlMeansDenoising(norm, None, 10, 7, 21)
    if debug:
        show_step("3) Denoised (BEST for gutter)", den)

    return den


# =========================
# STEP 4-6: Gutter detection
# =========================
def detect_edges_for_gutter(denoised, debug=True):
    blur = cv2.GaussianBlur(denoised, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    if debug:
        show_step("4) Edges (from denoised)", edges)

    return edges


def find_gutter_line_from_denoised(denoised, edges, debug=True):
    """
    Return:
    - best_line in FULL IMAGE coords: (x1,y1,x2,y2)
    - fallback vertical x if fail
    """
    h, w = denoised.shape

    sx1 = int(w * 0.45)
    sx2 = int(w * 0.95)

    roi_edges = edges[:, sx1:sx2]
    if debug:
        show_step("5) ROI Edges (search gutter)", roi_edges)

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=int(h * 0.35),
        maxLineGap=30
    )

    if lines is None:
        gutter_x = int(w * 0.78)
        if debug:
            vis = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            cv2.line(vis, (gutter_x, 0), (gutter_x, h - 1), (0, 0, 255), 3)
            show_step("6) Gutter FAILED -> fallback vertical (RED)", vis)
        return None, gutter_x

    best = None
    best_score = -1

    for L in lines:
        rx1, ry1, rx2, ry2 = L[0]

        dx = abs(rx2 - rx1)
        dy = abs(ry2 - ry1)
        length = (dx * dx + dy * dy) ** 0.5

        # harus vertikal-ish
        if dy < dx * 2.5:
            continue

        score = length
        if score > best_score:
            best_score = score
            best = (rx1, ry1, rx2, ry2)

    if best is None:
        gutter_x = int(w * 0.78)
        if debug:
            vis = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            cv2.line(vis, (gutter_x, 0), (gutter_x, h - 1), (0, 0, 255), 3)
            show_step("6) No vertical-ish line -> fallback vertical (RED)", vis)
        return None, gutter_x

    rx1, ry1, rx2, ry2 = best

    # map ke full coords
    x1 = sx1 + rx1
    x2 = sx1 + rx2
    y1 = ry1
    y2 = ry2

    if debug:
        vis = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)  # gutter line
        show_step("6) Detected Gutter LINE (GREEN)", vis)

    gutter_x = int((x1 + x2) / 2)
    return (x1, y1, x2, y2), gutter_x


# =========================
# STEP 7: Dynamic cut by line (GREEN)
# =========================
def dynamic_cut_left_by_line(gray, line, margin_px=10, debug=True):
    """
    Potong halaman kiri mengikuti garis miring:
    x_cut(y) = a*y + b

    Output:
    - left_page (grayscale)
    """

    h, w = gray.shape

    if line is None:
        # fallback: potong biasa
        cut_x = int(w * 0.78)
        left = gray[:, :cut_x]
        if debug:
            show_step("7) Dynamic cut fallback (vertical)", left)
        return left

    x1, y1, x2, y2 = line

    # hindari division by zero
    if y2 == y1:
        cut_x = int((x1 + x2) / 2)
        left = gray[:, :cut_x]
        if debug:
            show_step("7) Dynamic cut fallback (horizontal line?)", left)
        return left

    # x = a*y + b
    a = (x2 - x1) / float(y2 - y1)
    b = x1 - a * y1

    # cari max width output
    xs = []
    for y in range(h):
        x = int(a * y + b) - margin_px
        x = max(1, min(w - 1, x))
        xs.append(x)

    out_w = max(xs)
    out = np.full((h, out_w), 255, dtype=np.uint8)

    # copy row by row
    for y in range(h):
        xcut = xs[y]
        out[y, :xcut] = gray[y, :xcut]

    if debug:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for y in range(0, h, max(5, h // 80)):
            x = xs[y]
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        show_step("7) Dynamic cut points (RED dots)", vis)
        show_step("8) Left page after dynamic cut", out)

    return out


# =========================
# STEP 8: Binarize AFTER cut
# =========================
def binarize_after_crop(page1_gray, debug=True):
    norm = shadow_remove(page1_gray, sigma=25)
    norm = cv2.fastNlMeansDenoising(norm, None, 10, 7, 21)
    bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if debug:
        show_step("9) Binarized AFTER dynamic cut", bw)

    return bw


# =========================
# STEP 9: Crop rapat
# =========================
def crop_to_content(binary, pad=12, debug=True):
    ys, xs = np.where(binary < 250)
    if len(xs) == 0:
        return binary

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(binary.shape[1] - 1, x1 + pad)
    y1 = min(binary.shape[0] - 1, y1 + pad)

    cropped = binary[y0:y1 + 1, x0:x1 + 1]

    if debug:
        show_step("10) Cropped to content", cropped)

    return cropped


# =========================
# STEP 10: Deskew OCR-friendly
# =========================
def deskew_by_text(binary, debug=True):
    img = binary.copy()

    inv = 255 - img
    inv = cv2.dilate(inv, np.ones((3, 3), np.uint8), iterations=1)

    edges = cv2.Canny(inv, 50, 150)

    if debug:
        show_step("11) Deskew edges", edges)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=int(img.shape[1] * 0.30),
        maxLineGap=20
    )

    if lines is None:
        if debug:
            show_step("12) Deskew skipped (no lines)", img)
        return img

    angles = []
    for L in lines:
        x1, y1, x2, y2 = L[0]
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) < 25:
            angles.append(angle)

    if len(angles) < 5:
        if debug:
            show_step("12) Deskew skipped (not enough lines)", img)
        return img

    rot_angle = float(np.median(angles))

    (h, w) = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), rot_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if debug:
        show_step(f"12) Deskewed (angle={rot_angle:.2f})", rotated)

    return rotated


# =========================
# STEP 11: Sharpen
# =========================
def sharpen(binary, debug=True):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(binary, -1, kernel)

    if debug:
        show_step("14) Sharpened FINAL", sharp)

    return sharp


# =========================
# MAIN PIPELINE
# =========================
def scan_page1_only(input_path, output_path, debug=True):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Gambar tidak bisa dibaca: " + input_path)

    if debug:
        show_step("0) Original", img)

    den = preprocess_until_denoise(img, debug=debug)

    edges = detect_edges_for_gutter(den, debug=debug)

    gutter_line, gutter_x = find_gutter_line_from_denoised(den, edges, debug=debug)

    # POTONG PAKAI GARIS IJO (dynamic cut)
    page1_gray = dynamic_cut_left_by_line(den, gutter_line, margin_px=15, debug=debug)

    # binarize
    page1_bw = binarize_after_crop(page1_gray, debug=debug)

    # crop rapat
    page1_bw = crop_to_content(page1_bw, pad=20, debug=debug)

    # deskew
    page1_bw = deskew_by_text(page1_bw, debug=debug)

    # crop rapat lagi setelah rotasi
    page1_bw = crop_to_content(page1_bw, pad=12, debug=debug)

    # sharpen
    page1_bw = sharpen(page1_bw, debug=debug)

    cv2.imwrite(output_path, page1_bw)
    print("DONE ->", output_path)

    cv2.destroyAllWindows()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python OCR_FINAL_DYNAMIC_CUT.py <input_image> [output_image]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "hasil_page1.png"

    scan_page1_only(inp, out, debug=True)
