#!/usr/bin/env python3
"""Interactive Mobile SAM evaluation: click positive/negative points, save overlays + JSON."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.gridspec import GridSpec

from image_segmentation.config import default_config
from image_segmentation.inference import load_model, segment_with_prompts
from image_segmentation.io_utils import build_run_metadata, list_test_images, utc_timestamp, write_json
from image_segmentation.visualization import compose_eval_view, save_bgr_png


CFG = None
MODEL = None
PATHS: list[Path] = []
IDX = 0
IMAGE_BGR: np.ndarray | None = None
MASK: np.ndarray | None = None
POINTS: list[tuple[float, float]] = []
LABELS: list[int] = []
BOXES: list[tuple[float, float, float, float]] = []
BOX_MODE = False
DRAG_START: tuple[float, float] | None = None
DRAG_PATCH = None

FIG = None
AX = None
DISP = None
TRAY_AX = None
TRAY_DISP = None
SAVED_PNGS: list[Path] = []
META_TEXT = None
WARN_TEXT = None


def _ratio_to_float(r) -> float:
    try:
        return float(r)
    except TypeError:
        return r[0] / r[1]


def _dms_to_decimal(dms, ref: str) -> float:
    deg = _ratio_to_float(dms[0])
    minutes = _ratio_to_float(dms[1])
    seconds = _ratio_to_float(dms[2])
    dec = deg + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        dec = -dec
    return dec


def _get_exif_gps_lat_lon(image_path: Path) -> tuple[float, float] | None:
    """Return (lat, lon) from EXIF GPS tags if present, else None."""
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS
    except Exception:
        return None

    try:
        img = Image.open(image_path)
        exif = img.getexif()
        if not exif:
            # Fallback for some Pillow/backends
            exif = getattr(img, "_getexif", lambda: None)()  # noqa: SLF001
        if not exif:
            return None

        # Pillow's Exif object supports get_ifd(tag). Use it if available.
        gps_ifd = None
        if hasattr(exif, "get_ifd"):
            try:
                gps_ifd = exif.get_ifd(34853)  # GPSInfo
            except Exception:
                gps_ifd = None

        # Fallback: locate GPSInfo (34853) by name
        if gps_ifd is None:
            for k, v in getattr(exif, "items", lambda: [])():
                if TAGS.get(k) == "GPSInfo" or k == 34853:
                    gps_ifd = v
                    break
        if gps_ifd is None:
            return None

        # Normalize to a dict-like mapping of GPS tags
        if hasattr(gps_ifd, "items"):
            gps = {GPSTAGS.get(t, t): v for t, v in gps_ifd.items()}
        else:
            gps = {GPSTAGS.get(t, t): gps_ifd[t] for t in gps_ifd}

        lat = gps.get("GPSLatitude")
        lat_ref = gps.get("GPSLatitudeRef")
        lon = gps.get("GPSLongitude")
        lon_ref = gps.get("GPSLongitudeRef")
        if not (lat and lat_ref and lon and lon_ref):
            return None

        return (_dms_to_decimal(lat, str(lat_ref)), _dms_to_decimal(lon, str(lon_ref)))
    except Exception:
        return None


def _maybe_disable_toolbar() -> None:
    """Best-effort disable of interactive toolbar (pan/zoom) to avoid click coordinate drift."""
    if FIG is None:
        return
    mgr = FIG.canvas.manager
    tb = getattr(mgr, "toolbar", None)
    if tb is None:
        return
    # Try common backends (Qt, Tk). If it fails, we still guard against active modes in callbacks.
    try:
        if hasattr(tb, "setVisible"):  # Qt
            tb.setVisible(False)
    except Exception:
        pass


def _toolbar_mode() -> str:
    """Return current toolbar navigation mode string, if any."""
    if FIG is None:
        return ""
    mgr = FIG.canvas.manager
    tb = getattr(mgr, "toolbar", None)
    if tb is None:
        return ""
    return getattr(tb, "mode", "") or ""


def _sync_axes_to_image() -> None:
    """Ensure axes/data coordinates align with image pixel coordinates."""
    if AX is None or DISP is None or IMAGE_BGR is None:
        return
    h, w = IMAGE_BGR.shape[:2]
    # x in [0, w), y in [0, h) with origin at top-left (like image pixels)
    DISP.set_extent((0, w, h, 0))
    AX.set_xlim(0, w)
    AX.set_ylim(h, 0)


def _refresh_title() -> None:
    assert FIG is not None
    if not PATHS:
        return
    name = PATHS[IDX].name
    mgr = FIG.canvas.manager
    if mgr is not None and getattr(mgr, "set_window_title", None):
        try:
            mgr.set_window_title(f"Mobile SAM eval — {name} ({IDX + 1}/{len(PATHS)})")
        except Exception:
            pass
    FIG.suptitle(
        f"{name}  |  mode={'BOX' if BOX_MODE else 'POINT'}  |  left=foreground  right=background  "
        "b=toggle_box  s=save  r=reset  n=next  p=prev  d=del_point  backspace=del_box  q=quit",
        fontsize=10,
    )

    if META_TEXT is not None:
        gps = _get_exif_gps_lat_lon(PATHS[IDX])
        if gps is None:
            gps_txt = "gps: none"
        else:
            lat, lon = gps
            gps_txt = f"gps: {lat:.6f}, {lon:.6f}"
        total_prompt_pts = len(POINTS) + 4 * len(BOXES)
        META_TEXT.set_text(
            "\n".join(
                [
                    f"image: {name}",
                    gps_txt,
                    f"points: {len(POINTS)}  boxes: {len(BOXES)}  total_prompt_points: {total_prompt_pts}",
                ]
            )
        )

    if WARN_TEXT is not None:
        total_prompt_pts = len(POINTS) + 4 * len(BOXES)
        if total_prompt_pts > 48:
            WARN_TEXT.set_text(f"WARNING: prompt points {total_prompt_pts} > 48")
            WARN_TEXT.set_visible(True)
        else:
            WARN_TEXT.set_visible(False)


def _infer() -> None:
    global MASK
    assert CFG is not None
    assert MODEL is not None
    assert IMAGE_BGR is not None
    print(f"[infer] points={POINTS} labels={LABELS} boxes={BOXES}", file=sys.stderr, flush=True)
    if not POINTS and not BOXES:
        MASK = None
        return
    MASK = segment_with_prompts(
        MODEL,
        IMAGE_BGR,
        POINTS,
        LABELS,
        bboxes_xyxy=BOXES,
        imgsz=CFG.imgsz,
        conf=CFG.conf,
        device=CFG.device,
    )
    try:
        print(f"[infer] mask_none={MASK is None} mask_sum={0 if MASK is None else int(MASK.sum())}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _show() -> None:
    assert FIG is not None
    assert DISP is not None
    assert IMAGE_BGR is not None
    vis = compose_eval_view(IMAGE_BGR, MASK, POINTS, LABELS, boxes_xyxy=BOXES)
    DISP.set_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    FIG.canvas.draw_idle()


def _update_tray() -> None:
    """Show a simple strip of the most recent saved overlays."""
    assert FIG is not None
    if TRAY_DISP is None:
        return
    if not SAVED_PNGS:
        TRAY_DISP.set_data(np.zeros((1, 1, 3), dtype=np.uint8))
        FIG.canvas.draw_idle()
        return

    thumbs: list[np.ndarray] = []
    for p in SAVED_PNGS[-5:]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im_rgb.shape[:2]
        target_h = 90
        new_w = max(1, int(round(w * (target_h / max(1, h)))))
        thumbs.append(cv2.resize(im_rgb, (new_w, target_h), interpolation=cv2.INTER_AREA))

    if not thumbs:
        TRAY_DISP.set_data(np.zeros((1, 1, 3), dtype=np.uint8))
        FIG.canvas.draw_idle()
        return

    pad = 6
    total_w = sum(t.shape[1] for t in thumbs) + pad * (len(thumbs) - 1)
    canvas = np.zeros((thumbs[0].shape[0], total_w, 3), dtype=np.uint8)
    x = 0
    for t in thumbs:
        canvas[:, x : x + t.shape[1], :] = t
        x += t.shape[1] + pad

    TRAY_DISP.set_data(canvas)
    FIG.canvas.draw_idle()


def load_image(i: int) -> None:
    global IDX, IMAGE_BGR, MASK, POINTS, LABELS, BOXES, DRAG_START
    IDX = i
    p = PATHS[IDX]
    arr = cv2.imread(str(p))
    if arr is None:
        raise RuntimeError(f"Could not read image: {p}")
    IMAGE_BGR = arr
    POINTS = []
    LABELS = []
    BOXES = []
    MASK = None
    DRAG_START = None
    _refresh_title()
    if DISP is not None:
        _sync_axes_to_image()
        _show()


def _save_current() -> None:
    global MASK
    assert CFG is not None
    if IMAGE_BGR is None:
        return
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    stem = PATHS[IDX].stem
    ts = utc_timestamp()
    base = CFG.output_dir / f"{stem}_{ts}"
    vis = compose_eval_view(IMAGE_BGR, MASK, POINTS, LABELS, boxes_xyxy=BOXES)
    out_png = base.with_suffix(".png")
    save_bgr_png(out_png, vis)
    meta = build_run_metadata(
        source_path=PATHS[IDX],
        weights_path=CFG.weights_path,
        points=POINTS,
        labels=LABELS,
        extra={"imgsz": CFG.imgsz, "conf": CFG.conf, "device": CFG.device},
    )
    write_json(base.with_suffix(".json"), meta)
    SAVED_PNGS.append(out_png)
    _update_tray()

    # After saving, reset prompts so you can segment another object in the same image.
    POINTS.clear()
    LABELS.clear()
    BOXES.clear()
    MASK = None
    _refresh_title()
    _show()
    print(f"Saved {out_png} and {base.with_suffix('.json')}", file=sys.stderr)


def _norm_box(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
    return (xa, ya, xb, yb)


def on_press(event: MouseEvent) -> None:
    global DRAG_START, DRAG_PATCH
    if _toolbar_mode():
        return
    if not BOX_MODE:
        return
    if AX is None or event.inaxes != AX or event.xdata is None or event.ydata is None:
        return
    if IMAGE_BGR is None:
        return
    h, w = IMAGE_BGR.shape[:2]
    x = float(np.clip(event.xdata, 0, w - 1))
    y = float(np.clip(event.ydata, 0, h - 1))
    DRAG_START = (x, y)
    try:
        from matplotlib.patches import Rectangle

        if DRAG_PATCH is None:
            DRAG_PATCH = Rectangle((x, y), 0, 0, fill=False, edgecolor="cyan", linewidth=2)
            AX.add_patch(DRAG_PATCH)
        else:
            DRAG_PATCH.set_visible(True)
            DRAG_PATCH.set_xy((x, y))
            DRAG_PATCH.set_width(0)
            DRAG_PATCH.set_height(0)
        FIG.canvas.draw_idle()  # type: ignore[union-attr]
    except Exception:
        DRAG_PATCH = None


def on_motion(event: MouseEvent) -> None:
    if not BOX_MODE or DRAG_START is None or DRAG_PATCH is None:
        return
    if AX is None or event.inaxes != AX or event.xdata is None or event.ydata is None:
        return
    if IMAGE_BGR is None:
        return
    h, w = IMAGE_BGR.shape[:2]
    x0, y0 = DRAG_START
    x1 = float(np.clip(event.xdata, 0, w - 1))
    y1 = float(np.clip(event.ydata, 0, h - 1))
    xa, ya, xb, yb = _norm_box(x0, y0, x1, y1)
    DRAG_PATCH.set_xy((xa, ya))
    DRAG_PATCH.set_width(xb - xa)
    DRAG_PATCH.set_height(yb - ya)
    FIG.canvas.draw_idle()  # type: ignore[union-attr]


def on_release(event: MouseEvent) -> None:
    global DRAG_START
    if not BOX_MODE or DRAG_START is None:
        return
    if AX is None or event.inaxes != AX or event.xdata is None or event.ydata is None:
        DRAG_START = None
        return
    if IMAGE_BGR is None:
        DRAG_START = None
        return
    h, w = IMAGE_BGR.shape[:2]
    x0, y0 = DRAG_START
    x1 = float(np.clip(event.xdata, 0, w - 1))
    y1 = float(np.clip(event.ydata, 0, h - 1))
    xa, ya, xb, yb = _norm_box(x0, y0, x1, y1)
    DRAG_START = None

    if (xb - xa) < 3 or (yb - ya) < 3:
        return

    BOXES.append((xa, ya, xb, yb))
    print(f"[box] added {(xa, ya, xb, yb)} boxes={BOXES}", file=sys.stderr, flush=True)
    _refresh_title()
    _infer()
    _show()


def on_click(event: MouseEvent) -> None:
    # If pan/zoom is active, matplotlib will produce confusing xdata/ydata. Ignore clicks until user exits that mode.
    mode = _toolbar_mode()
    if mode:
        print(f"[click] ignored due to toolbar_mode={mode!r}", file=sys.stderr, flush=True)
        return
    if BOX_MODE:
        return
    if AX is None or event.inaxes != AX or event.xdata is None or event.ydata is None:
        print(
            f"[click] ignored inaxes={event.inaxes == AX} xdata={event.xdata} ydata={event.ydata}",
            file=sys.stderr,
            flush=True,
        )
        return
    if IMAGE_BGR is None:
        print("[click] ignored because IMAGE_BGR is None", file=sys.stderr, flush=True)
        return
    if event.button == 1:
        lab = 1
    elif event.button == 3:
        lab = 0
    else:
        print(f"[click] ignored button={event.button}", file=sys.stderr, flush=True)
        return
    h, w = IMAGE_BGR.shape[:2]
    x = float(np.clip(event.xdata, 0, w - 1))
    y = float(np.clip(event.ydata, 0, h - 1))
    print(
        f"[click] raw=({event.xdata:.2f},{event.ydata:.2f}) clipped=({x:.2f},{y:.2f}) "
        f"button={event.button} label={lab}",
        file=sys.stderr,
        flush=True,
    )
    POINTS.append((x, y))
    LABELS.append(lab)
    print(f"[click] points={POINTS} labels={LABELS}", file=sys.stderr, flush=True)
    _refresh_title()
    _infer()
    _show()


def on_key(event: KeyEvent) -> None:
    global MASK, BOX_MODE, DRAG_START
    if not event.key:
        return
    key = event.key.lower()
    if key == "q":
        assert FIG is not None
        plt.close(FIG)
        return
    if key == "r":
        POINTS.clear()
        LABELS.clear()
        BOXES.clear()
        MASK = None
        DRAG_START = None
        _refresh_title()
        _show()
        return
    if key == "b":
        BOX_MODE = not BOX_MODE
        print(f"[mode] BOX_MODE={BOX_MODE}", file=sys.stderr, flush=True)
        _refresh_title()
        FIG.canvas.draw_idle()  # type: ignore[union-attr]
        return
    if key == "d":
        if POINTS:
            pt = POINTS.pop()
            lab = LABELS.pop()
            print(f"[delete] point={pt} label={lab}", file=sys.stderr, flush=True)
            if not POINTS and not BOXES:
                MASK = None
            else:
                _infer()
            _refresh_title()
            _show()
        return
    if key in ("backspace", "delete"):
        if BOXES:
            bx = BOXES.pop()
            print(f"[delete] box={bx}", file=sys.stderr, flush=True)
            if not POINTS and not BOXES:
                MASK = None
            else:
                _infer()
            _refresh_title()
            _show()
        return
    if key == "s":
        _save_current()
        return
    if key == "n":
        if IDX + 1 < len(PATHS):
            load_image(IDX + 1)
        else:
            print("Already on last image.", file=sys.stderr)
        return
    if key == "p":
        if IDX > 0:
            load_image(IDX - 1)
        else:
            print("Already on first image.", file=sys.stderr)
        return


def main() -> None:
    global CFG, MODEL, PATHS, FIG, AX, DISP, TRAY_AX, TRAY_DISP, META_TEXT, WARN_TEXT
    CFG = default_config()
    PATHS = list_test_images(CFG.test_images_dir)
    if not PATHS:
        print(
            f"No images found in {CFG.test_images_dir.resolve()}.\n"
            "Add photos under data/test_images/ (see image_segmentation/README.md).",
            file=sys.stderr,
        )
        sys.exit(1)

    MODEL = load_model(CFG.weights_path)

    FIG = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 1, height_ratios=[8, 1], hspace=0.05)
    AX = FIG.add_subplot(gs[0])
    TRAY_AX = FIG.add_subplot(gs[1])
    plt.subplots_adjust(top=0.93)
    AX.set_axis_off()
    TRAY_AX.set_axis_off()
    _maybe_disable_toolbar()
    META_TEXT = AX.text(
        0.01,
        0.01,
        "",
        transform=AX.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
    )
    WARN_TEXT = AX.text(
        0.5,
        0.98,
        "",
        transform=AX.transAxes,
        ha="center",
        va="top",
        fontsize=16,
        color="yellow",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="red", alpha=0.6, edgecolor="none"),
        visible=False,
    )

    load_image(0)
    assert IMAGE_BGR is not None
    vis0 = compose_eval_view(IMAGE_BGR, MASK, POINTS, LABELS, boxes_xyxy=BOXES)
    DISP = AX.imshow(
        cv2.cvtColor(vis0, cv2.COLOR_BGR2RGB),
        aspect="equal",
        interpolation="nearest",
        origin="upper",
    )
    _sync_axes_to_image()
    TRAY_DISP = TRAY_AX.imshow(np.zeros((1, 1, 3), dtype=np.uint8), aspect="auto", interpolation="nearest")

    FIG.canvas.mpl_connect("button_press_event", on_click)
    FIG.canvas.mpl_connect("button_press_event", on_press)
    FIG.canvas.mpl_connect("motion_notify_event", on_motion)
    FIG.canvas.mpl_connect("button_release_event", on_release)
    FIG.canvas.mpl_connect("key_press_event", on_key)
    _refresh_title()

    plt.show()


if __name__ == "__main__":
    main()
