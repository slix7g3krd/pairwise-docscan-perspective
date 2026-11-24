"""文档扫描透视校正流水线。

针对以下线上问题的修复实现：

1. 低对比边缘角点检测失败   -> corner_fallback: 轮廓 -> Hough 直线 -> minAreaRect -> 整帧 四级回退
2. 去阴影把文字笔画压掉     -> shadow_remove + text_protect: 笔画掩膜保护，暗部不被拉白
3. 局部阈值在渐变背景成块   -> adaptive_threshold: 先光照 flatten 再 Sauvola 连续局部阈值
4. 批量处理方向元数据乱转   -> orient_normalize: EXIF 归一 + 去倾斜，输出方向一致
5. 双面扫描页面顺序错乱     -> page_sort: 正面叠 + 反面逆序的交错重排
6. 重复页面                 -> duplicate_detect: 感知哈希 + 精确哈希分组

所有函数确定性、无随机源；对非法输入返回文档化的安全默认值而不是抛异常。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:  # OpenCV 缺失时退化为安全默认值，不影响导入
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

_EPS = 1e-6


# ---------------------------------------------------------------------------
# 输入规整
# ---------------------------------------------------------------------------

def _to_gray(image):
    """把输入规整为 float32 灰度图 (0..255)。非法输入返回 None。"""
    if image is None:
        return None
    if isinstance(image, (str, Path)):
        if cv2 is None:
            return None
        arr = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        return None if arr is None else arr.astype(np.float32)
    arr = np.asarray(image)
    if arr.ndim == 0 or arr.size == 0:
        return None
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            if cv2 is not None:
                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                arr = arr[..., 0] * 0.114 + arr[..., 1] * 0.587 + arr[..., 2] * 0.299
        else:
            arr = arr[..., 0]
    if arr.ndim != 2 or min(arr.shape) < 2:
        return None
    return arr.astype(np.float32)


def _to_uint8(gray):
    return np.clip(gray, 0, 255).astype(np.uint8)


def _odd(k):
    k = int(round(k))
    return k if k % 2 == 1 else k + 1


def _unit_square():
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)


def _order_quad(pts):
    """四角点排序为 [左上, 右上, 右下, 左下]。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],  # 左上: x+y 最小
        pts[np.argmax(d)],  # 右上: x-y 最大
        pts[np.argmax(s)],  # 右下: x+y 最大
        pts[np.argmin(d)],  # 左下: x-y 最小
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. 角点检测 + 回退链
# ---------------------------------------------------------------------------

def _quad_sane(quad, h, w, min_ratio=0.15):
    if cv2 is None:
        return False
    area = cv2.contourArea(np.asarray(quad, dtype=np.float32))
    return area > min_ratio * h * w


def _corners_from_contour(gray8):
    """多级 Canny 阈值 + 最大四边形轮廓。低对比边缘靠逐步降低阈值兜底。"""
    blur = cv2.GaussianBlur(gray8, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    best_cnt, best_area = None, 0.0
    for low in (40, 25, 12):  # 逐档降低，覆盖低对比边缘
        edges = cv2.dilate(cv2.Canny(blur, low, low * 3), kernel)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area > best_area:
                best_cnt, best_area = c, area
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return _order_quad(approx.reshape(4, 2)), area
    return None, (best_cnt if best_cnt is not None else None)


def _corners_from_hough(gray8):
    """Hough 直线：取最上/下/左/右的支配直线求交点。"""
    edges = cv2.Canny(gray8, 20, 60)
    h, w = gray8.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(20, min(h, w) // 8),
                            minLineLength=min(h, w) // 4,
                            maxLineGap=min(h, w) // 20)
    if lines is None:
        return None
    hor, ver = [], []
    for ln in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = map(float, ln)
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        if ang < 30 or ang > 150:
            hor.append((x1, y1, x2, y2))
        elif 60 < ang < 120:
            ver.append((x1, y1, x2, y2))
    if len(hor) < 2 or len(ver) < 2:
        return None
    hor.sort(key=lambda l: (l[1] + l[3]) / 2)
    ver.sort(key=lambda l: (l[0] + l[2]) / 2)
    top, bot = hor[0], hor[-1]
    left, right = ver[0], ver[-1]

    def intersect(l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < _EPS:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        return px, py

    pts = [intersect(top, left), intersect(top, right),
           intersect(bot, right), intersect(bot, left)]
    if any(p is None for p in pts):
        return None
    quad = np.array(pts, dtype=np.float32)
    margin = 0.25 * max(h, w)  # 允许适度越界（透视下常见）
    if (quad[:, 0] < -margin).any() or (quad[:, 0] > w + margin).any() \
            or (quad[:, 1] < -margin).any() or (quad[:, 1] > h + margin).any():
        return None
    return _order_quad(quad)


def corner_fallback(image, min_area_ratio=0.15):
    """检测文档四角，失败时逐级回退，保证总有可用输出。

    返回 {"corners": (4,2) float32 [tl,tr,br,bl], "method": str, "confidence": float}
    method: contour > hough > minrect > frame > default
    """
    gray = _to_gray(image)
    if gray is None or cv2 is None:
        return {"corners": _unit_square(), "method": "default", "confidence": 0.0}
    h, w = gray.shape
    gray8 = _to_uint8(gray)
    frame = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    # 第 1 级：四边形轮廓
    quad, extra = _corners_from_contour(gray8)
    if quad is not None and _quad_sane(quad, h, w, min_area_ratio):
        conf = float(cv2.contourArea(quad) / (h * w))
        return {"corners": quad, "method": "contour", "confidence": min(1.0, conf)}

    # 第 2 级：Hough 直线求交（边缘断裂但直线特征仍在时有效）
    quad = _corners_from_hough(gray8)
    if quad is not None and _quad_sane(quad, h, w, min_area_ratio):
        return {"corners": quad, "method": "hough", "confidence": 0.6}

    # 第 3 级：边缘点集 minAreaRect（四边形不闭合时）
    cnt = extra if isinstance(extra, np.ndarray) else None
    pts = cnt if cnt is not None else np.column_stack(
        np.nonzero(cv2.Canny(gray8, 15, 45))[::-1]).astype(np.float32)
    if pts is not None and len(pts) >= 4:
        rect = cv2.minAreaRect(pts)
        quad = _order_quad(cv2.boxPoints(rect))
        if _quad_sane(quad, h, w, min_area_ratio):
            return {"corners": quad, "method": "minrect", "confidence": 0.4}

    # 第 4 级：整帧内缩，绝不失败
    inset = np.array([[2, 2], [w - 3, 2], [w - 3, h - 3], [2, h - 3]], dtype=np.float32)
    return {"corners": inset if min(h, w) > 8 else frame,
            "method": "frame", "confidence": 0.1}


def warp(image, corners=None, out_size=None, pad=0):
    """按四角点做透视校正。corners 缺省时走 corner_fallback。"""
    src = np.asarray(image) if not isinstance(image, (str, Path)) else None
    gray = _to_gray(image)
    if gray is None or cv2 is None:
        return image  # 无法处理时原样返回，不抛异常
    if src is None:
        src = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if corners is None:
        corners = corner_fallback(image)["corners"]
    quad = _order_quad(corners)
    if out_size is None:
        w = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
        h = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
        out_size = (max(8, w), max(8, h))
    w, h = out_size
    dst = np.array([[pad, pad], [w - 1 - pad, pad],
                    [w - 1 - pad, h - 1 - pad], [pad, h - 1 - pad]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(src, m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# 2. 光照估计 + 文字保护的去阴影
# ---------------------------------------------------------------------------

def illum_estimate(image, downscale=64):
    """估计逐像素光照背景（低频分量）。

    低分辨率高斯模糊后上采样：对渐变光照平滑跟随，不含文字笔画细节。
    返回与输入同尺寸的 float32 背景亮度图。
    """
    gray = _to_gray(image)
    if gray is None:
        return np.ones((1, 1), dtype=np.float32)
    if cv2 is None:
        return np.full_like(gray, float(np.mean(gray)))
    h, w = gray.shape
    scale = min(1.0, downscale / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    k = _odd(max(3, min(small.shape) // 4))
    small = cv2.GaussianBlur(small, (k, k), 0, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def text_protect(image, delta_frac=0.15, min_delta=10.0):
    """生成文字笔画保护掩膜 (uint8 0/255)。

    笔画 = 显著暗于局部光照背景的像素。去阴影/二值化时这些像素不被拉白。
    """
    gray = _to_gray(image)
    if gray is None:
        return np.zeros((1, 1), dtype=np.uint8)
    bg = illum_estimate(gray)
    delta = np.maximum(min_delta, delta_frac * bg)
    return np.where(gray < bg - delta, 255, 0).astype(np.uint8)


def shadow_remove(image, protect=True):
    """除法去阴影：gray / 光照背景 * 255。

    protect=True 时，文字笔画像素取 min(结果, 原图)，保证笔画不被压白。
    """
    gray = _to_gray(image)
    if gray is None:
        return np.asarray(image, dtype=np.float32).reshape(1, -1)
    bg = illum_estimate(gray)
    out = np.clip(gray / (bg + _EPS) * 255.0, 0, 255)
    if protect:
        mask = text_protect(gray) > 0
        out[mask] = np.minimum(out[mask], gray[mask])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. 渐变背景鲁棒的自适应阈值
# ---------------------------------------------------------------------------

def adaptive_threshold(image, k=0.2, r=128.0, window=None, flatten=True):
    """Sauvola 局部阈值，先 flatten 光照再阈值，避免渐变背景成块。

    flatten: 先除以光照估计，消除大尺度亮度渐变；
    局部均值/方差用 boxFilter 连续滑动窗（非硬分块），无块边界伪影。
    返回 uint8 二值图 (文字=0, 背景=255)。
    """
    gray = _to_gray(image)
    if gray is None:
        return np.full((1, 1), 255, dtype=np.uint8)
    flat = shadow_remove(gray, protect=True) if flatten else gray
    if cv2 is None:  # 无 cv2 时退化为全局 Otsu 的近似：中值阈值
        t = float(np.median(flat))
        return np.where(flat > t, 255, 0).astype(np.uint8)
    h, w = flat.shape
    win = window or int(np.clip(_odd(min(h, w) // 16), 15, 65))
    win = _odd(win)
    mean = cv2.boxFilter(flat, cv2.CV_32F, (win, win), borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.boxFilter(flat * flat, cv2.CV_32F, (win, win),
                           borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0.0))
    thresh = mean * (1.0 + k * (std / r - 1.0))
    return np.where(flat > thresh, 255, 0).astype(np.uint8)


def binarize(image, **kw):
    """完整二值化流水线：去阴影(含文字保护) + 光照平坦化自适应阈值。"""
    return adaptive_threshold(image, flatten=True, **kw)


# ---------------------------------------------------------------------------
# 4. 方向归一（EXIF + 去倾斜），批量输出方向一致
# ---------------------------------------------------------------------------

_EXIF_TRANSFORM = {
    1: lambda x: x,
    2: np.fliplr,
    3: lambda x: np.rot90(x, 2),
    4: np.flipud,
    5: lambda x: np.swapaxes(x, 0, 1),
    6: lambda x: np.rot90(x, -1),  # 存储为逆时针 90°，需顺时针转回
    7: lambda x: np.rot90(np.swapaxes(x, 0, 1), 2),
    8: lambda x: np.rot90(x, 1),
}


def _deskew_angle(gray8, max_angle=15.0):
    """由深色像素点集 minAreaRect 估计倾斜角；估计失败返回 0。"""
    if cv2 is None:
        return 0.0
    dark = gray8 < np.median(gray8) - 20
    ys, xs = np.nonzero(dark)
    if len(xs) < 50:
        return 0.0
    rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    angle = rect[2]
    if angle > 45:
        angle -= 90
    return angle if abs(angle) <= max_angle else 0.0


def orient_normalize(image, exif=None, deskew=True):
    """方向归一：先按 EXIF 方向转正，再小角度去倾斜。

    - 输入 list/tuple 时批量处理，返回同序 list（批量场景方向输出一致）。
    - exif: 1..8 的 EXIF Orientation；为 None 时若输入是路径/PIL 图则自动读取。
    - 无法处理的输入原样返回，不抛异常。
    """
    if isinstance(image, (list, tuple)):
        metas = exif if isinstance(exif, (list, tuple)) else [exif] * len(image)
        return [orient_normalize(im, exif=m, deskew=deskew)
                for im, m in zip(image, metas)]

    arr = image
    if isinstance(image, (str, Path)):
        try:
            from PIL import Image, ImageOps
            with Image.open(image) as im:
                arr = np.asarray(ImageOps.exif_transpose(im))  # 自动应用 EXIF
            exif = None  # 已应用，避免二次旋转
        except Exception:
            if cv2 is not None:
                arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            else:
                return image
    if arr is None:
        return image
    arr = np.asarray(arr)
    if arr.ndim < 2 or arr.size == 0:
        return image

    if exif in _EXIF_TRANSFORM:
        arr = _EXIF_TRANSFORM[exif](arr)
        arr = np.ascontiguousarray(arr)

    if deskew and cv2 is not None:
        gray = _to_gray(arr)
        if gray is not None:
            angle = _deskew_angle(_to_uint8(gray))
            if abs(angle) > 0.05:
                h, w = gray.shape
                m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                border = 255 if arr.ndim == 2 else (255,) * arr.shape[2]
                arr = cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return arr


# ---------------------------------------------------------------------------
# 5. 双面扫描页面重排
# ---------------------------------------------------------------------------

def _duplex_perm(n):
    """双面扫描顺序 -> 页码顺序的索引排列。

    扫描仪先扫所有正面 (1,3,5,...)，翻叠后反面逆序输出 (...,6,4,2)，
    返回把扫描序列重排为 1..n 页序的索引列表。
    """
    fronts = (n + 1) // 2
    page_no = [2 * i + 1 if i < fronts else 2 * (n - i) for i in range(n)]
    return sorted(range(n), key=lambda i: page_no[i])


def page_sort(pages, pattern="duplex"):
    """页面重排。

    - pages 为 int n: 返回重排索引排列；
    - pages 为序列: 返回重排后的同类型序列。
    pattern: "duplex" 正面叠+反面逆序交错 | "reverse" 整体逆序 | "simplex" 不变。
    """
    if isinstance(pages, (int, np.integer)):
        n = int(pages)
        if n <= 0:
            return []
        if pattern == "duplex":
            return _duplex_perm(n)
        if pattern == "reverse":
            return list(range(n - 1, -1, -1))
        return list(range(n))
    try:
        items = list(pages)
    except TypeError:
        return pages
    perm = page_sort(len(items), pattern=pattern)
    return [items[i] for i in perm]


# ---------------------------------------------------------------------------
# 6. 重复页面检测
# ---------------------------------------------------------------------------

def _ahash(gray, size=16):
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA) \
        if cv2 is not None else gray
    small = np.asarray(small, dtype=np.float32)
    return small > small.mean()


def _page_fingerprint(page, size=16):
    """返回 (md5_hex 或 None, ahash 布尔矩阵 或 None)。"""
    if isinstance(page, (str, Path)):
        data = Path(page).read_bytes()
        gray = _to_gray(page)
        return hashlib.md5(data).hexdigest(), (_ahash(gray, size) if gray is not None else None)
    if isinstance(page, (bytes, bytearray)):
        return hashlib.md5(bytes(page)).hexdigest(), None
    gray = _to_gray(page)
    if gray is None:
        return None, None
    return None, _ahash(gray, size)


def duplicate_detect(pages, hash_size=16, max_hamming=6):
    """检测重复/近重复页面，返回重复组列表（每组为 >=2 个下标，按下标升序）。

    精确重复用 md5（路径/字节输入）；近重复用感知哈希汉明距离。
    """
    try:
        items = list(pages)
    except TypeError:
        return []
    n = len(items)
    if n < 2:
        return []
    fps = [_page_fingerprint(p, hash_size) for p in items]

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            mi, hi = fps[i]
            mj, hj = fps[j]
            if mi is not None and mj is not None and mi == mj:
                union(i, j)
            elif hi is not None and hj is not None and hi.shape == hj.shape:
                if int(np.count_nonzero(hi != hj)) <= max_hamming:
                    union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted((g for g in groups.values() if len(g) > 1), key=lambda g: g[0])


# ---------------------------------------------------------------------------
# 通用：回退执行器 / 指标 / 校验 / 审计 / 确定性
# ---------------------------------------------------------------------------

def fallback(*fns):
    """依次尝试 callable，返回第一个非 None 结果。

    返回 {"value", "index", "errors"}；全部失败时 value=None, index=-1。
    """
    errors = []
    for i, fn in enumerate(fns):
        if not callable(fn):
            errors.append(f"arg{i} not callable: {type(fn).__name__}")
            continue
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - 回退链需要吞掉单级失败
            errors.append(f"arg{i} raised: {exc!r}")
            continue
        if value is not None:
            return {"value": value, "index": i, "errors": errors}
    return {"value": None, "index": -1, "errors": errors}


def metrics(image):
    """图像质量指标：对比度 / 锐度 / 墨迹覆盖率。"""
    gray = _to_gray(image)
    if gray is None:
        return {"valid": False, "contrast": 0.0, "sharpness": 0.0, "ink_coverage": 0.0}
    sharp = float(cv2.Laplacian(_to_uint8(gray), cv2.CV_32F).var()) if cv2 is not None else 0.0
    ink = float(np.mean(binarize(gray) == 0))
    return {"valid": True, "contrast": float(gray.std()),
            "sharpness": sharp, "ink_coverage": ink}


def validation(image, min_contrast=10.0, max_ink=0.6, min_ink=0.0005):
    """校验扫描结果是否可用。输入审计 dict 时校验其字段，否则现场计算。"""
    if isinstance(image, dict):
        checks = {
            "has_corners": "corners" in image or "corner_method" in image,
            "has_metrics": "metrics" in image or "metrics_after" in image,
        }
        return {"ok": all(checks.values()), "checks": checks}
    m = metrics(image)
    if not m["valid"]:
        return {"ok": False, "checks": {"valid_input": False}, "metrics": m}
    checks = {
        "valid_input": True,
        "contrast_ok": m["contrast"] >= min_contrast,
        "not_blank": m["ink_coverage"] >= min_ink,
        "not_saturated": m["ink_coverage"] <= max_ink,
    }
    return {"ok": all(checks.values()), "checks": checks, "metrics": m}


def audit(image, **kw):
    """跑完整流水线并输出审计报告（哪级回退被触发、前后指标对比）。"""
    gray = _to_gray(image)
    if gray is None:
        return {"ok": False, "reason": "invalid input"}
    corners = corner_fallback(gray)
    binary = binarize(gray)
    report = {
        "ok": True,
        "shape": list(gray.shape),
        "corner_method": corners["method"],
        "corner_confidence": corners["confidence"],
        "corners": corners["corners"].tolist(),
        "metrics_before": metrics(gray),
        "metrics_after": metrics(binary),
    }
    report["validation"] = validation(report)
    return report


def determinism(image=None, runs=2):
    """验证流水线确定性：同一输入多次运行输出逐字节一致。"""
    gray = _to_gray(image)
    if gray is None:  # 无输入时用固定合成图自测
        rng_y, rng_x = np.mgrid[0:64, 0:64].astype(np.float32)
        gray = 200.0 + 30.0 * np.sin(rng_x / 9.0) + 20.0 * np.cos(rng_y / 7.0)
        gray[20:24, 10:50] = 60.0
    hashes = []
    for _ in range(max(2, runs)):
        out = binarize(gray)
        hashes.append(hashlib.md5(out.tobytes()).hexdigest())
    return {"deterministic": len(set(hashes)) == 1, "hash": hashes[0], "runs": len(hashes)}
