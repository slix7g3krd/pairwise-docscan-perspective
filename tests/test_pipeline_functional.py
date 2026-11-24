"""针对六类线上问题的功能验证（合成图像）。"""
import numpy as np
import pytest

import imaging.docscan_perspective as mod

cv2 = pytest.importorskip("cv2")


def make_gradient_doc(h=200, w=160):
    """渐变光照背景 + 模拟文字笔画（细暗条）。"""
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    img = 120 + 100 * x + 30 * y  # 强渐变背景
    for row in range(30, 170, 20):  # 文字行
        img[row:row + 3, 20:140] = 40.0
    return img.astype(np.float32)


def test_corner_fallback_low_contrast():
    img = np.full((200, 200), 128, np.float32)
    img[30:170, 40:160] = 138  # 极低对比的页面区域
    res = mod.corner_fallback(img)
    assert res["corners"].shape == (4, 2)
    assert res["method"] in {"contour", "hough", "minrect", "frame"}
    # 无论哪级命中，角点都应大致框住页面区域而非崩溃
    assert res["confidence"] > 0


def test_corner_fallback_never_fails_on_garbage():
    for bad in (None, 1, "nonexistent.png", np.zeros((1, 1))):
        res = mod.corner_fallback(bad)
        assert res["corners"] is not None


def test_shadow_remove_preserves_strokes():
    img = make_gradient_doc()
    out = mod.shadow_remove(img, protect=True)
    mask = mod.text_protect(img) > 0
    assert mask.sum() > 0
    # 笔画像素不被拉白：输出不比原图更亮
    assert (out[mask] <= img[mask] + 1e-3).all()
    # 背景被拉平：输出背景的亮度方差应远小于原图
    bg = ~mask
    assert out[bg].std() < img[bg].std()


def test_adaptive_threshold_no_blocks_on_gradient():
    img = make_gradient_doc()
    binary = mod.adaptive_threshold(img)
    mask = mod.text_protect(img) > 0
    # 文字行为黑、背景为白，且全图一致（渐变背景不分块）
    assert (binary[mask] == 0).mean() > 0.9
    for region in (binary[10:60, 10:60], binary[140:190, 100:150]):
        assert (region == 255).mean() > 0.85


def test_orient_normalize_exif_and_batch():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    rot = mod.orient_normalize(img, exif=6, deskew=False)
    assert rot.shape == (4, 3)  # EXIF 6 -> 顺时针 90°
    assert np.array_equal(rot, np.rot90(img, -1))
    batch = mod.orient_normalize([img, img], exif=[6, 3], deskew=False)
    assert batch[0].shape == (4, 3) and batch[1].shape == (3, 4)
    assert np.array_equal(batch[1], np.rot90(img, 2))


def test_orient_normalize_deskew():
    img = np.full((200, 200), 255, np.uint8)
    img[80:120, 40:160] = 0
    m = cv2.getRotationMatrix2D((100, 100), 5.0, 1.0)
    skewed = cv2.warpAffine(img, m, (200, 200), borderValue=(255,))
    fixed = mod.orient_normalize(skewed, deskew=True)
    # 去倾斜后估计角应接近 0
    gray = mod._to_gray(fixed)
    assert abs(mod._deskew_angle(mod._to_uint8(gray))) < 1.5


def test_page_sort_duplex():
    # 6 页双面：扫描序 [1,3,5,6,4,2] -> 重排为 [1,2,3,4,5,6]
    assert mod.page_sort(6) == [0, 5, 1, 4, 2, 3]
    scanned = ["p1", "p3", "p5", "p6", "p4", "p2"]
    assert mod.page_sort(scanned) == ["p1", "p2", "p3", "p4", "p5", "p6"]
    # 奇数页：扫描序 [1,3,5,4,2]
    assert mod.page_sort(["p1", "p3", "p5", "p4", "p2"]) == ["p1", "p2", "p3", "p4", "p5"]
    assert mod.page_sort(4, pattern="simplex") == [0, 1, 2, 3]


def test_duplicate_detect():
    a = make_gradient_doc()
    b = a.copy()
    c = np.roll(a, 3, axis=0)  # 内容不同
    noisy = a + 1.5  # 近重复（整体微亮）
    groups = mod.duplicate_detect([a, b, c, noisy])
    flat = [i for g in groups for i in g]
    assert 0 in flat and 1 in flat
    assert 2 not in flat
    # 完全相同的字节输入
    assert mod.duplicate_detect([b"x", b"x", b"y"]) == [[0, 1]]


def test_warp_and_determinism():
    img = make_gradient_doc()
    warped = mod.warp(img, corners=[[10, 10], [150, 5], [155, 195], [5, 190]])
    assert warped.ndim == 2 and min(warped.shape) >= 8
    assert mod.determinism(img)["deterministic"]


def test_audit_and_validation():
    report = mod.audit(make_gradient_doc())
    assert report["ok"] and report["corner_method"]
    assert mod.validation(report)["ok"]
    assert not mod.validation(1)["ok"]


def test_fallback_chain():
    res = mod.fallback(lambda: None, lambda: 42)
    assert res["value"] == 42 and res["index"] == 1
    res = mod.fallback(1, lambda: (_ for _ in ()).throw(ValueError("x")))
    assert res["value"] is None and len(res["errors"]) == 2
