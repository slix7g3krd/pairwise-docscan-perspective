import pytest
import imaging.docscan_perspective as mod

CASES = [
    ("corner_fallback", (1)),
    ("illum_estimate", (1)),
    ("adaptive_threshold", (1)),
    ("orient_normalize", (1)),
    ("page_sort", (1)),
    ("duplicate_detect", (1)),
    ("text_protect", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("warp", (1)),
    ("metrics", (1)),
    ("shadow_remove", (1)),
    ("binarize", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
