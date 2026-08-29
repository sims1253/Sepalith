"""
Tests for token_patterns.py — synthetic corpus with known counts.
Run: uv run python test_token_patterns.py  (or pytest).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_patterns import PATTERNS, coverage, iter_files, shortlist


def _make_corpus(tmp):
    d1 = os.path.join(tmp, "pkgA")
    os.makedirs(d1)
    with open(os.path.join(d1, "a.R"), "w") as f:
        # 3 pipes, 2 assigns, 1 $, 1 %in%
        f.write("x %>% y %>% z %>% w\n")
        f.write("a <- 1; b <<- 2\n")
        f.write("df$col; v %in% c(1,2)\n")
    with open(os.path.join(tmp, "b.R"), "w") as f:
        # 2 more pipes, 1 [[, 1 ::
        f.write("x %>% y %>% z\nl[[1]]; stats::median(l)\n")
    return [os.path.join(d1, "a.R"), os.path.join(tmp, "b.R")]


def test_known_counts():
    with tempfile.TemporaryDirectory() as tmp:
        files = _make_corpus(tmp)
        cov = coverage(files)
        by = {r["pattern"]: r for r in cov["rows"]}
        assert by["%>%"]["count"] == 5
        # documented upper-bound behavior: "<-" also matches inside "<<-"
        assert by["<-"]["count"] == 2
        assert by["<<-"]["count"] == 1
        assert by["$"]["count"] == 1
        assert by["%in%"]["count"] == 1
        assert by["[["]["count"] == 1
        assert by["::"]["count"] == 1  # ":::" absent → no substring inflation here


def test_byte_share_bounded_and_sorted():
    with tempfile.TemporaryDirectory() as tmp:
        cov = coverage(_make_corpus(tmp))
        assert 0.0 < sum(r["byte_share"] for r in cov["rows"]) < 1.0
        shares = [r["byte_share"] for r in cov["rows"]]
        assert shares == sorted(shares, reverse=True)
        assert cov["total_bytes"] > 0


def test_shortlist_min_count():
    with tempfile.TemporaryDirectory() as tmp:
        cov = coverage(_make_corpus(tmp))
        assert "%>%" in shortlist(cov, min_count=5)
        assert "%>%" not in shortlist(cov, min_count=6)
        assert shortlist(cov, min_count=1000) == []


def test_iter_files_no_match_is_loud():
    try:
        iter_files(["/nonexistent/**/*.R"])
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "no files matched" in str(e)


def test_patterns_complete():
    # every entry is (tag, str) and no duplicates
    pats = [p for _, p in PATTERNS]
    assert len(pats) == len(set(pats))
    assert all(isinstance(t, str) and isinstance(p, str) for t, p in PATTERNS)
    assert "%>%" in pats and "<-" in pats and "!!!" in pats


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests green")
