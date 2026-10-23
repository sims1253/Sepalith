stopifnot(identical(trimChars("--a-b--", "-"), "a-b"))
stopifnot(identical(trimChars("..x..", ".", side = "left"), "x.."))
stopifnot(identical(trimChars("]]x]]", "]", side = "right"), "]]x"))
stopifnot(identical(trimChars("^\\x\\^", "^\\"), "x"))

stopifnot(identical(stripAffix(c("tmp_a.csv", "b.txt"), "tmp_", ".csv"), c("a", "b.txt")))
stopifnot(identical(stripAffix("abc", prefix = "a"), "bc"))
stopifnot(identical(stripAffix("abc", suffix = "c"), "ab"))

stopifnot(identical(padText(c("a", "abc"), 3), c("  a", "abc")))
stopifnot(identical(padText("a", 3, "right", pad = "."), "a.."))
stopifnot(identical(padText("a", 4, "both"), " a  "))
stopifnot(identical(padText("toolong", 3), "toolong"))

stopifnot(identical(truncateText(c("abcdefgh", "ab", NA), 5), c("ab...", "ab", NA)))
stopifnot(identical(truncateText("abcdefgh", 5, "head"), "...gh"))
stopifnot(identical(truncateText("abcdefgh", 6, "middle", ellipsis = "~"), "abc~gh"))
stopifnot(inherits(tryCatch(truncateText("abc", 2), error = identity), "error"))
stopifnot(inherits(tryCatch(truncateText("abc", NA), error = identity), "error"))
stopifnot(identical(truncateText("the quick brown fox", 14, "word"), "the quick..."))
stopifnot(identical(truncateText("the quick brown", 12, "word"), "the quick..."))

w = wrapLine("the quick brown fox jumps over", width = 10L)
stopifnot(identical(w, c("the quick", "brown fox", "jumps over")))
w2 = wrapLine("aa bb cc dd", width = 5L, indent = 2L)
stopifnot(identical(w2, c("aa bb", "  cc", "  dd")))
stopifnot(identical(wrapLine("   "), character()))

stopifnot(identical(alignBlock(c("ab", "c"), 4L), c("ab  ", "c   ")))
stopifnot(identical(alignBlock("ab", 4L, "right"), "  ab"))
stopifnot(identical(alignBlock("ab", 5L, "centre"), " ab  "))
stopifnot(identical(alignBlock(c("a b c", "dd e", "end"), 8L, "justify"),
  c("a   b  c", "dd     e", "end     ")))
stopifnot(identical(alignBlock("toolongline", 4L), "toolongline"))

al = alignColumns(c("name,n,note", "a,1,x", "bbbb,22"))
stopifnot(identical(al, c("name   n  note", "----  --  ----", "a      1  x", "bbbb  22")))
stopifnot(identical(alignColumns(c("x|y", "zz|w"), sep = "|", gap = 1L, rule = FALSE),
  c("x  y", "zz w")))
stopifnot(identical(alignColumns("h1,h2", rule = FALSE), "h1  h2"))
