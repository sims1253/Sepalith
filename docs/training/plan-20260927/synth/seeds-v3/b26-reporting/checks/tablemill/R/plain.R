stopifnot(identical(truncateText(c("abcdefgh", "ab", NA), 6), c("abc...", "ab", NA)))
stopifnot(identical(truncateText("abcdefgh", 6, "left"), "...fgh"))
stopifnot(identical(truncateText("abcdefgh", 6, "middle"), "ab...h"))
stopifnot(identical(truncateText("abcdefgh", 5, "middle", ellipsis = "~"), "ab~gh"))
stopifnot(inherits(try(truncateText("abcdef", 3), silent = TRUE), "try-error"))

stopifnot(identical(padText("ab", 4), "ab  "))
stopifnot(identical(padText(7, 3, "right", fill = "0"), "007"))
stopifnot(identical(padText("a", 4, "centre", fill = "."), ".a.."))

w = wrapCell(c("one two three", "", NA), 7)
stopifnot(identical(w[[1]], c("one two", "three")), identical(w[[2]], ""), identical(w[[3]], ""))
stopifnot(identical(wrapCell("abcdefghij", 6)[[1]], "abc..."))

sc = stackCells(c("one two three", "9"), c(7L, 2L), c("left", "right"))
stopifnot(identical(sc, c("one two |  9", "three   |   ")))
stopifnot(identical(stackCells("x", 3L, "centre", sep = ""), " x "))

stopifnot(identical(plainCaption("Hi", 4), c(" Hi ", "----")))
stopifnot(identical(plainCaption("Hi", 3, "below", underline = "~"), c("~~~", "Hi ")))

df = data.frame(name = c("alpha beta gamma", NA), n = c(1, 22))
pt = asPlainText(df, maxWidth = 10L)
stopifnot(identical(pt[1:2], c("name       |  n", "-----------+---")))
stopifnot(identical(pt[3:4], c("alpha beta |  1", "gamma      |   ")))
stopifnot(identical(pt[5], "           | 22"), length(pt) == 5L)
pc = asPlainText(df, maxWidth = 10L, caption = "Counts")
stopifnot(identical(pc[1:2], c("    Counts     ", "---------------")), length(pc) == 7L)

stopifnot(identical(alignDecimal(c("1.5", "12.25", "300", NA)),
                    c("  1.5 ", " 12.25", "300   ", "      ")))
stopifnot(identical(alignDecimal(c("1,5", "10"), mark = ","), c(" 1,5", "10  ")))

rec = data.frame(id = 7L, city = "Oslo", note = NA)
stopifnot(identical(plainRecord(rec), c("id:   7", "city: Oslo", "note: ")))
stopifnot(identical(plainRecord(rec, style = "dots")[2], "city .. Oslo"))
bx = plainRecord(rec, style = "table")
stopifnot(identical(bx[c(1, 3)], c("+------+------+", "| city | Oslo |")), length(bx) == 5L)
stopifnot(inherits(try(plainRecord(rec, row = 2), silent = TRUE), "try-error"))
stopifnot(inherits(try(plainRecord(list(a = 1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(asPlainText(list(a = 1)), silent = TRUE), "try-error"))
