m <- matrix(c(1, 5, NA, 2, 8, 3), nrow = 2, dimnames = list(c("r1", "r2"), c("a", "b", "c")))

# rowSpans / colSpans
rs <- rowSpans(m, na.rm = TRUE)
stopifnot(identical(dim(rs), c(2L, 2L)), identical(unname(rs["r1", ]), c(1, 8)))
stopifnot(is.na(rowSpans(m)["r1", "min"]))
cs <- colSpans(m, na.rm = TRUE)
stopifnot(identical(rownames(cs), c("a", "b", "c")), identical(unname(cs["c", ]), c(3, 8)))

# groupMeans / groupTotals
tx <- data.frame(shop = c("x", "y", "x", "y"), amount = c(10, 5, 20, NA))
stopifnot(identical(as.vector(groupMeans(tx, "amount", "shop")), c(15, 5)))
stopifnot(identical(as.vector(groupTotals(tx, "amount", "shop")), c(30, 5)))
stopifnot(inherits(try(groupTotals(tx, "nope", "shop"), silent = TRUE), "try-error"))

# monthlyTotals
tx$date <- as.Date(c("2024-02-03", "2024-01-15", "2024-02-20", "2024-01-02"))
mt <- monthlyTotals(tx, "amount")
stopifnot(identical(mt$month, c("2024-01", "2024-02")), identical(mt$amount, c(5, 30)))
stopifnot(inherits(try(monthlyTotals(tx, "amount", date = "shop"), silent = TRUE), "try-error"))

# attachLookup keeps row order and warns about unknown keys
lk <- data.frame(shop = c("y", "x"), city = c("Bonn", "Kiel"))
al <- attachLookup(tx, lk, "shop")
stopifnot(identical(al$city, c("Kiel", "Bonn", "Kiel", "Bonn")), identical(al$amount, tx$amount))
w <- tryCatch(attachLookup(tx, lk[1, ], "shop"), warning = function(w) conditionMessage(w))
stopifnot(grepl("1 key", w))

# toWide / toLong
long <- data.frame(id = c(1, 1, 2, 2), wk = c("m1", "m2", "m1", "m2"), y = 1:4)
wide <- toWide(long, "id", "wk", "y")
stopifnot(identical(names(wide), c("id", "y_m1", "y_m2")), identical(wide$y_m2, c(2L, 4L)))
wd <- data.frame(id = 1:2, m9 = c(1, 2), m10 = c(3, 4))
lg <- toLong(wd, "id", c("m9", "m10"))
stopifnot(identical(lg$key, c("m9", "m10", "m9", "m10")), identical(lg$value, c(1, 3, 2, 4)))
