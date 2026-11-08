stopifnot(identical(standardise_names(c(" Order ID", "order-id", "__")),
                    c("order_id", "order_id_1", "column")))
stopifnot(identical(standardise_names("Amount (EUR)"), "amount_eur"))

df <- data.frame(a = 1, b = "x")
stopifnot(identical(check_known_cols(df, c("a", "b"), "cols"), df))
stopifnot(inherits(try(check_known_cols(df, "z", "cols"), silent = TRUE), "try-error"))

raw <- data.frame(i = c("1", "2"), n = c("1.5", "2"), l = c("TRUE", "false"),
                  d = c("2024-01-02", "2024-02-03"), ch = 1:2, stringsAsFactors = FALSE)
co <- coerce_columns(raw, integer = "i", numeric = "n", logical = "l",
                     date = "d", character = "ch")
stopifnot(is.integer(co$i), is.numeric(co$n), identical(co$l, c(TRUE, FALSE)))
stopifnot(inherits(co$d, "Date"), identical(co$ch, c("1", "2")))
stopifnot(inherits(try(coerce_columns(raw, date = "zz"), silent = TRUE), "try-error"))

tx <- data.frame(a = c(" x ", "  ", "A   b", "N/A"), b = 1:4, stringsAsFactors = FALSE)
tr <- tidy_text_columns(tx)
stopifnot(identical(tr$a, c("x", NA, "A b", NA)), identical(tr$b, 1:4))
stopifnot(identical(attr(tr, "changed"), c(a = 4L)))
tr2 <- tidy_text_columns(tx, "a", squish = FALSE, lower = TRUE, blank_as_na = FALSE)
stopifnot(identical(tr2$a, c("x", "", "a   b", NA)))
tr3 <- tidy_text_columns(tx, "a", trim = FALSE, squish = FALSE, na_strings = NULL)
stopifnot(identical(tr3$a, c(" x ", "  ", "A   b", "N/A")), attr(tr3, "changed")[["a"]] == 0L)

rep_df <- column_report(data.frame(a = c(NA, 2, 2), b = c("p", "q", NA)))
stopifnot(identical(rep_df$column, c("a", "b")), identical(rep_df$type, c("numeric", "character")))
stopifnot(identical(rep_df$n_missing, c(1L, 1L)), identical(rep_df$n_unique, c(1L, 2L)))
stopifnot(identical(rep_df$example, c("2", "p")))

dd <- data.frame(id = c("a", "b", "c"), e = NA, k = 1, v = c(1, 2, 2), w = c(1, 2, 2),
                 stringsAsFactors = FALSE)
stopifnot(identical(attr(drop_columns_by_rule(dd, "empty"), "dropped"), "e"))
stopifnot(identical(attr(drop_columns_by_rule(dd, "constant"), "dropped"), c("e", "k")))
stopifnot(identical(attr(drop_columns_by_rule(dd, "duplicate"), "dropped"), "w"))
stopifnot(identical(names(drop_columns_by_rule(dd, "id_like")), c("e", "k", "v", "w")))
stopifnot(identical(attr(drop_columns_by_rule(dd, "constant", keep = "k"), "dropped"), "e"))

fm <- data.frame(x = c(1, NA, 3, NA), y = c("a", "a", NA, "b"), stringsAsFactors = FALSE)
stopifnot(identical(fill_missing(fm, "x", "zero")$x, c(1, 0, 3, 0)))
stopifnot(identical(fill_missing(fm, "x", "mean")$x, c(1, 2, 3, 2)))
stopifnot(identical(fill_missing(fm, "x", "median")$x, c(1, 2, 3, 2)))
stopifnot(identical(fill_missing(fm, "y", "mode")$y, c("a", "a", "a", "b")))
stopifnot(identical(fill_missing(fm, c("x", "y"), "locf")$x, c(1, 1, 3, 3)))
stopifnot(identical(fill_missing(fm, "y", "locf")$y, c("a", "a", "a", "b")))
