stopifnot(identical(is_valid(c("a@b.co", "a@b", NA), "email"), c(TRUE, FALSE, NA)))
stopifnot(isTRUE(is_valid("x.y+z@mail.ex-ample.org", "email")))
stopifnot(identical(is_valid(c("https://ex.org:8080/a?b=1", "ftp://x.org", "http://"), "url"),
  c(TRUE, FALSE, FALSE)))
stopifnot(identical(is_valid(123456, "phone"), TRUE))
stopifnot(identical(is_valid(c("+4930123456", "12", NA), "phone"), c(TRUE, FALSE, NA)))
stopifnot(identical(is_valid(c("SW1A1AA", "ab1", NA), "postcode"), c(TRUE, FALSE, NA)))
stopifnot(identical(is_valid(c("2024-02-29", "2023-02-29", "2024-13-01"), "birth_date"),
  c(TRUE, FALSE, FALSE)))
stopifnot(identical(is_valid(c("DE", "DEU", NA), "country"), c(TRUE, FALSE, NA)))

d <- data.frame(
  email = c("a@b.co", "bad", NA),
  phone = c("12", "+4912345678", "+4912345678"),
  postcode = c("AB12CD", "AB12CD", "x"),
  birth_date = c("1990-01-01", "1990-02-30", NA),
  country = c("DE", "FR", "france"),
  stringsAsFactors = FALSE
)
fi <- find_invalid(d)
stopifnot(identical(fi$field, c("email", "phone", "postcode", "birth_date", "country")))
stopifnot(identical(fi$row, c(2L, 1L, 3L, 2L, 3L)))
stopifnot(nrow(find_invalid(data.frame(other = 1))) == 0L)
stopifnot(identical(find_invalid(d["phone"])$value, "12"))

vs <- validity_summary(d)
stopifnot(identical(vs$field, names(d)), identical(vs$n_missing, c(1L, 0L, 0L, 1L, 0L)))
stopifnot(identical(vs$n_invalid, rep(1L, 5L)), vs$share_valid[[1]] == 0.5)
stopifnot(identical(validity_summary(d, fields = c("zip", "phone"))$field, "phone"))
stopifnot(is.na(validity_summary(data.frame(email = NA_character_))$share_valid))

msgs <- describe_invalid(fi)
stopifnot(identical(msgs[[1]], "Row 2: `bad` is not a valid email."))
stopifnot(identical(msgs[[4]], "Row 2: `1990-02-30` is not a valid birth date."))
short <- describe_invalid(fi, max_lines = 2L)
stopifnot(length(short) == 3L, identical(short[[3]], "... and 3 more."))
stopifnot(identical(describe_invalid(fi[0, ]), character()))

stopifnot(isTRUE(check_window_args("a", 0, 5, 1, 1)))
err <- function(expr) tryCatch({ expr; "" }, error = conditionMessage)
stopifnot(identical(err(check_window_args(1, 0, 5, 1, 1)), "`x` must be a character vector."))
stopifnot(identical(err(check_window_args("a", -1, 5, 1, 1)), "`min_chars` must not be negative."))
stopifnot(identical(err(check_window_args("a", 3, 2, 1, 1)),
  "`max_chars` must not be smaller than `min_chars`."))
stopifnot(identical(err(check_window_args("a", 0, 5, NA, 1)), "`before` must be a single number."))
stopifnot(identical(err(check_window_args("a", 0, 5, 1, -2)), "`after` must not be negative."))

mw <- match_window(c("the quick brown fox", "none"), "brown", before = 2L, after = 2L)
stopifnot(identical(mw, c("k brown f", NA)))
stopifnot(identical(match_window("abc", "a", before = 5L, after = 0L), "a"))
stopifnot(identical(match_window("abc", "c", before = 1L, after = 5L), "bc"))
