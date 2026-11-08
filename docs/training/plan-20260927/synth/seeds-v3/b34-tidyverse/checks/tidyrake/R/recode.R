# skipped: needs dplyr (band_amount, classify_status, flag_orders)

# Helper defined in columns.R
check_known_cols <- function(data, cols, arg) {
  if (length(setdiff(cols, names(data))) > 0L) stop("unknown columns")
  invisible(data)
}

lp <- label_priority(c(95, 70, 41, 10, 3, NA))
stopifnot(identical(lp, c("critical", "high", "medium", "low", "none", NA)))
th <- c(critical = 60, high = 55, medium = 50, low = 1)
stopifnot(identical(label_priority(50, th), "medium"))
stopifnot(inherits(try(label_priority("x"), silent = TRUE), "try-error"))

sv <- data.frame(q1 = c("Yes", " n", "maybe"), q2 = c("agree", "Neutral", "x"),
                 q3 = c("never", "ALWAYS", ""), q4 = c("***", "*****", "-"),
                 stringsAsFactors = FALSE)
stopifnot(identical(recode_answers(sv, "q1", "yes_no")$q1, c(TRUE, FALSE, NA)))
a5 <- recode_answers(sv, "q2", "agree5")$q2
stopifnot(is.ordered(a5), identical(as.integer(a5), c(4L, 3L, NA)))
f4 <- recode_answers(sv, "q3", "freq4")$q3
stopifnot(identical(as.character(f4), c("never", "always", NA)), nlevels(f4) == 4L)
stopifnot(identical(recode_answers(sv, "q4", "stars")$q4, c(3L, 5L, NA)))

cl <- collapse_levels(c("a", "a", "a", "b", "b", "c", NA), min_n = 2L)
stopifnot(identical(levels(cl), c("a", "b", "other")), identical(as.character(cl[6]), "other"))
stopifnot(is.na(cl[7]), identical(levels(collapse_levels(c("a", "b"), 1L)), c("a", "b")))
