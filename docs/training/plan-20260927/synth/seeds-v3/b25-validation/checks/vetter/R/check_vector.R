msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
cls <- function(expr) class(tryCatch(expr, error = function(e) e))[1]

stopifnot(identical(cls(abort_arg("a", "%s is bad.")), "vetter_error"))
stopifnot(identical(msg(abort_arg("a", "%s is %d.", 3L)), "`a` is 3."))

stopifnot(identical(check_length(1:3, min_len = 1), 1:3))
stopifnot(identical(msg(check_length(1:2, "v", min_len = 3)),
                    "`v` must have at least 3 elements, not 2."))
stopifnot(identical(msg(check_length(1:5, "v", max_len = 4)),
                    "`v` must have at most 4 elements, not 5."))

stopifnot(identical(check_numeric(c(1, 2)), c(1, 2)))
stopifnot(identical(msg(check_numeric("a", "v")), "`v` must be numeric, not character."))
stopifnot(identical(msg(check_numeric(c(1, NA), "v")),
                    "`v` must not contain missing values (found 1)."))
stopifnot(identical(check_numeric(c(1, NA), allow_na = TRUE), c(1, NA)))
stopifnot(identical(msg(check_numeric(c(1, -2), "v", lower = 0)),
                    "`v` must be at least 0; element 2 is -2."))
stopifnot(identical(msg(check_numeric(c(9, 1), "v", upper = 5)),
                    "`v` must be at most 5; element 1 is 9."))

stopifnot(identical(check_character(c("ab", "c")), c("ab", "c")))
stopifnot(identical(msg(check_character(1, "v")), "`v` must be character, not numeric."))
stopifnot(identical(msg(check_character(c("ab", "c"), "v", min_chars = 2)),
                    "`v` must have at least 2 characters; element 2 has 1."))
stopifnot(identical(msg(check_character(c("ab", "cde"), "v", max_chars = 2)),
                    "`v` must have at most 2 characters; element 2 has 3."))

stopifnot(identical(check_sorted(c(1, 2, 2)), c(1, 2, 2)))
stopifnot(identical(msg(check_sorted(c(1, 2, 2), "v", strict = TRUE)),
                    "`v` must be increasing; element 3 breaks the order."))
stopifnot(identical(check_sorted(c(3, 1), decreasing = TRUE), c(3, 1)))
stopifnot(identical(msg(check_sorted(c(3, 1, 2), "v", decreasing = TRUE)),
                    "`v` must be decreasing; element 3 breaks the order."))
stopifnot(identical(check_sorted(5), 5))

stopifnot(identical(check_unique(c("a", NA, NA)), c("a", NA, NA)))
stopifnot(identical(msg(check_unique(c("a", "b", "a"), "v")),
                    "`v` must be unique; a appears more than once."))
stopifnot(identical(cls(check_unique(c(NA, NA), ignore_na = FALSE)), "vetter_error"))

stopifnot(identical(check_vector(c(1, 2), "numeric"), c(1, 2)))
stopifnot(identical(check_vector("a", "character"), "a"))
stopifnot(identical(check_vector(TRUE, "logical"), TRUE))
stopifnot(identical(msg(check_vector(1, "logical", "f")), "`f` must be logical, not numeric."))
stopifnot(identical(msg(check_vector("a", "factor", "f")),
                    "`f` must be factor, not character."))
stopifnot(identical(check_vector(c("a", "b"), "id"), c("a", "b")))
stopifnot(identical(cls(check_vector(c("a", "a"), "id")), "vetter_error"))
stopifnot(identical(msg(check_vector(numeric(0), "numeric", "v")),
                    "`v` must have at least 1 elements, not 0."))

stopifnot(identical(check_vector(c(1, 2), "integer"), c(1, 2)))
stopifnot(identical(msg(check_vector(1.5, "integer", "n")),
                    "`n` must contain whole numbers only."))
stopifnot(identical(check_vector(as.Date("2024-01-01"), "date"), as.Date("2024-01-01")))
stopifnot(identical(msg(check_vector("2024", "date", "d")), "`d` must be Date, not character."))

p <- check_paired(c(1, 2, 3), c(2, 4, 6))
stopifnot(identical(names(p), c("x", "y", "weights", "n")), identical(p$weights, c(1, 1, 1)))
stopifnot(identical(p$n, 3L))
stopifnot(identical(check_paired(c(1, NA), c(NA, 2), allow_na = TRUE)$n, 2L))
stopifnot(identical(msg(check_paired(c(NA_real_, NA), c(1, 2), allow_na = TRUE)),
                    "`x` must contain at least one observed value."))
stopifnot(identical(msg(check_paired(c(1, 2), c(NA_real_, NA), allow_na = TRUE)),
                    "`y` must contain at least one observed value."))
stopifnot(identical(msg(check_paired(1, 1)), "`x` must have at least 2 elements, not 1."))
stopifnot(identical(msg(check_paired(1:2, c(1, 2, 3))),
                    "`y` must have the same length as `x` (2), not 3."))
stopifnot(identical(msg(check_paired(1:2, 1:2, weights = c(0, 0))),
                    "`weights` must not all be zero."))
stopifnot(identical(msg(check_paired(1:2, 1:2, weights = 1)),
                    "`weights` must have the same length as `x` (2), not 1."))
stopifnot(identical(cls(check_paired(1:2, 1:2, weights = c(-1, 2))), "vetter_error"))
