msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
pdf(NULL)

stopifnot(identical(msg(panel_type_error("age", "numeric")),
                    "`age` cannot be drawn as a numeric panel."))

r <- panel_numeric(c(1, 2, 3, NA, Inf, 4), "age")
stopifnot(identical(r$kind, "numeric"), identical(r$n, 4L), identical(r$n_missing, 2L))
stopifnot(identical(msg(panel_numeric("a", "age")), "`age` cannot be drawn as a numeric panel."))

r <- panel_factor(c("a", "b", "b", NA, "c"), "grp", max_levels = 2L)
stopifnot(identical(r$kind, "factor"), identical(r$n, 4L), identical(r$n_missing, 1L))
stopifnot(identical(msg(panel_factor(1:3, "grp")), "`grp` cannot be drawn as a factor panel."))

r <- panel_logical(c(TRUE, FALSE, TRUE, NA), "flag")
stopifnot(identical(r$kind, "logical"), identical(r$n, 3L), identical(r$n_missing, 1L))
stopifnot(identical(msg(panel_logical(1, "flag")), "`flag` cannot be drawn as a logical panel."))

d <- as.Date("2024-01-01") + c(0, 10, 40, NA, 90)
r <- panel_date(d, "when")
stopifnot(identical(r$kind, "date"), identical(r$n, 4L), identical(r$n_missing, 1L))
stopifnot(identical(msg(panel_date("x", "when")), "`when` cannot be drawn as a date panel."))

stopifnot(identical(panel_kind(Sys.Date()), "date"), identical(panel_kind(NA), "logical"))
stopifnot(identical(panel_kind(1.5), "numeric"), identical(panel_kind("a"), "factor"))
stopifnot(is.na(panel_kind(list(1))))

df <- data.frame(a = c(1, 2, 3), b = c("x", "y", "x"), c = c(TRUE, NA, FALSE),
                 d = as.Date("2024-01-01") + 0:2)
s <- draw_panels(df)
stopifnot(identical(s$kind, c("numeric", "factor", "logical", "date")))
stopifnot(identical(s$n, c(3L, 3L, 2L, 3L)))
stopifnot(identical(msg(draw_panels(df, c("a", "zz"))), "Unknown variable(s): zz."))
df$e <- I(list(1, 2, 3))
w <- tryCatch(draw_panels(df, c("a", "e")), warning = function(w) conditionMessage(w))
stopifnot(identical(w, "Skipping `e`: no panel for class AsIs."))

invisible(dev.off())
