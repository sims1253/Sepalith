# skipped: needs tibble: partialEffectData(); needs ggplot2: plotPartialEffect(),
# plotAllPartials(); needs purrr/tibble/dplyr: partialEffectRange()
dat <- data.frame(
  x = c(1, 2, 3, 10), g = factor(c("a", "b", "b", "a")),
  h = c("u", "v", "v", "v"), m = c(NA, NA, NA, NA), stringsAsFactors = FALSE
)
tv <- typicalValues(dat, exclude = "m")
stopifnot(nrow(tv) == 1, tv$x == 2.5, identical(as.character(tv$g), "a"), tv$h == "v")
stopifnot(identical(levels(tv$g), c("a", "b")))
tm <- typicalValues(dat)
stopifnot(nrow(tm) == 1, is.na(tm$m))

fakeFit <- list(smooth = list(
  list(label = "s(x)", term = "x"),
  list(label = "te(x,z)", term = c("x", "z"))
))
stopifnot(identical(smoothTermVar(fakeFit, "te(x,z)"), "x"))
stopifnot(inherits(try(smoothTermVar(fakeFit, "s(q)"), silent = TRUE), "try-error"))

grid <- partialEffectGrid(dat, "x", n = 5)
stopifnot(nrow(grid) == 5, identical(grid$x, seq(1, 10, length.out = 5)))
stopifnot(all(grid$h == "v"))
stopifnot(inherits(try(partialEffectGrid(dat, "h"), silent = TRUE), "try-error"))
