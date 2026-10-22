# skipped: needs ranger (fit.ranger.model, predict.mc_ranger, ranger.importance,
# ranger.oob.error, print.mc_ranger)
df <- data.frame(y = 1:3, s = c("a", "b", "a"), f = factor(c("u", "v", "u")), n = 1:3,
  stringsAsFactors = FALSE)

# forest.prepare
prep <- forest.prepare(df, c("s", "f", "n"))
stopifnot(is.factor(prep$data$s), identical(names(prep$levels), c("s", "f")))
stopifnot(identical(prep$levels$s, c("a", "b")))
stopifnot(inherits(try(forest.prepare(df, "zz"), silent = TRUE), "try-error"))

# forest.align
nd <- data.frame(s = c("b", "q"), f = c("u", NA), stringsAsFactors = FALSE)
warned <- FALSE
al <- withCallingHandlers(forest.align(nd, prep$levels),
  warning = function(w) { warned <<- TRUE; invokeRestart("muffleWarning") })
stopifnot(warned, identical(levels(al$s), c("a", "b")), is.na(al$s[2]), al$f[1] == "u")
