d <- data.frame(e1 = c(1, 2, NA), e2 = c(3, 4, NA), m1 = c(5, 1, 2), s1 = c(2, NA, 4),
  s2 = c(4, 2, 4), c1 = c(1, 1, 1))
w <- score_wellbeing(d, c("e1", "e2"), "m1", c("s1", "s2"), "c1")
stopifnot(identical(names(w), c("energy", "mood", "sleep", "social", "total")))
stopifnot(w$energy[1] == 2, is.na(w$energy[3]), w$mood[2] == 1, w$sleep[2] == 2)
stopifnot(w$social[3] == 1, w$total[1] == mean(c(2, 5, 3, 1)))
ws <- score_wellbeing(d, c("e1", "e2"), "m1", c("s1", "s2"), "c1", method = "sum")
stopifnot(ws$energy[2] == 6, ws$sleep[1] == 6)
stopifnot(inherits(try(score_wellbeing(d, "zz", "m1", "s1", "c1"), silent = TRUE), "try-error"))
stopifnot(inherits(try(score_wellbeing(d, "e1", "zz", "s1", "c1"), silent = TRUE), "try-error"))
stopifnot(inherits(try(score_wellbeing(d, "e1", "m1", "zz", "c1"), silent = TRUE), "try-error"))
stopifnot(inherits(try(score_wellbeing(d, "e1", "m1", "s1", "zz"), silent = TRUE), "try-error"))
stopifnot(inherits(try(score_wellbeing(1, "e1", "m1", "s1", "c1"), silent = TRUE), "try-error"))

r <- data.frame(a = c(1, 5, NA), b = c(3, 5, NA), c = c(NA, 5, 2))
stopifnot(identical(c(score_subscale(r, c("a", "b", "c"), "sum")), c(NA, 15, NA)))
stopifnot(all.equal(c(score_subscale(r, c("a", "b", "c"), "mean")), c(2, 5, 2)))
stopifnot(all.equal(c(score_subscale(r, c("a", "b", "c"), "prorated")), c(6, 15, 6)))
stopifnot(all.equal(c(score_subscale(r, c("a", "b", "c"), "median")), c(2, 5, 2)))
p <- score_subscale(r, c("a", "b", "c"), "pomp")
stopifnot(all.equal(c(p), c(25, 100, 25)), attr(p, "method") == "pomp")
stopifnot(identical(attr(p, "n_answered"), c(2, 3, 1)))
stopifnot(inherits(try(score_subscale(r, "zz"), silent = TRUE), "try-error"))
stopifnot(inherits(try(score_subscale(r, "a", min_value = 5), silent = TRUE), "try-error"))

b <- data.frame(x1 = c(20, 5), x2 = c(10, 2), y1 = c(12, 3), z1 = c(20, 40), z2 = c(10, 0))
bo <- score_burnout(b, c("x1", "x2"), "y1", c("z1", "z2"))
stopifnot(identical(bo$exhaustion_flag, c(TRUE, FALSE)))
stopifnot(identical(bo$cynicism_flag, c(TRUE, FALSE)))
stopifnot(identical(bo$efficacy_flag, c(TRUE, FALSE)), identical(bo$n_flags, c(3, 0)))
stopifnot(inherits(try(score_burnout(b, "q", "y1", "z1"), silent = TRUE), "try-error"))
stopifnot(inherits(try(score_burnout(b, "x1", "q", "z1"), silent = TRUE), "try-error"))
stopifnot(inherits(try(score_burnout(b, "x1", "y1", "q"), silent = TRUE), "try-error"))
stopifnot(inherits(try(score_burnout(b, "x1", "y1", "z1", cutoffs = c(a = 1)), silent = TRUE),
  "try-error"))

sc <- data.frame(depression = c(0, 6, 20), anxiety = c(4, 9, 3), stress = c(13, 7, 30))
cl <- classify_severity(sc)
stopifnot(identical(as.character(cl$depression), c("normal", "mild", "extreme")))
stopifnot(identical(as.character(cl$anxiety), c("mild", "severe", "normal")))
stopifnot(identical(as.character(cl$stress), c("severe", "normal", "extreme")))
stopifnot(is.ordered(cl$stress))
stopifnot(inherits(try(classify_severity(sc, depression = 1:3), silent = TRUE), "try-error"))
stopifnot(inherits(try(classify_severity(sc, anxiety = c(4, 3, 5, 6)), silent = TRUE),
  "try-error"))
stopifnot(inherits(try(classify_severity(sc, stress = c(1, 1, 2, 3)), silent = TRUE), "try-error"))

su <- summarise_scores(data.frame(a = c(1, 3, NA), b = c(NA, NA, NA), c = c(2, 2, 2)))
stopifnot(identical(su$subscale, c("a", "b", "c")), identical(su$n, c(2, 0, 3)))
stopifnot(su$mean[1] == 2, is.na(su$mean[2]), su$sd[3] == 0, su$max[1] == 3)
stopifnot(identical(su$status, c("partial", "partial", "complete")))
stopifnot(inherits(try(summarise_scores(data.frame(a = 1), "z"), silent = TRUE), "try-error"))

sp <- validate_scoring_spec(list(items = c("a", "b"), min_value = 1, max_value = 5))
stopifnot(sp$min_items == 0.5, sp$method == "mean", identical(sp$reverse, character()))
bad <- list(
  list(items = 1, min_value = 1, max_value = 5),
  list(items = "a", min_value = "1", max_value = 5),
  list(items = "a", min_value = 1, max_value = 1:2),
  list(items = "a", min_value = 5, max_value = 1),
  list(items = "a", min_value = 1, max_value = 5, min_items = 2),
  list(items = "a", min_value = 1, max_value = 5, method = "mode"),
  list(items = "a", min_value = 1, max_value = 5, reverse = "b"),
  list(items = "a", min_value = 1, max_value = 5, extra = 1)
)
for (s in bad) {
  stopifnot(inherits(try(validate_scoring_spec(s), silent = TRUE), "try-error"))
}
