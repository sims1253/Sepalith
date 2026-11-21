cv <- wqi_rating_curves()
stopifnot(identical(names(cv), c("do", "ph", "turbidity", "nitrate", "phosphate")))
stopifnot(all(vapply(cv, function(z) length(z$x) == length(z$q), logical(1))))

w <- water_quality_index(do = c(100, 60), ph = c(7, 7), turbidity = c(0, 25),
                         nitrate = c(0, 10), phosphate = c(0, 0.6))
stopifnot(identical(names(w), c("q_do", "q_ph", "q_turbidity", "q_nitrate", "q_phosphate",
                                "wqi")))
stopifnot(w$q_do == c(100, 60), w$q_ph == c(90, 90), w$q_turbidity == c(100, 55))
stopifnot(w$q_nitrate == c(100, 50), w$q_phosphate == c(100, 50))
stopifnot(abs(w$wqi[1] - (30 + 18 + 15 + 15 + 20)) < 1e-9)
w1 <- water_quality_index(do = 100, ph = 7)
stopifnot(is.na(w1$q_nitrate), abs(w1$wqi - (0.3 * 100 + 0.2 * 90) / 0.5) < 1e-9)
stopifnot(inherits(try(water_quality_index(), silent = TRUE), "try-error"))

t <- trophic_state_index(chl = exp(1), tp = exp(1), secchi = 1)
stopifnot(abs(t$tsi_chl - 40.41) < 1e-9, abs(t$tsi_tp - 18.57) < 1e-9, t$tsi_secchi == 60)
stopifnot(as.character(t$state) == "oligotrophic")
stopifnot(nrow(trophic_state_index(c(1, 50), 20, 2)) == 2L)
stopifnot(inherits(try(trophic_state_index(0, 1, 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(trophic_state_index(1, -1, 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(trophic_state_index(1, 1, 0), silent = TRUE), "try-error"))

ib <- ion_balance(ca = 40.078, mg = 24.305, na = 22.990, k = 39.098,
                  hco3 = 61.017, cl = 35.453, so4 = 96.06, no3 = 62.004)
stopifnot(identical(names(ib), c("cations", "anions", "sum_cations", "sum_anions",
                                 "error_pct")))
stopifnot(ib$cations[["ca"]] == 2, ib$cations[["k"]] == 1, ib$anions[["so4"]] == 2)
stopifnot(ib$sum_cations == 6, ib$sum_anions == 5, abs(ib$error_pct - 100 / 11) < 1e-9)

stopifnot(identical(classify_index(c(10, 95), "wqi"), c("very_bad", "excellent")))
stopifnot(identical(classify_index(c(45, 75), "tsi"), c("mesotrophic", "hypereutrophic")))
stopifnot(identical(classify_index(c(-1, 7), "ion_error"), c("excellent", "doubtful")))
stopifnot(identical(classify_index(c(0.5, 0.9), "comfort"), c("good", "ideal")))
