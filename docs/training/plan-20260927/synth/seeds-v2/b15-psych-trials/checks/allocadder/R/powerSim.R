set.seed(10)

# simContinuousTrial / simBinaryTrial
ct <- simContinuousTrial(5, 7, delta = 1, sd = 2)
stopifnot(nrow(ct) == 12, identical(levels(ct$arm), c("control", "active")))
bt <- simBinaryTrial(4, 4, p1 = 0, p2 = 1)
stopifnot(identical(bt$y, c(0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L)))

# analyseContinuousTrial / analyseBinaryTrial
p <- analyseContinuousTrial(simContinuousTrial(30, 30, delta = 5, sd = 1))
stopifnot(p < 0.001)
stopifnot(analyseBinaryTrial(simBinaryTrial(5, 5, 0, 0)) == 1)
stopifnot(analyseBinaryTrial(simBinaryTrial(5, 5, 1, 1)) == 1)
stopifnot(analyseBinaryTrial(simBinaryTrial(40, 40, 0, 1)) < 0.001)

# simulatePower
pw <- simulatePower(30, delta = 1, sd = 1, nSim = 200, seed = 1)
stopifnot(pw$power > 0.8, pw$nSim == 200)
pb <- simulatePower(20, endpoint = "binary", p1 = 0.95, p2 = 0.99, nSim = 100, seed = 2)
stopifnot(!is.na(pb$power))
stopifnot(inherits(try(simulatePower(20, delta = 1), silent = TRUE), "try-error"))

# powerCurve
pc <- powerCurve(c(5, 50), delta = 1, sd = 1, nSim = 100)
stopifnot(nrow(pc) == 2, pc$power[2] > pc$power[1])

# findSampleSize
fs <- findSampleSize(0.8, lower = 5, upper = 60, delta = 1, sd = 1, nSim = 100)
stopifnot(fs$n1 > 5, fs$n1 <= 60, fs$power >= 0.8)
