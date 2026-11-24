stopifnot(identical(names(gaugeDatums()), c("LAT", "CD", "NAP", "MSL")))
stopifnot(gaugeDatums()[["MSL"]] > gaugeDatums()[["NAP"]])

r <- gaugeReading(" nl042", c(3, 1, 2), c(1.3, 1.1, 1.2), quality = c("ok", "BAD", "ok"))
stopifnot(is(r, "GaugeReading"), r@station == "NL042", identical(r@time, c(1, 2, 3)))
stopifnot(identical(r@level, c(1.1, 1.2, 1.3)), identical(r@quality, c("bad", "ok", "ok")))
stopifnot(identical(gaugeReading("NL042", 1, 1, datum = "nap")@datum, "NAP"))
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(fails(gaugeReading(42, 1, 1)), fails(gaugeReading("NL042", c(1, NA), 1:2)))
stopifnot(fails(gaugeReading("NL042", 1:2, 1)),
  fails(gaugeReading("NL042", 1:3, 1:3, c("ok", "ok"))))
stopifnot(fails(gaugeReading("NL042", 1, 1, datum = c("LAT", "CD"))))
stopifnot(fails(gaugeReading("NL042", 1, 1, datum = "ODN")))

stopifnot(isTRUE(validGaugeReading(r)))
bad <- r
bad@station <- NA_character_
bad@datum <- "XYZ"
bad@time <- c(2, 1, 3)
bad@level <- 1
bad@quality <- c("ok", "ok", "weird")
msgs <- validGaugeReading(bad)
stopifnot(length(msgs) == 5L, grepl("station", msgs[1]), grepl("datum", msgs[2]))
stopifnot(grepl("time", msgs[3]), grepl("level", msgs[4]), grepl("quality", msgs[5]))
bad@datum <- c("LAT", "CD")
bad@quality <- "ok"
msgs <- validGaugeReading(bad)
stopifnot(any(grepl("`datum` must be a single", msgs)), any(grepl("`quality` must have", msgs)))

readingLevel(r) <- c(2, 3, 4)
stopifnot(identical(readingLevel(r), c(2, 3, 4)), fails(updateReading(r, level = 1:2)))
readingQuality(r) <- "Suspect"
stopifnot(identical(readingQuality(r), rep("suspect", 3)))
uq <- updateReading(r, quality = c("ok", "bad", "ok"))
stopifnot(identical(uq@quality, c("ok", "bad", "ok")))
stopifnot(fails(updateReading(r, quality = c("ok", "ok"))), fails(updateReading(r, quality = "x")))
readingDatum(r) <- "msl"
stopifnot(readingDatum(r) == "MSL", isTRUE(all.equal(readingLevel(r), c(2, 3, 4) - 1.12)))
stopifnot(isTRUE(all.equal(updateReading(r, datum = "LAT")@level, c(2, 3, 4))))
stopifnot(fails(updateReading(r, datum = "ODN")))
both <- updateReading(r, level = 1:3, datum = "LAT")
stopifnot(isTRUE(all.equal(both@level, 1:3 + 1.12)), identical(updateReading(r), r))

r3 <- gaugeReading("NL042", 1:4, c(1, 2, 5, 3), quality = c("ok", "filled", "bad", "ok"))
sub <- r3[c(4, 2)]
stopifnot(identical(sub@time, c(2, 4)), identical(sub@quality, c("filled", "ok")))
stopifnot(identical(subsetReading(r3, r3@level > 2)@level, c(5, 3)))
stopifnot(fails(subsetReading(r3, c(TRUE, FALSE))))

fr <- gaugeReading("NL042", 1:10, c(1, 1.1, 9, 1.2, 2, 2, 2, 2, 2, 2),
  quality = c(rep("ok", 9), "bad"))
q <- flagReadings(fr)@quality
stopifnot(identical(q[1:5], c("ok", "ok", "bad", "suspect", "suspect")), identical(q[6:10],
  c(rep("suspect", 4), "bad")))
stopifnot(identical(flagReadings(fr, flat = 7L)@quality[6:9], rep("ok", 4)))
stopifnot(identical(flagReadings(fr, range = c(0, 10), spike = 10)@quality[3], "ok"))

s <- summariseReading(r3)
stopifnot(s$n == 4L, s$usable == 3L, s$mean == 2, s$low == 1, s$high == 3, s$range == 2)
stopifnot(summariseReading(r3, flags = "ok")$usable == 2L)
stopifnot(is.na(summariseReading(r3, flags = "none")$mean))

out <- capture.output(res <- showReading(r3))
stopifnot(length(out) == 8L, grepl("NL042 \\(LAT\\), 4 readings", out[1]), identical(res, r3))
stopifnot(grepl("span:   1 to 4 h", out[2]), grepl("high:   3.00 m", out[6]))
stopifnot(grepl("bad=1, filled=1", out[8]), identical(capture.output(show(r3)), out))
stopifnot(length(capture.output(show(r3[r3@level > 9]))) == 1L)
