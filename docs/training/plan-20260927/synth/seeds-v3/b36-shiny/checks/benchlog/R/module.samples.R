# skipped: needs shiny (samples.ui, samples.server)
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
t0 <- as.POSIXct("2024-03-01 09:00", tz = "UTC")
t1 <- as.POSIXct("2024-03-02 09:00", tz = "UTC")

s <- samples.empty()
stopifnot(nrow(s) == 0L, identical(names(s), c("sample_id", "volume", "concentration",
                                               "temperature", "status", "updated")))

rec <- list(sample_id = "BL-2024-00001", volume = 100, concentration = 20, temperature = -80)
s <- samples.add(s, rec, time = t0)
s <- samples.add(s, modifyList(rec, list(sample_id = "BL-2024-00002")), time = t0)
stopifnot(nrow(s) == 2L, identical(s$status, c("received", "received")))
stopifnot(fails(samples.add(s, rec)))

u <- samples.update(s, "BL-2024-00002", list(volume = 80, concentration = 25,
                                             temperature = -20), time = t1)
stopifnot(identical(u$volume, c(100, 80)), identical(u$temperature, c(-80, -20)))
stopifnot(identical(u$status[2], "stored"), identical(u$updated[2], t1))
stopifnot(fails(samples.update(s, "nope", rec)))

c1 <- samples.consume(s, "BL-2024-00001", 40, time = t1)
stopifnot(identical(c1$volume[1], 60), identical(c1$status[1], "in_use"))
c2 <- samples.consume(c1, "BL-2024-00001", 60, time = t1)
stopifnot(identical(c2$status[1], "depleted"), identical(c2$volume[1], 0))
stopifnot(fails(samples.consume(s, "BL-2024-00001", 500)), fails(samples.consume(s, "x", 1)))

d <- samples.discard(s, "BL-2024-00002", time = t1)
stopifnot(identical(d$status, c("received", "discarded")), identical(d$volume, c(100, 0)))
stopifnot(fails(samples.discard(d, "BL-2024-00002")), fails(samples.discard(d, "x")))

ss <- samples.stats(c2)
stopifnot(identical(names(ss), c("n", "received", "stored", "in_use", "depleted", "discarded",
                                 "volume")))
stopifnot(ss$n == 2L, ss$received == 1L, ss$depleted == 1L, ss$volume == 100)
ss2 <- samples.stats(d)
stopifnot(ss2$discarded == 1L, ss2$stored == 0L, ss2$in_use == 0L, ss2$volume == 100)
stopifnot(samples.stats(samples.empty())$n == 0L)
