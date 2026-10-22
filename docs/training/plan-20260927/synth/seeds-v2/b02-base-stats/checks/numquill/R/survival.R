tm <- c(1, 2, 2, 3, 4, 5, 5, 6)
st <- c(1, 1, 0, 1, 0, 1, 1, 0)

rt <- risk.table(tm, st)
stopifnot(identical(rt$time, c(1, 2, 3, 4, 5, 6)), identical(rt$n.risk, c(8, 7, 5, 4, 3, 1)))
stopifnot(identical(rt$n.event, c(1, 1, 1, 0, 2, 0)), sum(rt$n.censor) == 3)
stopifnot(inherits(try(risk.table(c(1, NA), c(1, 0)), silent = TRUE), "try-error"))
stopifnot(inherits(try(risk.table(1:2, c(1, 2)), silent = TRUE), "try-error"))

km <- kaplan.meier(tm, st)
stopifnot(inherits(km, "nq.km"), km$n == 8L, km$events == 5)
stopifnot(all.equal(km$table$surv, cumprod(c(7 / 8, 6 / 7, 4 / 5, 1 / 3))))
stopifnot(all(km$table$lower <= km$table$surv & km$table$surv <= km$table$upper))
km0 <- kaplan.meier(c(1, 2, 3), c(1, 1, 1))
stopifnot(km0$table$surv[3] == 0, is.na(km0$table$lower[3]), is.na(km0$table$upper[3]))

na <- nelson.aalen(tm, st)
stopifnot(inherits(na, "nq.na"), all.equal(na$table$cumhaz, cumsum(c(1 / 8, 1 / 7, 1 / 5, 2 / 3))))
stopifnot(all(na$table$lower < na$table$cumhaz))

stopifnot(km.quantile(km, 0.5) == 5, is.na(km.quantile(kaplan.meier(1:4, c(1, 0, 0, 0)))))

lr <- logrank.test(tm, st, rep(c("a", "b"), 4))
stopifnot(lr$p.value > 0, lr$p.value <= 1, lr$observed == 2)
stopifnot(inherits(try(logrank.test(tm, st, rep(1, 8)), silent = TRUE), "try-error"))

out <- utils::capture.output(res <- print(km))
stopifnot(identical(res, km), grepl("Median survival: 5", out[2]))
out <- utils::capture.output(res <- print(na))
stopifnot(identical(res, na), grepl("^Nelson-Aalen", out[1]))
