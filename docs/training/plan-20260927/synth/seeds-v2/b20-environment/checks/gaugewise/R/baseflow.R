flow <- c(rep(1, 10), 10, 6, 3, 2, rep(1, 16))

# check_baseflow_input
stopifnot(identical(check_baseflow_input(flow), flow))
stopifnot(inherits(try(check_baseflow_input(c(flow, NA)), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_baseflow_input(1:5), silent = TRUE), "try-error"))

# the three filters keep baseflow between zero and total flow
b1 <- baseflow_one_param(flow)
b2 <- baseflow_two_param(flow)
b3 <- baseflow_fixed_interval(flow)
for (b in list(b1, b2, b3)) {
  stopifnot(length(b) == length(flow), all(b >= 0), all(b <= flow + 1e-12))
}
stopifnot(b3[11] == 1, b1[11] < 10)
stopifnot(inherits(try(baseflow_one_param(flow, alpha = 1), silent = TRUE), "try-error"))

# separate_baseflow / baseflow_index
sb <- separate_baseflow(flow, "fixed_interval", interval = 5)
stopifnot(identical(sb$baseflow, b3), all(sb$quickflow == sb$flow - sb$baseflow))
stopifnot(baseflow_index(flow, flow) == 1, is.na(baseflow_index(0, 0)))

# recession_constant on a pure exponential recession
rec <- 100 * 0.9^(0:39)
rc <- recession_constant(rec, quantile_cut = 1)
stopifnot(abs(rc$k - 0.9) < 1e-12, rc$n_pairs == 39L)
stopifnot(is.na(suppressWarnings(recession_constant(rep(1, 40)))$k))

# baseflow_events merges short dips and tolerates missing days
q <- c(0, 2, 0, 3, 0, 0, 0, 5, NA, 0)
ev <- baseflow_events(q, rep(0, 10), threshold = 0.5, min_gap = 2L)
stopifnot(identical(ev$start, c(2L, 8L)), identical(ev$end, c(4L, 8L)), ev$peak_day[1] == 4L)
stopifnot(nrow(baseflow_events(rep(0, 5), rep(0, 5))) == 0)
