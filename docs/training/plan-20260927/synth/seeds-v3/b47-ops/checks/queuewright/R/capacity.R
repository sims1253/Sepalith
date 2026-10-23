plan <- data.frame(period = c(1, 1, 2, 2), machine_hours = c(30, 20, 60, 10),
  labour_hours = c(10, 10, 20, 20), tooling_sets = c(1, 1, 2, 2),
  storage_m3 = c(5, 5, 5, 20))
warns <- character(0)
u <- withCallingHandlers(check_capacity(plan, 60, 40, 4, 20),
  warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
stopifnot(identical(u$machine_util, c(50, 70) / 60), identical(u$labour_util, c(0.5, 1)))
stopifnot(identical(u$tooling_util, c(0.5, 1)), identical(u$storage_util, c(0.5, 1.25)))
stopifnot(length(warns) == 2, any(grepl("machine_hours", warns)), any(grepl("storage_m3", warns)))
u2 <- withCallingHandlers(check_capacity(plan, 100, 10, 1, 100),
  warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
stopifnot(any(grepl("labour_hours", warns)), any(grepl("tooling_sets", warns)))

ok <- validate_capacity_args(10, 20, 3, 40)
stopifnot(identical(names(ok), c("machine_hours", "labour_hours", "tooling_sets", "storage_m3")))
err <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
stopifnot(grepl("machine_hours", err(validate_capacity_args("a", 1, 1, 1))))
stopifnot(grepl("labour_hours", err(validate_capacity_args(1, -1, 1, 1))))
stopifnot(grepl("whole numbers", err(validate_capacity_args(1, 1, 1.5, 1))))
stopifnot(grepl("tooling_sets", err(validate_capacity_args(1, 1, NA_real_, 1))))
stopifnot(grepl("storage_m3", err(validate_capacity_args(1, 1, 1, numeric(0)))))

orders <- data.frame(period = c("a", "a", "b"), quantity = c(10, 6, 4))
r_m <- resource_requirement(orders, "machine")
stopifnot(identical(r_m$required, c(16 * 0.5 + 2, 4 * 0.5 + 1)), identical(r_m$orders, c(2L, 1L)))
stopifnot(identical(resource_requirement(orders, "labour")$required, c(13, 3.5)))
stopifnot(identical(resource_requirement(orders, "oven")$required, c(8, 3)))
r_d <- resource_requirement(orders, "dock")
stopifnot(isTRUE(all.equal(r_d$required, c(1.3, 0.45))), all(r_d$resource == "dock"))

lv <- level_load(c(5, 15, 5), 10)
stopifnot(identical(lv$levelled, c(5, 10, 10)), lv$moved_forward == 5, lv$unplaced == 0)
lv2 <- level_load(c(5, 5, 18), 10)
stopifnot(identical(lv2$levelled, c(8, 10, 10)), lv2$moved_back == 8, lv2$moved_forward == 0)
lv3 <- level_load(c(10, 30), 10)
stopifnot(lv3$unplaced == 20, identical(lv3$levelled, c(10, 10)))

br <- bottleneck_report(u, threshold = 0.9)
stopifnot(br$resource[1] == "machine", br$bottleneck[1], sum(br$bottleneck) == 1)
stopifnot(br$over[br$resource == "storage"] == 1, br$critical[br$resource == "labour"] == 1)
stopifnot(br$peak_period[br$resource == "machine"] == 2, nrow(br) == 4)
