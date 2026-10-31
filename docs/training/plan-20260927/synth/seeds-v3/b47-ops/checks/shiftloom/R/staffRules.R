r = rbind(
  ann = c("E", "E", "E", "E", "E", "E", "E", "O"),
  bob = c("N", "N", "N", "N", "N", "O", "O", "O"),
  cy = c("L", "E", "O", "O", "O", "O", "O", "O"),
  dee = c("O", "O", "L", "L", "O", "N", "O", "E")
)
colnames(r) = paste0("d", 1:8)
v = checkRestRules(r)
stopifnot(identical(v$rule, c("consecutive", "nights", "hours", "hours", "quickReturn")))
stopifnot(identical(v$staff, c("ann", "bob", "ann", "bob", "cy")))
stopifnot(v$value[1] == 7, v$value[2] == 5)
stopifnot(v$value[3] == 56, v$limit[3] == 48)
stopifnot(nrow(checkRestRules(r, maxConsecutive = 7, maxNights = 5, maxHours = 60)) == 1)
stopifnot(nrow(checkRestRules(r[, 1, drop = FALSE])) == 0)
v2 = checkRestRules(r, hours = c(E = 4, L = 4, N = 12))
stopifnot(identical(v2$staff[v2$rule == "hours"], "bob"))

fr = fairnessReport(r, weekendDays = 6:7)
stopifnot(identical(fr$measure, c("nights", "lates", "weekends", "hours")))
stopifnot(fr$max[1] == 5, fr$range[1] == 5, fr$max[2] == 2, fr$min[2] == 0)
stopifnot(fr$max[3] == 2, fr$max[4] == 56, identical(attr(fr, "mostNights"), "bob"))
stopifnot(fairnessReport(matrix("O", 2, 2))$cv[1] == 0)

s = swapShifts(r, "ann", "dee", "d3")
stopifnot(s["ann", "d3"] == "L", s["dee", "d3"] == "E")
stopifnot(identical(swapShifts(r, "cy", "dee", 5), r))
err = function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
stopifnot(grepl("'zed' is not", err(swapShifts(r, "zed", "ann", 1))))
stopifnot(grepl("'zed' is not", err(swapShifts(r, "ann", "zed", 1))))
stopifnot(grepl("`day`", err(swapShifts(r, "ann", "bob", "d9"))))
stopifnot(grepl("'ann' without rest", err(swapShifts(r, "ann", "dee", "d6"))))
stopifnot(grepl("'bob' without rest", err(swapShifts(r, "ann", "bob", "d6"))))
stopifnot(grepl("'dee' without rest", err(swapShifts(r, "ann", "dee", "d7"))))

req = data.frame(staff = c("ann", "bob", "cy", "dee", "ann", "cy"),
  day = c("d1", "d2", "d3", "d3", "d2", "d1"),
  type = c("leave", "sick", "training", "preference", "training", "preference"),
  shift = c(NA, NA, NA, "N", NA, "E"))
a = applyAvailability(r, req)
stopifnot(a["ann", "d1"] == "A", a["bob", "d2"] == "S", a["cy", "d3"] == "O")
stopifnot(a["dee", "d3"] == "N", a["ann", "d2"] == "T", a["cy", "d1"] == "E")
rel = attr(a, "released")
stopifnot(identical(rel$staff, c("ann", "bob", "dee", "ann", "cy")), rel$shift[2] == "N")
stopifnot(grepl("Unknown request type 'swap'",
  err(applyAvailability(r, data.frame(staff = "ann", day = "d1", type = "swap")))))

stopifnot(identical(contractLimits("full")$maxHours, 48))
stopifnot(contractLimits("part")$maxHours == 24, contractLimits("part")$maxNights == 4)
cl = contractLimits("casual")
stopifnot(cl$maxConsecutive == 3, cl$minDaysOff == 3)
stopifnot(contractLimits("nightOnly")$maxNights == 6)
stopifnot(grepl("negative", err(contractLimits(base = list(maxHours = -1, maxConsecutive = 6,
  maxNights = 4, minDaysOff = 2)))))
