# eoq
e <- eoq(demandRate = 1000, orderCost = 10, holdingCost = 2)
stopifnot(abs(e$quantity - 100) < 1e-9, abs(e$ordersPerPeriod - 10) < 1e-9)
stopifnot(abs(e$totalCost - 200) < 1e-9)

# reorderPoint
rp <- reorderPoint(demandMean = 20, demandSd = 5, leadTime = 4, serviceLevel = 0.5)
stopifnot(rp$z == 0, rp$safetyStock == 0, rp$reorderPoint == 80)
stopifnot(reorderPoint(20, 5, 4)$safetyStock > 0)

# simulateSQ
sq <- simulateSQ(demand = rep(5, 6), s = 10, Q = 20, leadTime = 2L, initial = 12)
stopifnot(nrow(sq) == 6, all(sq$sold + sq$lost == sq$demand))
stopifnot(sq$ordered[1] == 20, sq$received[3] == 20)
stopifnot(inherits(try(simulateSQ(1:3, 1, 2, leadTime = 0), silent = TRUE), "try-error"))

# simulateRS
rs <- simulateRS(demand = rep(4, 6), R = 2, S = 20, leadTime = 1L)
stopifnot(all(rs$ordered[c(1, 3, 5)] == 0), rs$ordered[2] == 8, rs$received[3] == 8)

# policyMetrics
pm <- policyMetrics(sq, holdingCost = 1, orderCost = 50, lostSaleCost = 10)
stopifnot(pm$total == pm$holding + pm$ordering + pm$shortage, pm$fillRate <= 1)

# tuneSQ
tu <- tuneSQ(rep(5, 10), sGrid = c(5, 10), qGrid = c(10, 20))
stopifnot(nrow(tu) == 4, !is.unsorted(tu$total))
