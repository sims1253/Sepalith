# skipped: needs sf (nearest_facility, nearest_k_facilities, facilities_within,
#   snap_to_network, travel_time_matrix)

# access_by_band()
m <- matrix(c(5, 25, 70, NA,
              15, 12, 90, NA), nrow = 4)
res <- access_by_band(m, bands = c(30, 10))
stopifnot(
  identical(res$band, c("<=10", "<=30", ">30")),
  identical(res$n, c(1L, 1L, 1L)),
  isTRUE(all.equal(res$share, c(0.25, 0.25, 0.25))),
  inherits(try(access_by_band(matrix(numeric(0), nrow = 2)), silent = TRUE), "try-error")
)
