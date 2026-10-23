p <- parcel(c("A1", "A2", "B7"), c(500, 250, 1000), c("r2", "R2", "c1"),
  owner = c("Ames", "Ames", "Bolt"), assessed_value = c(1e5, NA, 4e5))
stopifnot(is(p, "Parcel"), identical(p@zone, c("R2", "R2", "C1")), length(p) == 3L)
stopifnot(identical(parcel("X", 10, "R1")@owner, "unknown"),
  is.na(parcel("X", 10, "R1")@assessed_value))
stopifnot(inherits(try(parcel(c("A", "A"), c(1, 2), "R1"), silent = TRUE), "try-error"))
bad_len <- function(...) inherits(try(parcel(c("A", "B", "C"), ...), silent = TRUE), "try-error")
stopifnot(bad_len(1:2, "R1"), bad_len(1:3, c("R1", "R2")), bad_len(1:3, "R1", owner = c("a", "b")))
stopifnot(bad_len(1:3, "R1", assessed_value = 1:2), !bad_len(1:3, "R1"))

stopifnot(isTRUE(valid_parcel(p)))
bad <- p
bad@area_m2 <- 1
bad@owner <- c("x", "y")
stopifnot(identical(valid_parcel(bad),
  c("`area_m2` must have one value per parcel.", "`owner` must have one value per parcel.")))
bad <- p
bad@parcel_id <- c("A1", "A1", NA)
bad@area_m2 <- c(1, 0, 2)
bad@zone <- c("R2", "res", "C1")
bad@owner <- c("a", NA, "b")
bad@assessed_value <- c(1, -1, NA)
msgs <- valid_parcel(bad)
stopifnot(length(msgs) == 5L, grepl("^`parcel_id`", msgs[1]), grepl("^`area_m2`", msgs[2]))
stopifnot(grepl("^`zone`", msgs[3]), grepl("^`owner`", msgs[4]),
  grepl("^`assessed_value`", msgs[5]))

s <- p[c(TRUE, FALSE, TRUE)]
stopifnot(identical(s@parcel_id, c("A1", "B7")), identical(s@assessed_value, c(1e5, 4e5)))
stopifnot(identical(parcel_subset(p, "B7")@owner, "Bolt"), identical(p[2:1]@area_m2, c(250, 500)))
stopifnot(inherits(try(parcel_subset(p, "Z9"), silent = TRUE), "try-error"))

q <- parcel_combine(p, parcel("C3", 75, "I1", owner = "Cole", assessed_value = 2e4))
stopifnot(length(q) == 4L, identical(q@zone[4], "I1"), identical(q@owner[4], "Cole"))
stopifnot(inherits(try(parcel_combine(p, p[1]), silent = TRUE), "try-error"))

df <- as(p, "data.frame")
stopifnot(is.data.frame(df), identical(names(df), c("parcel_id", "area_m2", "zone", "owner",
  "assessed_value")), nrow(df) == 3L)
stopifnot(identical(parcel_to_frame(p), df), identical(as(df, "Parcel"), p))
fp <- frame_to_parcel(data.frame(parcel_id = 7:8, area_m2 = c("3", "4"), zone = "a"))
stopifnot(identical(fp@owner, c("unknown", "unknown")), identical(fp@parcel_id, c("7", "8")))
stopifnot(identical(fp@zone, c("A", "A")), identical(fp@area_m2, c(3, 4)),
  all(is.na(fp@assessed_value)))
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(fails(frame_to_parcel(data.frame(area_m2 = 1, zone = "A"))))
stopifnot(fails(frame_to_parcel(data.frame(parcel_id = "Q", zone = "A"))))
stopifnot(fails(frame_to_parcel(data.frame(parcel_id = "Q", area_m2 = 1))))

sz <- parcel_summary(p)
stopifnot(identical(names(sz), c("group", "n", "area_m2", "mean_area_m2", "min_area_m2",
  "max_area_m2", "n_assessed", "assessed_value", "value_per_m2")))
stopifnot(identical(sz$group, c("C1", "R2")), identical(sz$n, c(1L, 2L)), sz$area_m2[2] == 750)
stopifnot(sz$min_area_m2[2] == 250, sz$max_area_m2[2] == 500, identical(sz$n_assessed, c(1L, 1L)))
stopifnot(sz$assessed_value[2] == 1e5, sz$value_per_m2[1] == 400, sz$value_per_m2[2] == 200)
so <- parcel_summary(p, "owner")
stopifnot(identical(so$group, c("Ames", "Bolt")), so$area_m2[1] == 750)
stopifnot(is.na(parcel_summary(parcel("Z", 5, "A"))$value_per_m2), sz$mean_area_m2[2] == 375)
stopifnot(fails(parcel_summary(p[integer()])))

out <- capture.output(show(p))
stopifnot(grepl("<Parcel> 3 parcels, 1750 m2", out[1]), length(out) == 5L)
