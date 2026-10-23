stopifnot(is.null(check.sample.id("BL-2024-00042")))
stopifnot(identical(check.sample.id(""), "`sample_id` is required."))
stopifnot(identical(check.sample.id("bl-24-1", name = "id"), "`id` must look like BL-2024-00042."))
stopifnot(grepl("must contain a year between 2000", check.sample.id("BL-1999-00001")))

stopifnot(is.null(check.volume(250)))
stopifnot(identical(check.volume(NA), "`volume` is required."))
stopifnot(identical(check.volume("12"), "`volume` must be a number."))
stopifnot(identical(check.volume(0), "`volume` must be positive."))
stopifnot(identical(check.volume(6000), "`volume` must be at most 5000."))

stopifnot(is.null(check.concentration(0)))
stopifnot(identical(check.concentration(-1), "`concentration` must not be negative."))
stopifnot(identical(check.concentration(2000, max = 1500),
                    "`concentration` must be at most 1500."))
stopifnot(identical(check.concentration(NULL), "`concentration` is required."))

stopifnot(is.null(check.temperature(-80)))
stopifnot(identical(check.temperature(-70), "`temperature` must be one of -196, -80, -20, 4."))
stopifnot(identical(check.temperature("cold"), "`temperature` must be a number."))

stopifnot(length(check.sample.form("BL-2023-00001", 100, 12.5, -20)) == 0L)
m <- check.sample.form("x", -5, NA, 21)
stopifnot(identical(names(m), c("sample_id", "volume", "concentration", "temperature")))
stopifnot(identical(m[["volume"]], "`volume` must be positive."))
stopifnot(identical(names(check.sample.form("BL-2023-00001", 100, -1, 4)), "concentration"))

lay <- data.frame(well = c("A1", "A2", "B1"), sample_id = c("s1", NA, NA),
                  type = c("sample", "blank", "standard"), dilution = c(1, 1, 10))
stopifnot(length(check.plate.layout(lay)) == 0L)
stopifnot(identical(check.plate.layout(lay[1:2]),
                    c("`type` column is missing.", "`dilution` column is missing.")))
bad <- data.frame(well = c("A1", "A1", "P24"), sample_id = c(NA, "s2", "s3"),
                  type = c("sample", "sample", "buffer"), dilution = c(0.5, 1, 1))
stopifnot(length(check.plate.layout(bad)) == 4L)
stopifnot(identical(check.plate.layout(bad[2:3, ], wells = 384L),
                    "`type` must be sample, standard, blank or control."))

stopifnot(length(check.storage.form(3, 2, 10, "b7")) == 0L)
sf <- check.storage.form(NA, 11, 0, "J1")
stopifnot(identical(names(sf), c("freezer", "rack", "box", "slot")))
stopifnot(identical(sf[["freezer"]], "`freezer` is required."))
stopifnot(identical(sf[["rack"]], "`rack` must be between 1 and 10."))
stopifnot(identical(sf[["box"]], "`box` must be between 1 and 25."))
stopifnot(identical(sf[["slot"]], "`slot` must be between A1 and I9."))
stopifnot(identical(check.storage.form(13, 1, 1, "")[["freezer"]],
                    "`freezer` must be between 1 and 12."))
stopifnot(identical(names(check.storage.form(1, NULL, 1, "")), c("rack", "slot")))
