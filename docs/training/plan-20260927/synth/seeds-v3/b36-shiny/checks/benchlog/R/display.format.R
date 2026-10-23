stopifnot(identical(fmt.volume(c(250, NA)), c("250.0 uL", "-")))
stopifnot(identical(fmt.volume(c(1500, 20)), c("1.5 mL", "0.0 mL")))
stopifnot(identical(fmt.volume(2e6), "2.0 L"), identical(fmt.volume(50, "mL", 2), "0.05 mL"))

stopifnot(identical(fmt.mass(500), "500.0 ng"), identical(fmt.mass(2500), "2.5 ug"))
stopifnot(identical(fmt.mass(3e6), "3.0 mg"), identical(fmt.mass(NA_real_), "-"))

stopifnot(identical(fmt.duration(45), "45.0 s"), identical(fmt.duration(90), "1.5 min"))
stopifnot(identical(fmt.duration(c(7200, 1800)), c("2.0 h", "0.5 h")))
stopifnot(identical(fmt.duration(120, "s", 0), "120 s"))

stopifnot(identical(fmt.storage.location(2, 5, 12, "a3"), "F02/R05/B12/A3"))
stopifnot(identical(fmt.storage.location(1, slot = "C7"), "F01/C7"))
stopifnot(identical(fmt.storage.location(rack = 3, box = 4), "R03/B04"))
stopifnot(identical(fmt.storage.location(), "unassigned"))

stopifnot(identical(fmt.status.badge(), list(label = "Received", class = "badge bg-info",
                                             icon = "inbox")))
stopifnot(identical(fmt.status.badge("stored")$icon, "snowflake"))
stopifnot(identical(fmt.status.badge("in_use")$label, "In use"))
stopifnot(identical(fmt.status.badge("depleted")$class, "badge bg-secondary"))
stopifnot(identical(fmt.status.badge("discarded")$icon, "trash"))

s <- list(sample_id = "BL-2024-00042", type = "DNA", volume = 200, concentration = 50,
          freezer = 1, rack = 2, box = 3, slot = "b4", received = as.Date("2024-03-01"))
card <- fmt.sample.card(s, today = as.Date("2024-03-11"))
stopifnot(identical(names(card), c("title", "subtitle", "volume", "amount", "location", "age")))
stopifnot(identical(card$subtitle, "Type: DNA"), identical(card$volume, "200.0 uL"))
stopifnot(identical(card$amount, "10.0 ug"), identical(card$location, "F01/R02/B03/B4"))
stopifnot(identical(card$age, "10 days"))
stopifnot(identical(fmt.sample.card(s, today = as.Date("2024-03-02"))$age, "1 day"))

tab <- data.frame(sample_id = c("S1", "S2"), volume = c(250, NA), concentration = c(12.34, NA),
                  temperature = c(-80, NA), status = c("stored", "in_use"),
                  updated = as.POSIXct(c("2024-03-01 10:30", NA), tz = "UTC"), extra = 1:2)
st <- fmt.sample.table(tab)
stopifnot(identical(names(st), c("Sample", "Volume", "Concentration", "Storage", "Status",
                                 "Updated")))
stopifnot(identical(st$Volume, c("250.0 uL", "-")),
          identical(st$Concentration, c("12.3 ng/uL", "-")))
stopifnot(identical(st$Storage, c("-80 C", "-")), identical(st$Status, c("Stored", "In use")))
stopifnot(identical(st$Updated, c("2024-03-01 10:30", "-")))
