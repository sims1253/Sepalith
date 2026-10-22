# skipped: mod_sample_entry_ui, mod_sample_entry_server (need shiny / bslib)

# check_barcode
stopifnot(length(check_barcode(" VT-000123 ", c("VT-000001"))) == 0L)
stopifnot(identical(check_barcode("", character()), "Barcode is required."))
stopifnot(identical(check_barcode(NULL, character()), "Barcode is required."))
stopifnot(grepl("already registered", check_barcode("VT-000001", "VT-000001")))
stopifnot(grepl("must look like", check_barcode("VT-12", character())))

# check_freezer_slot
stopifnot(length(check_freezer_slot("f2-r3-b07", character())) == 0L)
stopifnot(grepl("occupied", check_freezer_slot("F2-R3-B07", "F2-R3-B07")))
stopifnot(identical(check_freezer_slot(NULL, character()), "Freezer slot is required."))

# check_volume
stopifnot(length(check_volume(500, "plasma")) == 0L)
stopifnot(identical(check_volume(NA, "plasma"), "Volume is required."))
stopifnot(grepl("between 500 and 10000", check_volume(100, "urine")))
stopifnot(length(check_volume(5, "swab")) == 0L)

# check_collected_at
stopifnot(length(check_collected_at("2026-01-02 10:30")) == 0L)
stopifnot(grepl("must look like", check_collected_at("yesterday")))
stopifnot(grepl("future", check_collected_at("2999-01-01 00:00")))

# make_sample_record and sample_matrices
rec <- make_sample_record(" VT-000123", "plasma", "250", "2026-01-02 10:30", "f2-r3-b07")
stopifnot(nrow(rec) == 1L, identical(rec$barcode, "VT-000123"), identical(rec$volume_ul, 250))
stopifnot(identical(rec$collected_at, "2026-01-02 10:30:00"))
stopifnot(identical(rec$freezer_slot, "F2-R3-B07"))
stopifnot(identical(rec$discarded, 0L), "saliva" %in% sample_matrices())
