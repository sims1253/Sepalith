# skipped: mod_freezer_map_ui, mod_freezer_map_server (need shiny / bslib);
#   plot_box_grid (needs ggplot2)

# parse_slot / format_slot
p <- parse_slot(c("F2-R3-B07", "F1-R6-B100", NA))
stopifnot(identical(p$freezer, c("F2", "F1", NA)), identical(p$position, c(7L, 100L, NA)))
stopifnot(identical(format_slot("F2", "R3", 7), "F2-R3-B07"))
stopifnot(identical(parse_slot(format_slot("F4", "R1", 42))$position, 42L))
stopifnot(nrow(parse_slot(character())) == 0L)

# box_occupancy ignores missing and out-of-range positions
g <- box_occupancy(c("F2-R3-B07", "F2-R3-B11", NA, "F2-R3-B999"))
stopifnot(nrow(g) == 100L, sum(g$occupied) == 2L)
stopifnot(identical(g$slot[g$position == 11L], "F2-R3-B11"), g$row[g$position == 11L] == 2L)
stopifnot(sum(box_occupancy(character())$occupied) == 0L)

# rack_fill_table
reg <- data.frame(freezer_slot = c("F1-R1-B01", "F1-R1-B02", "F1-R4-B01", "F2-R1-B01", NA))
fill <- rack_fill_table(reg, "F1")
stopifnot(identical(fill$Rack, paste0("R", 1:6)))
stopifnot(identical(fill$Samples, c(2L, 0L, 0L, 1L, 0L, 0L)))
stopifnot(identical(fill$Free[1], 98L))
