# skipped: needs readr (read_arrivals_csv), readxl (read_berths_xlsx),
#   jsonlite (read_tides_json), dplyr (clean_arrivals, join_berths)

# standardise_names()
df <- data.frame(1, 2, 3, 4)
names(df) <- c("Call ID", "ETA (local)", "__Berth__", "call-id")
out <- standardise_names(df)
stopifnot(
  identical(names(out), c("call_id", "eta_local", "berth", "call_id_1")),
  identical(nrow(out), 1L)
)

# check_columns()
stopifnot(
  identical(check_columns(out, c("berth", "eta_local"), "x.csv"), out),
  inherits(try(check_columns(out, c("berth", "atd"), "x.csv"), silent = TRUE), "try-error"),
  grepl("x.csv is missing column\\(s\\): atd",
        tryCatch(check_columns(out, "atd", "x.csv"), error = conditionMessage))
)
