# skipped: needs rvest (scrape_tables)
stopifnot(
  identical(
    clean_names(c(" Total (EUR) ", "total eur", "", "%")),
    c("total_eur", "total_eur_1", "col", "col_1")
  ),
  identical(parse_number_col(c("1,234", "$5.5[2]", "n/a", "12 kEUR")), c(1234, 5.5, NA, 12)),
  identical(parse_percent_col(c("12.5%", "-", "100 %")), c(0.125, NA, 1)),
  identical(parse_date_col(c("2024-03-01", "02.01.2023", "soon")),
            as.Date(c("2024-03-01", "2023-01-02", NA)))
)

tables <- list(
  data.frame(a = 1, check.names = FALSE),
  data.frame(`Country Name` = c("A", "B"), `Pop.` = c("1,000", "2,500"), check.names = FALSE)
)
picked <- pick_table(tables, c("country name", "pop"))
stopifnot(
  identical(names(picked), c("country_name", "pop")),
  is.null(pick_table(tables, "gdp"))
)

df <- data.frame(n = c("1,000", "2"), ok = c("Yes", ""), d = c("2024-01-05", "x"))
out <- coerce_columns(df, c(n = "number", ok = "logical", d = "date"))
stopifnot(
  identical(out$n, c(1000, 2)),
  identical(out$ok, c(TRUE, FALSE)),
  inherits(out$d, "Date"),
  inherits(try(coerce_columns(df, c(zz = "number")), silent = TRUE), "try-error"),
  inherits(try(coerce_columns(df, c(n = "money")), silent = TRUE), "try-error")
)

combined <- combine_tables(list(p1 = data.frame(a = 1:2), p2 = data.frame(b = "x")))
stopifnot(
  identical(names(combined), c("source", "a", "b")),
  identical(combined$source, c("p1", "p1", "p2")),
  identical(combined$a, c(1L, 2L, NA)),
  identical(dim(combine_tables(list())), c(0L, 0L))
)
