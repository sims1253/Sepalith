counts <- data.frame(
  aesoc = c("GI", "GI", "GI", "Skin"),
  aedecod = c("Nausea", "Nausea", "Diarrhoea", "Rash"),
  arm = c("Active", "Placebo", "Active", "Placebo"),
  n = c(12L, 3L, 2L, 1L),
  n_arm = c(100L, 50L, 100L, 50L)
)
counts$pct <- 100 * counts$n / counts$n_arm

# fmt.n.pct / fmt.arm.header
stopifnot(identical(fmt.n.pct(c(3, 0), c(12.345, 0)), c("3 (12.3%)", "0")))
stopifnot(identical(fmt.arm.header("Placebo", 120), "Placebo (N=120)"))

# ae.table.wide: absent cells become "0"
w <- ae.table.wide(counts)
stopifnot(identical(names(w), c("aesoc", "aedecod", "Active (N=100)", "Placebo (N=50)")))
stopifnot(identical(w$`Placebo (N=50)`[w$aedecod == "Diarrhoea"], "0"))
stopifnot(identical(w$`Active (N=100)`[1], "12 (12.0%)"))

# ae.table.sort
s <- ae.table.sort(counts, "frequency")
stopifnot(identical(s$aedecod[1:2], c("Nausea", "Nausea")), s$aesoc[4] == "Skin")
a <- ae.table.sort(counts, "alphabetical")
stopifnot(identical(a$aedecod, c("Diarrhoea", "Nausea", "Nausea", "Rash")))

# ae.table.common
cm <- ae.table.common(counts, min.pct = 5)
stopifnot(nrow(cm) == 2, all(cm$aedecod == "Nausea"))

# ae.listing.serious
ae <- data.frame(
  usubjid = c("02", "01", "01"), trt01a = "A", aedecod = c("x", "y", "z"),
  aestdt = as.Date(c("2024-01-05", "2024-02-01", "2024-01-01")),
  aeendt = as.Date(c("2024-01-06", "2024-02-01", "2024-01-03")),
  aesev = "MILD", aerel = "NONE", aeout = "RECOVERED", aeser = c("Y", "Y", NA)
)
ls <- ae.listing.serious(ae)
stopifnot(nrow(ls) == 2, identical(ls$usubjid, c("01", "02")))
stopifnot(identical(ls$duration.days, c(1L, 2L)))
stopifnot(inherits(try(ae.listing.serious(ae[names(ae) != "aeser"]), silent = TRUE), "try-error"))

# write.ae.table
tmp <- tempfile(fileext = ".md")
stopifnot(identical(write.ae.table(w, tmp, "markdown"), tmp))
md <- readLines(tmp)
stopifnot(length(md) == nrow(w) + 2, startsWith(md[2], "|---"))
unlink(tmp)
