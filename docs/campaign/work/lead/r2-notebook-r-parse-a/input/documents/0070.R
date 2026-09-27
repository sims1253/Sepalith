cdss_read_skill_assignment_xlsx <- function(
  filename,
  taughtname = "Taught",
  requiredname = "Required",
  warnonly = FALSE,
  verbose = TRUE
) {
  t <- read.xlsx(xlsxFile = filename, sheet = taughtname)
  r <- read.xlsx(xlsxFile = filename, sheet = requiredname)
  sa <- cdss_tables2sa(t, r)
  check <- cdss_sa_compliance(sa, verbose)
  if (!check) {
    if (warnonly) {
      stop("The assignment tables are not skill assignment compliant!")
    } else {
      warning("The assignment tables are not skill assignment compliant!")
    }
  }
  sa
}