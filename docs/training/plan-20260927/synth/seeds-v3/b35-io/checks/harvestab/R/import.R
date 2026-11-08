dir <- tempfile("import")
dir.create(dir)
writeLines(c("\"hh_id\",\"village\",\"size\",\"visited\"", "\"H1\",\"Ama\",,FALSE",
  "\"H2\",\"Oro\",3,TRUE"), file.path(dir, "households.csv"))
writeLines(c("person_id,hh_id,age,sex", "P1,H1,70,f", "P2,H2,12,m"),
  file.path(dir, "persons.csv"))
writeLines(c("visit_id,hh_id,date,outcome", "V1,H2,2026-06-01,refused", "V2,H1,2026-06-02,done"),
  file.path(dir, "visits.csv"))
files <- c("households.csv", "persons.csv", "visits.csv")
utils::write.csv(data.frame(table = sub(".csv", "", files, fixed = TRUE), file = files,
  rows = c(2L, 2L, 2L), md5 = unname(tools::md5sum(file.path(dir, files)))),
  file.path(dir, "manifest.csv"), row.names = FALSE)

h <- import.households(dir)
stopifnot(identical(h$size, c(NA, 3L)), identical(h$visited, c(FALSE, TRUE)))
p <- import.persons(dir)
stopifnot(identical(p$age, c(70L, 12L)), identical(p$sex, c("f", "m")))
v <- import.visits(dir)
stopifnot(inherits(v$date, "Date"), identical(v$outcome, c("refused", "done")))
msg <- tryCatch(import.households(tempdir()), error = conditionMessage)
stopifnot(grepl("does not exist", msg))

m <- import.verify(dir)
stopifnot(all(m$ok), identical(m$file, files))
all <- import.all(dir)
stopifnot(identical(names(all), c("households", "persons", "visits")), identical(all$persons, p))

writeLines(c("person_id,hh_id,age,sex", "P1,H1,70,f", "P1,H9,12,m"), file.path(dir, "persons.csv"))
msg <- tryCatch(import.all(dir), error = conditionMessage)
stopifnot(identical(msg, "Checksum mismatch for `persons.csv`."))
msg <- tryCatch(import.persons(dir), error = conditionMessage)
stopifnot(identical(msg, "`persons.csv` has missing or duplicated `person_id` values."))
writeLines(c("person_id,hh_id,age,sex", "P1,H1,70,f", "P2,H9,12,m"), file.path(dir, "persons.csv"))
msg <- tryCatch(import.all(dir, verify = FALSE), error = conditionMessage)
stopifnot(identical(msg, "`persons` refers to unknown household(s): H9"))
writeLines(c("person_id,hh_id,age,sex", "P1,H1,70,f"), file.path(dir, "persons.csv"))
writeLines(c("visit_id,hh_id,date,outcome", "V1,H7,2026-06-01,done"), file.path(dir, "visits.csv"))
msg <- tryCatch(import.all(dir, verify = FALSE), error = conditionMessage)
stopifnot(identical(msg, "`visits` refers to unknown household(s): H7"))
writeLines(c("visit_id,hh_id,date", "V1,H7,2026-06-01"), file.path(dir, "visits.csv"))
msg <- suppressWarnings(tryCatch(import.visits(dir), error = function(e) "failed"))
stopifnot(is.character(msg))
writeLines(c("hh_id,village,size,visited", "H1,A,1,TRUE", "H1,B,2,FALSE"),
  file.path(dir, "households.csv"))
msg <- tryCatch(import.households(dir), error = conditionMessage)
stopifnot(identical(msg, "`households.csv` has missing or duplicated `hh_id` values."))
writeLines(c("hh_id,village,size,visited", ",A,1,TRUE"), file.path(dir, "households.csv"))
msg <- tryCatch(import.households(dir), error = conditionMessage)
stopifnot(identical(msg, "`households.csv` has missing or duplicated `hh_id` values."))
writeLines(c("visit_id,hh_id,date,outcome", "V1,H7,2026-06-01,done", "V1,H7,2026-06-02,done"),
  file.path(dir, "visits.csv"))
msg <- tryCatch(import.visits(dir), error = conditionMessage)
stopifnot(identical(msg, "`visits.csv` has missing or duplicated `visit_id` values."))
unlink(dir, recursive = TRUE)
