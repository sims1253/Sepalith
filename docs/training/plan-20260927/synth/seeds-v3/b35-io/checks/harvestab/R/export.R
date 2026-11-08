households <- data.frame(hh_id = c("H2", "H1"), village = c("Oro", "Ama"), size = c(3L, NA),
  visited = c(TRUE, FALSE), extra = 1:2, stringsAsFactors = FALSE)
persons <- data.frame(person_id = c("P3", "P1", "P2"), hh_id = c("H2", "H2", "H1"),
  age = c(40L, 12L, 70L), sex = c("f", "m", "f"), stringsAsFactors = FALSE)
visits <- data.frame(visit_id = c("V2", "V1"), hh_id = c("H1", "H2"),
  date = as.Date(c("2026-06-02", "2026-06-01")), outcome = c("done", "refused"),
  stringsAsFactors = FALSE)
dir <- tempfile("export")
dir.create(dir)

p <- export.households(households, dir)
stopifnot(identical(p, file.path(dir, "households.csv")))
stopifnot(identical(readLines(p), c("\"hh_id\",\"village\",\"size\",\"visited\"",
  "\"H1\",\"Ama\",,FALSE", "\"H2\",\"Oro\",3,TRUE")))
msg <- tryCatch(export.households(households, dir), error = conditionMessage)
stopifnot(grepl("already exists", msg))
msg <- tryCatch(export.households(households[c(NA, 1), ], dir, TRUE), error = conditionMessage)
stopifnot(identical(msg, "`hh_id` must be unique and non-missing in `households`."))
msg <- tryCatch(export.households(households[1:2], dir), error = conditionMessage)
stopifnot(identical(msg, "`households` is missing column(s): size, visited"))

p <- export.persons(persons, dir)
stopifnot(identical(utils::read.csv(p)$person_id, c("P2", "P1", "P3")))
msg <- tryCatch(export.persons(persons[-4], dir, overwrite = TRUE), error = conditionMessage)
stopifnot(identical(msg, "`persons` is missing column(s): sex"))

dup <- persons
dup$person_id[2] <- "P3"
msg <- tryCatch(export.persons(dup, dir, overwrite = TRUE), error = conditionMessage)
stopifnot(identical(msg, "`person_id` must be unique and non-missing in `persons`."))
p <- export.visits(visits, dir)
stopifnot(identical(utils::read.csv(p)$visit_id, c("V1", "V2")))
stopifnot(identical(utils::read.csv(p)$date, c("2026-06-01", "2026-06-02")))
msg <- tryCatch(export.visits(visits[-1], dir, overwrite = TRUE), error = conditionMessage)
stopifnot(identical(msg, "`visits` is missing column(s): visit_id"))
msg <- tryCatch(export.visits(visits[c(1, 1), ], dir, overwrite = TRUE), error = conditionMessage)
stopifnot(identical(msg, "`visit_id` must be unique and non-missing in `visits`."))

out <- file.path(dir, "all")
m <- export.all(list(households = households, persons = persons, visits = visits), out)
stopifnot(identical(m$table, c("households", "persons", "visits")))
stopifnot(identical(m$file, c("households.csv", "persons.csv", "visits.csv")))
stopifnot(identical(m$rows, c(2L, 3L, 2L)), all(nchar(m$md5) == 32))
stopifnot(identical(utils::read.csv(file.path(out, "manifest.csv"))$table, m$table))
msg <- tryCatch(export.all(list(households = households), out), error = conditionMessage)
stopifnot(identical(msg, "`tables` is missing element(s): persons, visits"))

writeLines("keep", file.path(out, "notes.txt"))
removed <- export.clean(out)
stopifnot(identical(removed, c(m$file, "manifest.csv")))
stopifnot(identical(list.files(out), "notes.txt"), length(export.clean(out)) == 0)
unlink(dir, recursive = TRUE)
