people <- data.frame(
  name = c("Ann Lee", "ann  lee ", "Bo", "Bo", NA, "Cy"),
  city = c("Oslo", "oslo", "Rome", "Rome", "Oslo", "Rome"),
  phone = c(NA, "123", "9", NA, "5", NA),
  seen = as.Date(c("2024-01-05", "2024-03-01", "2024-02-01", "2023-01-01", "2024-01-01",
                   "2024-01-01")),
  stringsAsFactors = FALSE
)

k <- dedupe.key(people, c("name", "city"))
stopifnot(identical(k[1], k[2]), is.na(k[5]), k[3] != k[6])
stopifnot(dedupe.key(people, "name", normalize = FALSE)[1] != dedupe.key(people, "name",
                                                                        normalize = FALSE)[2])

m <- mark.duplicates(people, c("name", "city"))
stopifnot(identical(m$dup.group, c(1L, 1L, 2L, 2L, NA, 3L)))
stopifnot(identical(m$dup.size, c(2L, 2L, 2L, 2L, 1L, 1L)))

cols <- c("name", "city")
stopifnot(identical(rownames(dedupe(people, cols, "first")), c("1", "3", "5", "6")))
stopifnot(identical(rownames(dedupe(people, cols, "last")), c("2", "4", "5", "6")))
stopifnot(identical(rownames(dedupe(people, cols, "most.complete")), c("2", "3", "5", "6")))
stopifnot(identical(rownames(dedupe(people, cols, "newest", "seen")), c("2", "3", "5", "6")))
stopifnot(identical(rownames(dedupe(people, cols, "oldest", "seen")), c("1", "4", "5", "6")))
stopifnot(inherits(try(dedupe(people, cols, "newest"), silent = TRUE), "try-error"))

stopifnot(identical(collapse.values(c(NA, 3, 1), "first"), 3))
stopifnot(identical(collapse.values(c(2, 3, NA), "last"), 3))
stopifnot(identical(collapse.values(c(2, 3, 1), "min"), 1))
stopifnot(identical(collapse.values(c(2, 3, 1), "max"), 3))
stopifnot(identical(collapse.values(c("a", "b", "a"), "paste"), "a; b"))
stopifnot(is.na(collapse.values(c(NA, NA))))

co <- coalesce.duplicates(people, cols, rules = c(seen = "max", phone = "paste"))
stopifnot(nrow(co) == 4L, identical(co$name, c("Ann Lee", "Bo", NA, "Cy")))
stopifnot(identical(co$phone, c("123", "9", "5", NA)))
stopifnot(identical(co$seen[1], as.Date("2024-03-01")))

s <- dedupe.summary(people, cols)
stopifnot(identical(names(s), c("n", "n.unique", "n.duplicated", "n.incomplete",
                                "largest.group")))
stopifnot(identical(s$n, 6L), identical(s$n.unique, 3L), identical(s$n.duplicated, 2L))
stopifnot(identical(s$n.incomplete, 1L), identical(s$largest.group, 2L))

stopifnot(inherits(try(dedupe(people, c("name", "zip")), silent = TRUE), "try-error"))
stopifnot(inherits(try(dedupe(people, cols, "newest", "when"), silent = TRUE), "try-error"))
stopifnot(inherits(try(mark.duplicates(people, "zip"), silent = TRUE), "try-error"))

ra <- list(name = "Ann  Lee", email = "ANN@x.org", phone = "+49 30 123", postcode = "sw1a 1aa",
           birth = "1990-01-02")
rb <- list(name = "ann lee", email = "ann@x.org ", phone = "4930123", postcode = "SW1A1AA",
           birth = "1990-01-03")
cr <- compare.records(ra, rb)
stopifnot(identical(names(cr), c("fields", "score")))
stopifnot(identical(cr$fields, c(name = TRUE, email = TRUE, phone = TRUE, postcode = TRUE,
                                 birth = FALSE)))
stopifnot(isTRUE(all.equal(cr$score, 5 / 6)))
rc <- list(name = NA, email = "b@y.org", phone = NA, postcode = NA, birth = NA)
cr2 <- compare.records(ra, rc)
stopifnot(identical(unname(is.na(cr2$fields)), c(TRUE, FALSE, TRUE, TRUE, TRUE)), cr2$score == 0)
none <- list(name = NA, email = NA, phone = NA, postcode = NA, birth = NA)
stopifnot(is.na(compare.records(none, none)$score))
