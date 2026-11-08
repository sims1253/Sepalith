stopifnot(identical(isValidEmail(c("a@b.de", "a@b", NA, " x.y@z.org ")),
                    c(TRUE, FALSE, FALSE, TRUE)))
stopifnot(identical(isValidEmail(NA, allowMissing = TRUE), TRUE))

stopifnot(identical(isValidPhone(c("+49 (30) 123-4567", "12", NA)), c(TRUE, FALSE, FALSE)))
stopifnot(identical(isValidPhone(NA, allowMissing = TRUE), TRUE))

stopifnot(identical(isValidIsoDate(c("2024-02-29", "2023-02-29", "1/2/2024", NA)),
                    c(TRUE, FALSE, FALSE, FALSE)))
stopifnot(identical(isValidIsoDate(NA, allowMissing = TRUE), TRUE))

stopifnot(identical(isValidPostcode(c("10115", "1011", NA), "DE"), c(TRUE, FALSE, FALSE)))
stopifnot(identical(isValidPostcode(c("1012 ab", "0123AB"), "NL"), c(TRUE, FALSE)))
stopifnot(identical(isValidPostcode(c("SW1A 1AA", "M1 1AE", "123"), "UK"),
                    c(TRUE, TRUE, FALSE)))
stopifnot(identical(isValidPostcode(c("12345-6789", "1234"), "US"), c(TRUE, FALSE)))

fi <- fieldIssues("f", c("a", "b", "c"), c(TRUE, FALSE, FALSE), "bad")
stopifnot(identical(names(fi), c("row", "field", "value", "problem")))
stopifnot(identical(fi$row, 2:3), identical(fi$value, c("b", "c")))
stopifnot(nrow(fieldIssues("f", "a", TRUE, "bad")) == 0L)

contacts <- data.frame(
  id = c("a1", "a2", "a2", NA),
  email = c("a@x.de", "bad", "c@x.de", "d@x.de"),
  phone = c("030 1234567", NA, "1", NA),
  postcode = c("10115", "10117", "999", "80331"),
  birthDate = c("1990-01-01", NA, "1990-13-01", NA),
  stringsAsFactors = FALSE
)
iss <- validateContacts(contacts)
stopifnot(identical(names(iss), c("row", "field", "value", "problem")))
stopifnot(identical(iss$row, c(2L, 3L, 3L, 3L, 3L, 4L)))
stopifnot(identical(iss$field, c("email", "id", "phone", "postcode", "birthDate", "id")))
stopifnot(identical(iss$problem[2], "missing or duplicated"))
iss_strict <- validateContacts(contacts, optional = character())
stopifnot(sum(iss_strict$field == "phone") == 3L, sum(iss_strict$field == "birthDate") == 3L)
stopifnot(inherits(try(validateContacts(contacts[, 1:3]), silent = TRUE), "try-error"))

v <- validateContact(list(id = "x", email = "x@y.com", postcode = "10115"))
stopifnot(identical(names(v), c("ok", "problems")))
stopifnot(identical(names(v$ok), c("id", "email", "phone", "postcode", "birthDate")))
stopifnot(all(v$ok), length(v$problems) == 0L)
v2 <- validateContact(list(id = "", email = "nope", phone = "12", postcode = "SW1A 1AA",
                           birthDate = "2020-02-30"), country = "UK")
stopifnot(identical(unname(v2$ok), c(FALSE, FALSE, FALSE, TRUE, FALSE)))
stopifnot(identical(names(v2$problems), c("id", "email", "phone", "birthDate")))
stopifnot(identical(v2$problems[["id"]], "`id` is required."))
stopifnot(identical(v2$problems[["phone"]], "`phone` has an invalid format: 12"))
stopifnot(identical(v2$problems[["birthDate"]], "`birthDate` has an invalid format: 2020-02-30"))
v3 <- validateContact(list(id = "a", email = "a@b.de", postcode = "1"))
stopifnot(identical(unname(v3$problems), "`postcode` has an invalid format: 1"))
