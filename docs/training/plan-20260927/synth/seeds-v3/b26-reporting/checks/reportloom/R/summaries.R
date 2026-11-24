# S3 methods sourced into an environment need registering for dispatch.
for (cls in c("numeric", "factor", "logical", "Date", "data.frame")) {
  registerS3method("describe", cls, get(paste0("describe.", cls)))
}

stopifnot(identical(pluralise(1, "day"), "1 day"))
stopifnot(identical(pluralise(3, "value"), "3 values"))
stopifnot(identical(pluralise(2, "entry", "entries"), "2 entries"))

stopifnot(identical(list.phrase(character()), ""))
stopifnot(identical(list.phrase("a"), "a"))
stopifnot(identical(list.phrase(c("a", "b", "c")), "a, b and c"))
stopifnot(identical(list.phrase(c("a", "b"), "or"), "a or b"))

stopifnot(identical(describe(c(1, 2, 6, NA), "Score"),
                    "Score ranges from 1.0 to 6.0 (mean 3.0); 1 value missing."))
stopifnot(identical(describe.numeric(c(NA_real_, NA), "s"), "s has no observed values."))
stopifnot(identical(describe(c(0.25, 0.5), "p", digits = 2),
                    "p ranges from 0.25 to 0.50 (mean 0.38)."))

f <- factor(c("b", "a", "b", "c", "b", "a", NA, NA))
stopifnot(identical(describe(f, "Grade"),
                    "Grade is most often b (3), a (2) or c (1); 2 values missing."))
stopifnot(identical(describe(f, "Grade", top = 1), "Grade is most often b (3); 2 values missing."))
stopifnot(identical(describe(factor(NA), "g"), "g has no observed values."))

stopifnot(identical(describe(c(TRUE, FALSE, TRUE, TRUE), "Paid"),
                    "Paid is true for 3 of 4 (75.0%)."))
stopifnot(identical(describe(c(TRUE, NA), "q", digits = 0),
                    "q is true for 1 of 1 (100%); 1 value missing."))

dd <- as.Date(c("2024-01-01", "2024-01-31", NA))
stopifnot(identical(describe(dd, "Visit"),
                    "Visit runs from 2024-01-01 to 2024-01-31 (30 days); 1 value missing."))
stopifnot(identical(describe(as.Date("2024-01-01"), "v"),
                    "v runs from 2024-01-01 to 2024-01-01 (0 days)."))

df <- data.frame(n = c(1, 3), g = c("x", "x"), ok = c(TRUE, FALSE))
dsc <- describe(df)
stopifnot(identical(names(dsc), c("n", "g", "ok")))
stopifnot(identical(dsc[["g"]], "g is most often x (2)."))
stopifnot(identical(dsc[["ok"]], "ok is true for 1 of 2 (50.0%)."))
