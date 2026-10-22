# guessDelimiter
stopifnot(identical(guessDelimiter(c("a;b;c", "1;2;3")), ";"))
stopifnot(identical(guessDelimiter(c("a|b", "1|2", "")), "|"))
stopifnot(identical(guessDelimiter(c("abc", "def")), ","))
stopifnot(inherits(try(guessDelimiter(character(0)), silent = TRUE), "try-error"))

# looks* helpers ignore NA and the `na` strings
stopifnot(looksNumeric(c("1", NA_character_)))
stopifnot(looksNumeric(c("1.5", "-2e3", ".5", "NA")))
stopifnot(!looksNumeric(c("1.5", "abc")))
stopifnot(looksInteger(c("12", NA, "-3")))
stopifnot(!looksInteger(c("1.2", "3")))
stopifnot(looksDate(c("2026-01-02", NA, "")))
stopifnot(!looksDate(c("2026-13-40")))
stopifnot(looksLogical(c("true", "F", NA)))
stopifnot(!looksLogical(c(NA_character_, "")))

# guessColumnTypes
types <- guessColumnTypes(list(
  a = c("1", "2"), b = c("1.5", NA), c = c("2026-01-01", "2026-02-01"),
  d = c("T", "F"), e = c("x", "y")
))
stopifnot(identical(unname(types), c("integer", "numeric", "Date", "logical", "character")))
stopifnot(identical(names(types), c("a", "b", "c", "d", "e")))

# readCsvSniffed and profileCsv
tmp <- tempfile(fileext = ".csv")
writeLines(c("id;score;day;name", "1;2.5;2026-01-01;ann", "2;NA;2026-01-02;", "3;4;;bo"), tmp)
df <- readCsvSniffed(tmp)
stopifnot(is.integer(df$id), is.numeric(df$score), inherits(df$day, "Date"))
stopifnot(identical(attr(df, "sniffed")$delimiter, ";"))
stopifnot(is.na(df$score[2]))
prof <- profileCsv(tmp)
stopifnot(identical(prof$type, c("integer", "numeric", "Date", "character")))
stopifnot(prof$missing[2] == round(1 / 3, 3), prof$distinct[1] == 3L)
stopifnot(identical(prof$example[4], "ann"), attr(prof, "rows") == 3L)
unlink(tmp)
