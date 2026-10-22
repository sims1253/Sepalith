# skipped: needs rlang (error paths of checkCharacter, checkPadChar and
#   strTruncate signal classed rlang errors)
stopifnot(identical(strSquish("  too   many\tspaces "), "too many spaces"))
stopifnot(identical(strSquish(NA_character_), NA_character_))

stopifnot(identical(strPadLeft(c("7", "42", "512"), 3), c("  7", " 42", "512")))
stopifnot(identical(strPadLeft(c("a", NA), 2, pad = "0"), c("0a", NA)))
stopifnot(identical(strPadRight(c("7", "toolong"), 3, pad = "."), c("7..", "toolong")))

stopifnot(identical(strTruncate(c("short", "a considerably longer label", NA), 10),
  c("short", "a consi...", NA)))
stopifnot(identical(strTruncate("abcdef", 4, ellipsis = "~"), "abc~"))

stopifnot(identical(strToSnake(c("monthlyNetTotal", "Net Total.2025")),
  c("monthly_net_total", "net_total_2025")))
stopifnot(identical(strToSnake("__already__snake__"), "already_snake"))
stopifnot(identical(strToCamel(c("monthly_net_total", "", NA)), c("monthlyNetTotal", "", NA)))

stopifnot(identical(strCountMatches(c("a.b.c", "abc", NA), ".", fixed = TRUE), c(2L, 0L, NA)))
stopifnot(identical(strCountMatches("2025-01-31", "[0-9]+"), 3L))

stopifnot(identical(checkCharacter("a"), "a"), identical(checkPadChar("0"), "0"))
