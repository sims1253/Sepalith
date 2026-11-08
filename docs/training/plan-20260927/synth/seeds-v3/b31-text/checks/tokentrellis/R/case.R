sp = splitIdentifier(c("parseHTTPResponse", "user_id-v2", "  Total Count ", NA))
stopifnot(identical(sp[[1]], c("parse", "http", "response")))
stopifnot(identical(sp[[2]], c("user", "id", "v2")))
stopifnot(identical(sp[[3]], c("total", "count")), identical(sp[[4]], character()))

ids = c("parseHTTPResponse", "user.id", NA)
stopifnot(identical(convertCase(ids, "snake"), c("parse_http_response", "user_id", NA)))
stopifnot(identical(convertCase(ids, "kebab"), c("parse-http-response", "user-id", NA)))
stopifnot(identical(convertCase("Max Value", "dot"), "max.value"))
stopifnot(identical(convertCase(ids, "camel"), c("parseHttpResponse", "userId", NA)))
stopifnot(identical(convertCase(ids, "pascal"), c("ParseHttpResponse", "UserId", NA)))
stopifnot(identical(convertCase("maxValue", "constant", prefix = "opt"), "OPT_MAX_VALUE"))

stopifnot(identical(
  detectCase(c("a_b", "a-b", "a.b", "aB", "AbC", "A_B", "a b", "ab", NA)),
  c("snake", "kebab", "dot", "camel", "pascal", "constant", "other", "snake", NA)
))
stopifnot(identical(detectCase(character()), character()))

stopifnot(identical(titleCase("the lord OF the rings"), "The Lord of the Rings"))
stopifnot(identical(titleCase(c("what it is for", NA, "  ")), c("What It Is For", NA, "")))
stopifnot(identical(titleCase("a tale of two", minor = character()), "A Tale Of Two"))

cp = caseProfile(c("AbC12!", "", NA))
stopifnot(identical(cp$nUpper, c(2L, 0L, 0L)), identical(cp$nOther, c(1L, 0L, 0L)))
stopifnot(identical(cp$nDigit, c(2L, 0L, 0L)), cp$shareUpper[1] == 2 / 3)
stopifnot(is.na(cp$shareUpper[2]))

df = data.frame(firstName = 1, `last-name` = 2, `__` = 3, check.names = FALSE)
rc = renameColumns(df)
stopifnot(identical(names(rc), c("first_name", "last_name", "__")))
stopifnot(identical(attr(rc, "renamed"), c(firstName = "first_name", `last-name` = "last_name")))
stopifnot(identical(names(renameColumns(df, "camel")), c("firstName", "lastName", "__")))
dup = suppressWarnings(renameColumns(data.frame(aB = 1, a_b = 2)))
stopifnot(identical(names(dup), c("a_b", "a_b_1")))
