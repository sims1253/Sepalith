stopifnot(identical(needs.quoting(c("a", NA), "sql"), c(TRUE, FALSE)))
stopifnot(identical(needs.quoting(c("a", "a,b", " a", "a;b"), "csv"), c(FALSE, TRUE, TRUE, FALSE)))
stopifnot(identical(needs.quoting(c("file.txt", "my file"), "shell"), c(FALSE, TRUE)))
stopifnot(identical(needs.quoting(c("abc", "a.c"), "regex"), c(FALSE, TRUE)))

stopifnot(identical(enquote(c("O'Hara", NA)), c("'O''Hara'", "NULL")))
stopifnot(identical(enquote(c("plain", "a,b", "say \"x\"", NA), "csv"),
  c("plain", "\"a,b\"", "\"say \"\"x\"\"\"", "")))
stopifnot(identical(enquote("a", "csv", always = TRUE), "\"a\""))
stopifnot(identical(enquote(c("file.txt", "it's"), "shell"), c("file.txt", "'it'\\''s'")))
stopifnot(identical(enquote("a.b", "regex"), "\\Qa.b\\E"))
stopifnot(grepl(enquote("x\\E.y", "regex"), "zx\\E.yz", perl = TRUE))

stopifnot(identical(dequote(c("'O''Hara'", "NULL")), c("O'Hara", NA)))
stopifnot(identical(dequote(c("plain", "\"say \"\"x\"\"\"", ""), "csv"),
  c("plain", "say \"x\"", NA)))
stopifnot(identical(dequote(c("file.txt", "'it'\\''s'"), "shell"), c("file.txt", "it's")))
stopifnot(identical(dequote(enquote("x\\E.y", "regex"), "regex"), "x\\E.y"))

stopifnot(identical(quoted.split("a,\"b,c\",\"say \"\"hi\"\"\",,d"),
  c("a", "b,c", "say \"hi\"", "", "d")))
stopifnot(identical(quoted.split("x;'y;z'", sep = ";", quote = "'"), c("x", "y;z")))
stopifnot(identical(quoted.split(""), ""))
stopifnot(inherits(tryCatch(quoted.split("\"open"), warning = identity), "warning"))

stopifnot(identical(quoted.join(c("a", "b,c", NA, "q\"")), "a,\"b,c\",,\"q\"\"\""))
stopifnot(identical(quoted.split(quoted.join(c("x", "y,z"))), c("x", "y,z")))
stopifnot(identical(quoted.join(c("a", "b"), sep = "\t"), "a\tb"))
stopifnot(inherits(tryCatch(enquote(1), error = identity), "error"))
stopifnot(inherits(tryCatch(dequote(1, "csv"), error = identity), "error"))
