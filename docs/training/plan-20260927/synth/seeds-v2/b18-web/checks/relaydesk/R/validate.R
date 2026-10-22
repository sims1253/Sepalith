stopifnot(
  identical(requireString("  hi ", "title")$value, "hi"),
  grepl("required", requireString(NULL, "title")$error),
  grepl("empty", requireString("   ", "title")$error),
  grepl("exceeds 3", requireString("abcd", "title", maxChars = 3L)$error)
)

stopifnot(
  identical(requireInteger("4", "page")$value, 4L),
  grepl("whole number", requireInteger("2.5", "page")$error),
  grepl("between 1 and 10", requireInteger(11, "perPage", max = 10L)$error)
)

stopifnot(
  identical(requireChoice(" HIGH ", "priority", c("low", "high"))$value, "high"),
  identical(requireChoice("eu", "region", c("EU", "US"))$value, "EU"),
  grepl("one of: low, high", requireChoice("mid", "priority", c("low", "high"))$error)
)

stopifnot(
  identical(requireEmail("Ana@Example.ORG", "reporter")$value, "ana@example.org"),
  !is.null(requireEmail("ana@localhost", "reporter")$error),
  identical(optionalField(requireString, default = "x")(NULL, "d")$value, "x")
)

ok = validateTicketInput(list(title = "Printer", reporter = "a@b.co", queue = "it",
                              tags = list("hw", "hw")))
bad = validateTicketInput(list(title = "", reporter = "nope", tags = list("Bad Tag")))
stopifnot(
  isTRUE(ok$ok),
  identical(ok$data$priority, "normal"),
  identical(ok$data$tags, "hw"),
  isFALSE(bad$ok),
  identical(sort(names(bad$errors)), c("queue", "reporter", "tags", "title")),
  isFALSE(validateTicketInput("text")$ok)
)

stopifnot(
  identical(validatePaging(NULL, NULL)$data, list(page = 1L, perPage = 25L)),
  identical(validatePaging("3", "")$data$page, 3L),
  isFALSE(validatePaging("0", "500")$ok)
)
