# skipped: needs stringr: extractTable, maskEmails

txt = c(one = "Mail A@X.org or a@x.org, see https://ex.com/p. Ticket OPS-12",
        two = NA, three = "Due 2024-03-01 or 1.3.2024, not 31.02.2024")

# extractPatterns / extractPattern
stopifnot(all(c("email", "url", "isoDate") %in% names(extractPatterns())))
hits = extractPattern(c("a1 b2 a1", NA), "[a-z][0-9]")
stopifnot(identical(hits[[1]], c("a1", "b2")), length(hits[[2]]) == 0)

# extractEmails / extractUrls / extractTicketIds
stopifnot(identical(extractEmails(txt)[["one"]], "a@x.org"))
stopifnot(identical(extractUrls(txt)[["one"]], "https://ex.com/p"))
stopifnot(identical(extractTicketIds(txt)[["one"]], "OPS-12"))
stopifnot(length(extractTicketIds(txt, prefixes = "DEV")[["one"]]) == 0)

# extractDates
d = extractDates(txt)
stopifnot(identical(d[["three"]], as.Date("2024-03-01")), length(d[["two"]]) == 0)
stopifnot(length(extractDates(txt, unique = FALSE)[["three"]]) == 2)
