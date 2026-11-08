pats <- entity_patterns()
stopifnot(identical(names(pats), c("email", "url", "phone", "hashtag")))
stopifnot(grepl(pats$hashtag, "a #tag", perl = TRUE))

stopifnot(identical(mask_keep("secret"), "s****t"))
stopifnot(identical(mask_keep("ab"), "**"))
stopifnot(identical(mask_keep(c("abcdef", "xy"), keep_head = 2L, keep_tail = 0L, char = "#"),
  c("ab####", "##")))

stopifnot(identical(replacement_for("a@b.co", "email"), "[EMAIL]"))
stopifnot(identical(replacement_for("abc", "x", "stars"), "***"))
stopifnot(identical(replacement_for("abcd", "x", "ends"), "a**d"))
stopifnot(identical(replacement_for("abcd", "x", "drop"), ""))
stopifnot(identical(replacement_for("ab", "url", "digest"), "[URL:0125]"))

docs <- c("mail a@b.co, see https://ex.org/p. Call +49 30 1234 #hot", "plain")
r <- redact_entities(docs)
stopifnot(identical(r[[1]], "mail [EMAIL], see [URL]. Call [PHONE] [HASHTAG]"))
stopifnot(identical(r[[2]], "plain"))
stopifnot(identical(redact_entities("x #ab y", types = "hashtag", style = "stars"), "x *** y"))
stopifnot(identical(redact_entities("a@b.co #t", types = "email", style = "drop"), " #t"))

stopifnot(identical(mask_email("alice@example.org"), "a****@e******.org"))
stopifnot(identical(mask_email(c("bob@x.io", "nope"), keep = 2L), c("bo*@*.io", NA)))
stopifnot(identical(mask_email("alice@example.org", keep_domain = TRUE), "a****@example.org"))
stopifnot(identical(mask_email("al@ex.org", keep_tld = FALSE), "a*@e*.***"))
stopifnot(inherits(tryCatch(mask_email("a@b.c", keep = -1), error = identity), "error"))

rs <- redaction_summary(docs, r)
stopifnot(rs$n_docs == 2L, rs$n_changed == 1L, rs$share_changed == 0.5)
stopifnot(rs$chars_removed == nchar(docs[[1]]) - nchar(r[[1]]))
stopifnot(is.na(redaction_summary(character(), character())$share_changed))
stopifnot(identical(rs$tags, c(email = 1L, url = 1L, phone = 1L, hashtag = 1L)))
stopifnot(identical(redaction_summary("a", "[URL] [URL]", types = "url")$tags, c(url = 2L)))
