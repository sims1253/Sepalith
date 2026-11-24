docs <- c(
  "Mail ann@example.org or ann@example.org, see https://ex.org/a. #news on 2024/05/01",
  "Call +49 (30) 123-456 today #news #Sale",
  NA
)

m <- extract_matches(docs, "#[a-z]+")
stopifnot(identical(m[[1]], "#news"), identical(m[[3]], character()))
stopifnot(length(extract_matches(c("aa", "b"), "a", unique = FALSE)[[1]]) == 2L)

em <- extract_entities(docs, "email")
stopifnot(identical(em[[1]], "ann@example.org"), length(em[[2]]) == 0L)
stopifnot(length(extract_entities(docs, "email", unique = FALSE)[[1]]) == 2L)
stopifnot(identical(extract_entities(docs, "url")[[1]], "https://ex.org/a"))
stopifnot(identical(extract_entities(docs, "phone")[[2]], "+4930123456"))
stopifnot(identical(extract_entities(docs, "hashtag", lower = TRUE)[[2]], c("news", "sale")))
stopifnot(identical(extract_entities(docs, "date")[[1]], "2024-05-01"))

et <- entity_table(docs, types = c("hashtag", "email"))
stopifnot(identical(et$doc, c(1L, 1L, 2L, 2L)))
stopifnot(identical(et$type, c("hashtag", "email", "hashtag", "hashtag")))
stopifnot(identical(entity_table(docs, "phone")$value, "+4930123456"))
stopifnot(nrow(entity_table(docs)) == 7L)

ec <- entity_counts(docs)
stopifnot(identical(ec$n_email, c(2L, 0L, 0L)), identical(ec$n_date, c(1L, 0L, 0L)))
stopifnot(identical(ec$n_total, c(5L, 3L, 0L)))
stopifnot(identical(entity_counts(docs, unique = TRUE)$n_email, c(1L, 0L, 0L)))
stopifnot(inherits(tryCatch(entity_counts(1), error = identity), "error"))

fe <- first_entities(docs[[1]])
stopifnot(identical(fe[["email"]], "ann@example.org"), identical(fe[["date"]], "2024-05-01"))
stopifnot(is.na(fe[["phone"]]), identical(fe[["url"]], "https://ex.org/a"))
stopifnot(identical(first_entities("x #t", default = "")[["hashtag"]], "t"))
stopifnot(all(is.na(first_entities(NA_character_))))

lm <- locate_matches(c("abc key def key", "none"), "key", context = 2L)
stopifnot(identical(lm$start, c(5L, 13L)), identical(lm$end, c(7L, 15L)))
stopifnot(identical(lm$before, c("c ", "f ")), identical(lm$after, c(" d", "")))
stopifnot(nrow(locate_matches("zzz", "q")) == 0L)
