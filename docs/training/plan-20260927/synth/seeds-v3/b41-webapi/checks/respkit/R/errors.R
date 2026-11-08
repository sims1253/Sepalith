p <- parse.error.body(list(type = "about:blank", title = "Bad", detail = "x missing"))
stopifnot(identical(p, list(code = "about:blank", message = "Bad", detail = "x missing")))
j <- parse.error.body(list(errors = list(list(code = "E1", title = "T", detail = "D"))), "jsonapi")
stopifnot(identical(j, list(code = "E1", message = "T", detail = "D")))
stopifnot(all(is.na(unlist(parse.error.body(list(), "jsonapi")))))
g <- parse.error.body(list(error = list(status = "NOT_FOUND", message = "gone",
  details = list(list(reason = "deleted")))), "google")
stopifnot(identical(g, list(code = "NOT_FOUND", message = "gone", detail = "deleted")))
q <- parse.error.body(list(errors = list(list(message = "no", path = list("user", "name"),
  extensions = list(code = "FORBIDDEN")))), "graphql")
stopifnot(identical(q, list(code = "FORBIDDEN", message = "no", detail = "user.name")))
pl <- parse.error.body(" Service down ", "plain")
stopifnot(identical(pl$message, "Service down"), is.na(pl$code), is.na(pl$detail))
stopifnot(is.na(parse.error.body("", "plain")$message))

v1 <- validation.errors(list(errors = list(
  list(source = list(pointer = "/data/attributes/name"), detail = "is blank"))), "jsonapi")
stopifnot(identical(v1$field, "name"), identical(v1$message, "is blank"))
v2 <- validation.errors(list(errors = list(name = list("blank", "short"), age = list("nan"))),
  "rails")
stopifnot(identical(v2$field, c("name", "name", "age")), identical(v2$message[3], "nan"))
v3 <- validation.errors(list(email = list("bad"), non_field_errors = list("x")), "django")
stopifnot(identical(v3$field, "email"), identical(v3$message, "bad"))
v4 <- validation.errors(list(error = list(details = list(list(fieldViolations = list(
  list(field = "a", description = "d1"), list(field = "b", description = "d2")))))), "google")
stopifnot(identical(v4$field, c("a", "b")), identical(v4$message, c("d1", "d2")))
stopifnot(nrow(validation.errors(list(), "jsonapi")) == 0L)

e <- error.condition(404L, p, "http://h/x")
stopifnot(inherits(e, "respkit_not_found"), inherits(e, "respkit_http_error"))
stopifnot(identical(conditionMessage(e), "HTTP 404: Bad (x missing) [http://h/x]"))
e2 <- error.condition(503L, pl)
stopifnot(inherits(e2, "respkit_server_error"), identical(conditionMessage(e2),
  "HTTP 503: Service down"))
e3 <- error.condition(418L, list(code = NA, message = NA, detail = NA))
stopifnot(identical(class(e3), c("respkit_http_error", "error", "condition")))
stopifnot(identical(conditionMessage(e3), "HTTP 418: no message"))
r <- tryCatch(stop(error.condition(429L, p)), respkit_rate_limited = function(c) c$status)
stopifnot(identical(r, 429L))

ok <- check.response(list(status = 200L, content_type = "application/json",
  body = charToRaw("{}")))
stopifnot(all(ok), length(attr(ok, "problems")) == 0L)
bad <- check.response(list(status = 500L, content_type = "text/html", body = raw(20)),
  max.bytes = 10)
stopifnot(identical(as.vector(bad), c(FALSE, FALSE, FALSE, TRUE)))
stopifnot(identical(attr(bad, "problems")[1], "Check `status` failed: got 500."))
stopifnot(identical(attr(bad, "problems")[3], "Check `size` failed: got 20."))
em <- check.response(list(status = 204L, content_type = NULL, body = raw(0)), type = NULL)
stopifnot(identical(as.vector(em), c(TRUE, TRUE, TRUE, FALSE)))
stopifnot(all(check.response(list(status = 204L, body = raw(0)), type = NULL, empty.ok = TRUE)))

f <- error.families(c(401L, 404L, 400L, 429L, 502L, 403L, 410L), paste0("u", 1:7))
stopifnot(identical(f$family, c("client_auth", "client_missing", "client_other",
  "rate_limited", "server")))
stopifnot(identical(f$n, c(2L, 2L, 1L, 1L, 1L)))
stopifnot(identical(f$example, c("u1", "u2", "u3", "u4", "u5")))
f0 <- error.families(integer(0), character(0))
stopifnot(all(f0$n == 0L), all(is.na(f0$example)))
