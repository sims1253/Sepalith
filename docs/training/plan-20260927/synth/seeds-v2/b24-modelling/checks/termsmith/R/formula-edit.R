err <- tryCatch(check_formula("y ~ x"), error = function(e) e)
stopifnot(inherits(err, "termsmith_bad_formula"), grepl("not character", conditionMessage(err)))
stopifnot(identical(check_formula(y ~ x), y ~ x))

env <- new.env()
f <- stats::as.formula("y ~ a + b", env = env)
added <- add_terms(f, c("c", "a"))
stopifnot(identical(deparse1(added), "y ~ a + b + c"), identical(environment(added), env))
stopifnot(identical(deparse1(add_terms(y ~ 0 + a, "b")), "y ~ a + b - 1"))

stopifnot(identical(deparse1(drop_terms(f, "b")), "y ~ a"))
stopifnot(identical(deparse1(drop_terms(f, c("a", "b"))), "y ~ 1"))
stopifnot(identical(drop_terms(f, "zz"), f))

stopifnot(identical(deparse1(swap_response(f, "log(z)")), "log(z) ~ a + b"))
stopifnot(identical(deparse1(swap_response(~x, quote(w))), "w ~ x"))

stopifnot(identical(lhs_vars(cbind(s, t) ~ x), c("s", "t")), identical(lhs_vars(~x), character()))
stopifnot(identical(rhs_vars(y ~ x + log(w) + x:w), c("x", "w")))
stopifnot(identical(rhs_vars(y ~ .), character()))

iw <- attr(stats::terms(interact_with(y ~ a + b, "g")), "term.labels")
stopifnot(all(c("g", "a:g", "b:g") %in% iw))
stopifnot(identical(deparse1(interact_with(y ~ g, "g")), "y ~ g"))

stopifnot(identical(formula_signature(y ~ a + b + a:b), formula_signature(y ~ b:a + b + a)))
stopifnot(identical(formula_signature(~1), "~ 1"))
stopifnot(identical(formula_signature(y ~ a - 1), "y ~ a - 1"))

dat <- data.frame(y = 1, p = 2, q = 3)
stopifnot(identical(deparse1(expand_dot(y ~ ., dat)), "y ~ p + q"))
stopifnot(identical(expand_dot(y ~ p, dat), y ~ p))
