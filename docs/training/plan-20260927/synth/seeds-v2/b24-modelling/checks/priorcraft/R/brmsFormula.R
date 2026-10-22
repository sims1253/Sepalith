# skipped: needs brms: asBrmsFormula()
stopifnot(identical(randomInterceptTerm("site"), "(1 | site)"))
stopifnot(inherits(try(randomInterceptTerm(""), silent = TRUE), "try-error"))
stopifnot(identical(randomSlopeTerm(c("dose", "age"), "site"), "(1 + dose + age | site)"))
stopifnot(identical(randomSlopeTerm("dose", "site", correlated = FALSE), "(1 + dose || site)"))

stopifnot(identical(validateGroupSpec(NULL), list()))
stopifnot(inherits(try(validateGroupSpec(list(a = 1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(validateGroupSpec(list("x")), silent = TRUE), "try-error"))

spec <- formulaSpec("y", c("dose", "dose"), groups = list(site = "dose", day = character()))
stopifnot(inherits(spec, "pcFormulaSpec"), identical(spec$fixed, "dose"))
stopifnot(identical(spec$random, c("(1 + dose | site)", "(1 | day)")))
stopifnot(inherits(try(formulaSpec("y", "y"), silent = TRUE), "try-error"))
warned <- tryCatch(formulaSpec("y", groups = list(g = "x")), warning = function(w) "warned")
stopifnot(identical(warned, "warned"))

spec2 <- addDistParam(spec, "sigma", character())
stopifnot(identical(spec2$dpars$sigma, "1"))
stopifnot(inherits(try(addDistParam(spec, "mu", "x"), silent = TRUE), "try-error"))

stopifnot(identical(specRhs(spec), "dose + (1 + dose | site) + (1 | day)"))
stopifnot(identical(specRhs(formulaSpec("y")), "1"))
stopifnot(identical(specRhs(formulaSpec("y", intercept = FALSE)), "0"))
stopifnot(identical(specRhs(formulaSpec("y", "x", intercept = FALSE)), "0 + x"))

f <- asStanarmFormula(formulaSpec("y", "x"))
stopifnot(inherits(f, "formula"), identical(deparse(f), "y ~ x"))
stopifnot(inherits(try(asStanarmFormula(spec2), silent = TRUE), "try-error"))

out <- capture.output(res <- print(spec2))
stopifnot(identical(res, spec2), length(out) == 3, identical(out[3], "  sigma ~ 1"))
