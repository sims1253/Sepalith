stopifnot(abs(trapezoid.rule(function(x) x^2, 0, 1, 1000L) - 1 / 3) < 1e-6)
stopifnot(abs(trapezoid.rule(function(x) 2 * x + 1, 0, 2, 3L) - 6) < 1e-12)

stopifnot(abs(midpoint.rule(function(x) x^2, 0, 1, 1000L) - 1 / 3) < 1e-6)
stopifnot(abs(midpoint.rule(identity, -1, 3, 1L) - 4) < 1e-12)

stopifnot(abs(simpson.rule(function(x) x^3, 0, 2, 2L) - 4) < 1e-12)
stopifnot(abs(simpson.rule(sin, 0, pi, 9L) - 2) < 1e-3)

gl <- gauss.legendre.nodes(3L)
stopifnot(all.equal(gl$nodes, c(-sqrt(3 / 5), 0, sqrt(3 / 5))))
stopifnot(all.equal(gl$weights, c(5, 8, 5) / 9))
stopifnot(all.equal(sum(gauss.legendre.nodes(10L)$weights), 2))
stopifnot(identical(gauss.legendre.nodes(1L), list(nodes = 0, weights = 2)))

stopifnot(abs(gauss.legendre.rule(exp, 0, 1, 10L) - (exp(1) - 1)) < 1e-12)

as1 <- adaptive.simpson(exp, 0, 1)
stopifnot(abs(as1 - (exp(1) - 1)) < 1e-9, is.null(attr(as1, "depth.hit")))
as2 <- adaptive.simpson(function(x) sin(1 / (x + 0.01)), 0, 1, tol = 1e-14, max.depth = 2L)
stopifnot(isTRUE(attr(as2, "depth.hit")))

qi <- quad.integrate("dnorm", -3, 3, rule = "gauss", n = 30L, check = TRUE)
stopifnot(abs(qi - (pnorm(3) - pnorm(-3))) < 1e-10, attr(qi, "abs.diff") < 1e-8)
stopifnot(is.null(attributes(quad.integrate(cos, 0, 1, rule = "midpoint"))))

ro <- romberg(exp, 0, 1)
stopifnot(abs(ro - (exp(1) - 1)) < 1e-10, isTRUE(attr(ro, "converged")))
stopifnot(!attr(romberg(function(x) sqrt(x), 0, 1, tol = 0, max.iter = 4L), "converged"))
stopifnot(inherits(try(romberg(exp, 0, 1, max.iter = 1L), silent = TRUE), "try-error"))
