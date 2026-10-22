quad2 <- function(x) sum((x - c(1, -2))^2) + x[1] * x[2]

g <- num.gradient(quad2, c(0, 0))
stopifnot(all.equal(g, c(-2 + 0, 4 + 0), tolerance = 1e-6))
stopifnot(length(num.gradient(function(x) sum(x^2), c(1, 2, 3))) == 3L)

H <- num.hessian(quad2, c(0.3, 0.7))
stopifnot(isSymmetric(H), all.equal(H, matrix(c(2, 1, 1, 2), 2), tolerance = 1e-4))

fit <- optim.safe(c(0, 0), quad2)
stopifnot(fit$converged, all.equal(fit$par, c(8, -10) / 3, tolerance = 1e-4))
bad <- optim.safe(c(0, 0), function(x) stop("boom"))
stopifnot(!bad$converged, identical(bad$message, "boom"), all(is.na(bad$par)))
box <- optim.safe(c(0.5, 0.5), quad2, method = "L-BFGS-B", lower = 0, upper = 1)
stopifnot(box$converged, all(box$par >= 0 & box$par <= 1))

ms <- multi.start(rbind(c(0, 0), c(5, 5), c(-3, 2)), quad2)
stopifnot(ms$n.converged == 3L, all.equal(ms$value, quad2(c(8, -10) / 3), tolerance = 1e-6))
stopifnot(inherits(try(multi.start(matrix(0, 1, 2), function(x) stop("x")), silent = TRUE),
  "try-error"))

set.seed(1)
xn <- stats::rnorm(400, 3, 2)
mn <- mle.normal(c(xn, NA, Inf))
stopifnot(abs(mn$estimate[["mean"]] - mean(xn)) < 1e-3)
stopifnot(abs(mn$estimate[["sd"]] - sqrt(mean((xn - mean(xn))^2))) < 1e-3)
stopifnot(all(mn$se > 0), all.equal(mn$se[["mean"]], mn$estimate[["sd"]] / 20, tolerance = 1e-2))

xg <- stats::rgamma(500, shape = 3, rate = 2)
mg <- mle.gamma(xg)
stopifnot(abs(mg$estimate[["shape"]] - 3) < 0.6, abs(mg$estimate[["rate"]] - 2) < 0.5)
stopifnot(is.finite(mg$loglik), all(mg$se > 0))

xw <- stats::rweibull(500, shape = 1.5, scale = 4)
mw <- mle.weibull(c(xw, -1))
stopifnot(abs(mw$estimate[["shape"]] - 1.5) < 0.2, abs(mw$estimate[["scale"]] - 4) < 0.5)

nr <- newton.raphson(quad2, c(10, 10))
stopifnot(nr$converged, all.equal(nr$par, c(8, -10) / 3, tolerance = 1e-6))
nr0 <- newton.raphson(quad2, c(10, 10), max.iter = 1L)
stopifnot(!nr0$converged, nr0$iterations == 1L)
rosen <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
stopifnot(all.equal(newton.raphson(rosen, c(-1.2, 1))$par, c(1, 1), tolerance = 1e-4))
