for (m in c("euler", "midpoint", "heun", "ralston", "rk4")) {
  tab <- ode.tableau(m)
  stopifnot(abs(sum(tab$b) - 1) < 1e-12, all(abs(rowSums(tab$a) - tab$c) < 1e-12))
  stopifnot(identical(names(tab), c("a", "b", "c", "order")))
}
stopifnot(ode.tableau("rk4")$order == 4L)
stopifnot(ode.tableau()$order == 1L)
stopifnot(ode.tableau("ralston")$c[2] == 2 / 3)

decay <- function(t, y) -y
stopifnot(ode.step(decay, 0, 1, 0.1, "euler") == 0.9)
stopifnot(abs(ode.step(decay, 0, 1, 0.1, "midpoint") - 0.905) < 1e-12)
stopifnot(abs(ode.step(decay, 0, 1, 0.1, "heun") - 0.905) < 1e-12)
stopifnot(abs(ode.step(decay, 0, 1, 0.1, "ralston") - 0.905) < 1e-12)
stopifnot(abs(ode.step(decay, 0, 1, 0.1, "rk4") - exp(-0.1)) < 1e-7)
stopifnot(abs(ode.step(function(t, y) t, 0, 0, 1, "midpoint") - 0.5) < 1e-12)

stopifnot(ode.stability(-1, "euler") == 0, ode.stability(-2, "heun") == 1)
stopifnot(Mod(ode.stability(-1, "midpoint") - 0.5) < 1e-12)
stopifnot(Mod(ode.stability(-1, "ralston") - 0.5) < 1e-12)
stopifnot(Mod(ode.stability(-0.1) - ode.step(decay, 0, 1, 0.1, "rk4")) < 1e-15)
stopifnot(abs(Mod(ode.stability(2i, "euler")) - sqrt(5)) < 1e-12)
stopifnot(length(ode.stability(c(-1, -2))) == 2L)

osc <- function(t, y) c(y[2], -y[1])
sol <- ode.solve(osc, c(pos = 1, vel = 0), seq(0, pi, length.out = 5), substeps = 20L)
stopifnot(identical(colnames(sol), c("time", "pos", "vel")), nrow(sol) == 5L)
stopifnot(abs(sol[5, "pos"] + 1) < 1e-6, abs(sol[5, "time"] - pi) < 1e-12)
stopifnot(identical(colnames(ode.solve(decay, 1, c(0, 1))), c("time", "y1")))
stopifnot(inherits(try(ode.solve(decay, 1, c(1, 0)), silent = TRUE), "try-error"))
stopifnot(inherits(try(ode.solve(decay, NA_real_, c(0, 1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(ode.solve(decay, 1, c(0, 1), substeps = 0), silent = TRUE), "try-error"))

ad <- ode.adaptive(decay, 1, c(0, 2), tol = 1e-6)
stopifnot(abs(tail(ad$t, 1) - 2) < 1e-12, abs(ad$y[nrow(ad$y), 1] - exp(-2)) < 1e-3)
stopifnot(nrow(ad$y) == ad$accepted + 1L, ad$accepted > 10L)
ad2 <- ode.adaptive(decay, 1, c(0, 2), h0 = 1, tol = 1e-8)
stopifnot(ad2$rejected >= 1L, ad2$accepted > ad$accepted)

oo <- ode.order(decay, 1, 1, exp(-1), "rk4")
stopifnot(identical(names(oo), c("steps", "h", "error", "order")), is.na(oo$order[1]))
stopifnot(all(abs(oo$order[-1] - 4) < 0.2))
oo <- ode.order(decay, 1, 1, exp(-1), "euler", steps = c(100L, 200L))
stopifnot(abs(oo$order[2] - 1) < 0.05, oo$error[2] < oo$error[1])
