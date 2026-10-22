pars <- data.frame(a = c(1, 1.5, 2), b = c(-1, 0, 1), c = c(0, 0.2, 0.1))

# irt_prob_2pl / irt_prob_3pl
p2 <- irt_prob_2pl(c(-1, 0, 1), pars$a, pars$b)
stopifnot(identical(dim(p2), c(3L, 3L)), isTRUE(all.equal(p2[2, 2], 0.5)))
p3 <- irt_prob_3pl(-50, pars$a, pars$b, pars$c)
stopifnot(isTRUE(all.equal(as.vector(p3), pars$c)))

# item_info_2pl / item_info_3pl
i2 <- item_info_2pl(0, 1, 0)
stopifnot(isTRUE(all.equal(as.vector(i2), 0.25)))
stopifnot(isTRUE(all.equal(item_info_3pl(0.3, 1.2, 0, 0), item_info_2pl(0.3, 1.2, 0))))

# test_information / theta_se
ti <- test_information(c(-1, 0, 1), pars, "2pl")
stopifnot(length(ti) == 3, all(ti > 0))
stopifnot(isTRUE(all.equal(theta_se(0, pars, "3pl"), 1 / sqrt(test_information(0, pars, "3pl")))))

# estimate_theta_eap: monotone in sum score, finite even with steep items
resp <- rbind(c(0, 0, 0), c(1, 0, 0), c(1, 1, 1), c(NA, NA, NA))
eap <- estimate_theta_eap(resp, pars)
stopifnot(nrow(eap) == 4, !is.unsorted(eap$theta[1:3]), abs(eap$theta[4]) < 1e-8)
steep <- data.frame(a = c(40, 40), b = c(-3, 3))
stopifnot(all(is.finite(estimate_theta_eap(rbind(c(0, 1)), steep)$theta)))

# simulate_responses
sim <- simulate_responses(rnorm(50), pars, "3pl", seed = 1)
stopifnot(identical(dim(sim), c(50L, 3L)), all(sim %in% 0:1))
stopifnot(identical(colnames(sim), c("i1", "i2", "i3")))
