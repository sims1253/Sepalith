lkiener2 <- function(x, m = 0, g = 1, a = 3.2, w = 3.2) {
	funnlslm2 <- function(x, m, g, a, w, lpi) { 
		fn  <- function(lp) x - qlkiener2(lp, m, g, a, w)
		opt <- minpack.lm::nls.lm(par=lpi, fn=fn)
		opt$par
	}
    fuv <- function(u, v) which.min(abs(u-v))[1]
	lk  <- lkiener1(range(x), m, g, k=aw2k(a, w))
	lp2 <- lk*exp(-lk*aw2d(a, w)*g^0.5)
	lr2 <- funnlslm2(range(x), m, g, a, w, range(lp2))
	l5  <- seq(range(lr2)[1], range(lr2)[2],
	           length.out=max(50001, length(x)*51))
	q5  <- qlkiener2(l5, m, g, a, w)
	id5 <- sapply(x, fuv, q5)
	lpi <- l5[id5]
	lpi
}