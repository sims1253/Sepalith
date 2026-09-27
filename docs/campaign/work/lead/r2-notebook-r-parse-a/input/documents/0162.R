rkiener2 <- function(n, m = 0, g = 1, a = 3.2, w = 3.2) {
	p <- runif(n)
	v <- qkiener2(p, m, g, a, w)
	v
}