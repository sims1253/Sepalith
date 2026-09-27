qkiener2 <- function(p, m = 0, g = 1, a = 3.2, w = 3.2, 
                     lower.tail = TRUE, log.p = FALSE) {
	if(log.p)       p <- exp(p)
	if(!lower.tail) p <- 1-p
	k <- aw2k(a, w)
	# since v1.9.0
	v <- m + sqrt(3)/pi*g*k*(- exp(-logit(p)/a) + exp(logit(p)/w))/2
	v
}