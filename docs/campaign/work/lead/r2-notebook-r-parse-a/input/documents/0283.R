stepwise_replacement <- function(func, pars1, pars2, symmetrical = TRUE, direction = "up",...){
	direction <- tolower(direction)
	stopifnot(direction %in% c("up","down","both"))
	
	up                   <- direction %in% c("up","both")
	down                 <- direction %in% c("down","both")
	N                    <- length(pars1)
	
	pars1Mat            <- matrix(pars1, ncol = N + 1, nrow = N)
	pars2Mat            <- matrix(pars2, ncol = N + 1, nrow = N)
	
	RM_1_2_up            <- matrix(ncol = N + 1, nrow = N)
	RM_1_2_down          <- RM_1_2_up
	RM_2_1_up            <- RM_1_2_up
	RM_2_1_down          <- RM_1_2_up
	
	# based on 1-> 2 upward
	r1ind                     <- lower.tri(pars1Mat, TRUE)
	r2ind                     <- upper.tri(pars1Mat)
	
	RM_1_2_up[r1ind]          <- pars1Mat[r1ind]
	RM_1_2_up[r2ind]          <- pars2Mat[r2ind]
	
	RM_1_2_down[r1ind[N:1, ]] <- pars1Mat[r1ind[N:1, ]]
	RM_1_2_down[r2ind[N:1, ]] <- pars2Mat[r2ind[N:1, ]]
	
	RM_2_1_up[r1ind]          <- pars2Mat[r1ind]
	RM_2_1_up[r2ind]          <- pars1Mat[r2ind]
	
	RM_2_1_down[r1ind[N:1, ]] <- pars2Mat[r1ind[N:1, ]]
	RM_2_1_down[r2ind[N:1, ]] <- pars1Mat[r2ind[N:1, ]]
	
	dec                       <- matrix(NA, nrow = N, ncol = 4)
	if (up){
		dec[, 1]              <- diff(apply(RM_1_2_up, 2, func, ...))
	}
	if (down){
		dec[, 2]              <- rev(diff(apply(RM_1_2_down, 2, func, ...)))
	}
	
	if (symmetrical){
		if (up){
			dec[, 3]          <- -diff(apply(RM_2_1_up, 2, func, ...))
		}
		if (down){
			dec[, 4]          <- rev(-diff(apply(RM_2_1_down, 2, func, ...)))
		}
	}
	
	
	dec_avg               <- rowMeans(dec, na.rm = TRUE)
	names(dec_avg)        <- names(pars1)
	dec_avg
}