format1 <- function(x){
    ns <- length(x$x)/4
    out <- matrix('',ns+3,6)
    out[1,1:2] <- c("J","err[J]")
    out[2,1:2] <- x$J
    out[3,] <- c("Ar39Ar40","errAr39Ar40",
                 "Ar36Ar40","errAr36Ar40",
                 "Ar39Ar36","errAr39Ar36")
    i90 <- findmatches(labels=names(x$x),prefixes=c("Ar39Ar40"))
    i60 <- findmatches(labels=names(x$x),prefixes=c("Ar36Ar40"))
    i96 <- findmatches(labels=names(x$x),prefixes=c("Ar39Ar36"))
    for (i in 1:ns){
        out[i+3,1] <- x$x[i90[i]]
        out[i+3,2] <- sqrt(x$covmat[i90[i],i90[i]])
        out[i+3,3] <- x$x[i60[i]]
        out[i+3,4] <- sqrt(x$covmat[i60[i],i60[i]])
        out[i+3,5] <- x$x[i96[i]]
        out[i+3,6] <- sqrt(x$covmat[i96[i],i96[i]])
    }
    out
}