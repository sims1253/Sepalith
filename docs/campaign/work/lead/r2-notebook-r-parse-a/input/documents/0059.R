getJABCDEF <- function(Z,Slabels,nl){
    J <- matrix(0,nrow=nl,ncol=length(Z$intercepts))
    hasKglass <- "K-glass" %in% Z$labels
    hasCasalt <- "Ca-salt" %in% Z$labels
    if (hasCasalt){
        i67ca <- getindices(Z,"Ca-salt","Ar36","Ar37")
        i97ca <- getindices(Z,"Ca-salt","Ar39","Ar37")
    }
    if (hasKglass)
        i09k <- getindices(Z,"K-glass","Ar40","Ar39")
    for (i in 1:length(Slabels)){
        j <- (i-1)*6
        label <- Slabels[i]
        i60 <- getindices(Z,label,"Ar36","Ar40")
        i70 <- getindices(Z,label,"Ar37","Ar40")
        i80 <- getindices(Z,label,"Ar38","Ar40")
        i90 <- getindices(Z,label,"Ar39","Ar40")
        i68cl <- getindices(Z,paste("Cl:",label,sep=""),"Ar36","Ar38")
        J[j+1,i60]    <- 1                        # A
        if (hasCasalt) J[j+2,c(i67ca,i70)] <- 1   # B
        J[j+3,c(i68cl,i80)] <- 1                  # C
        if (hasKglass) J[j+4,c(i09k,i90)] <- 1    # D
        J[j+5,i90] <- 1                           # E
        if (hasCasalt) J[j+6,c(i70,i97ca)] <- 1   # F
    }
    return(J)
}