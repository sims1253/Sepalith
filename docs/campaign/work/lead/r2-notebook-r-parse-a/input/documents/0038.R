Lon2Index <- function(lon, lonmin, lonmax) {
    if (is.null(lon)) {
        stop("Parameter 'lon' cannot be NULL.")
    }
    if (!is.numeric(lon)) {
        stop("Parameter 'lon' must be numeric.")
    }
    if (!is.vector(lon)) {
        stop("Parameter 'lon' must be a vector.")
    }
    if (!is.numeric(lonmin) | !is.numeric(lonmax)) {
        stop("Parameter 'lonmin' and 'lonmax' must be numeric.")
    }
    if (!is.vector(lonmin) | !is.vector(lonmax)) {
        stop("Parameter 'lonmin'and 'lonmax' must be a vector.")
    }

    vlonmax <- max(lon)
    vlonmin <- min(lon)
    if (vlonmin < 0 & !(vlonmax > 180)) { # -180 to 180
        if (lonmin < -180) {
            stop("Change parameter 'lonmin' to match longitudes ",
                 "in the range -180 to 180.")
        } else if (lonmin > 180) {
            lonmin <- lonmin - 360
        } 
        if (lonmax < -180) {
            stop("Change parameter 'lonmax' to match longitudes ",
                 "in the range -180 to 180.")
        } else if (lonmax > 180) {
            lonmax <- lonmax - 360
        }
        if (lonmin > lonmax) {
            index <- c(which(lon >= lonmin), which(lon <= lonmax))  
        } else {
            index <- which(lon >= lonmin & lon <= lonmax)
        }
    } else if (vlonmin < 0 & vlonmax > 180) { # -360 to 360
        if (lonmin > lonmax) {
            index <- c(which(lon >= lonmin), which(lon <= lonmax))
        } else {
            index <- which(lon >= lonmin & lon <= lonmax)
        }
    } else { # 0 : 360
        if (lonmin < 0) {
            lonmin <- lonmin + 360
        } else if (lonmin > 360) {
            lonmin <- lonmin - 360
        } 
        if (lonmax < 0) {
            lonmax <- lonmax + 360
        } else if (lonmax > 360) {
            lonmax <- lonmax - 360
        }
        if (lonmin > lonmax) {
            index <- c(which(lon >= lonmin), which(lon <= lonmax))
        } else {
            index <- which(lon >= lonmin & lon <= lonmax)
        }
    }

return(index)
}