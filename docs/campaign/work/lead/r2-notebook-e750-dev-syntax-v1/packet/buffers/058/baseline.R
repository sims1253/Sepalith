# functions.R

# functions: con_check, con_concur, con_filter, con_view

# Idea: allow con_view to work without a response variable. Maybe by coloring all cells the same color?

# Idea cluster only 1 of the 2 directions

## ---------------------------------------------------------------------------

#' @importFrom grDevices colorRampPalette
RedGrayBlue <- colorRampPalette(c("firebrick", "lightgray", "#375997"))

## ---------------------------------------------------------------------------

#' @title Check connectedness of multiple factors in a dataframe
#'
#' @description
#' Multiple factors in a dataframe are said to be connected if a model matrix
#' based on those factors is full rank.
#'
#' This function provides a formula interface to the lfe::compfactor() function
#' to check for connectedness of the factors.
#'
#' @param data A dataframe
#' 
#' @param formula A formula with multiple factor names in the dataframe,
#' like \code{y ~ f1 + f2 + f3}
#' 
#' @param WW Pass-through argument to `compfactor`
#'
#' @param dropNA If TRUE, observed data that are `NA` will be dropped.
#' 
#' @return A vector with integers representing the group membership of each observation.
#' 
#' @author Kevin Wright
#' 
#' @examples 
#' # In the data_eccleston dataframe, each pair of factors is connected.
#' con_check(data_eccleston, ~ row + trt)
#' con_check(data_eccleston, ~ col + trt)
#' con_check(data_eccleston, ~ row + col)
#' # But all three factors are COMPLETELY disconnected into 16 groups.
#' con_check(data_eccleston, ~ row + col + trt)
#' 
#' @references 
#' None
#' 
#' @export 
con_check <- function(data=NULL, formula=NULL, WW=TRUE, dropNA=TRUE) {
  formula <- deparse(formula)
  .resp <- trimws( strsplit(formula, "~")[[1]][1] ) # left of tilde
  rhs <- strsplit(formula, "~")[[1]][2] # right of tilde
  factor_vars <- trimws( strsplit(rhs, "[+*]")[[1]] )

  # If there is a response, drop rows that have missing values in response
  if(.resp != "" & dropNA) data = data[ !is.na(data[[.resp]]), ]
  
  # Make sure the data are factors
  out <- lfe::compfactor(lapply(data[factor_vars], as.factor), WW=TRUE)
  
  return(out)
}

## ---------------------------------------------------------------------------

#' @title View connectedness of two factors in a dataframe using a levelplot
#'
#' @description
#' If there is replication for the treatment combination cells in a
#' two-way table, the replications are averaged together (or counted)
#' before constructing the heatmap.
#' 
#' By default, rows and columns are clustered using the 'incidence' matrix of 0s and 1s.
#' 
#' The function checks to see if the cells in the heatmap form a connected
#' set.  If not, the data is divided into connected subsets and the subset
#' group number is shown within each cell.
#' 
#' By default, missing values in the response are deleted.
#'
#' Factor levels are shown along the left and bottom sides.
#'
#' The number of cells in each column/row is shown along the top/right sides.
#'
#' If the 2 factors are disconnected, the group membership ID is shown in
#' each cell.
#' 
#' @param data A dataframe
#' 
#' @param formula A formula with two (or more) factor names in the dataframe
#' like \code{yield ~ f1 *f2}
#' 
#' @param fun.aggregate The function to use for aggregating data in cells. Default is mean.
#' 
#' @param xlab Label for x axis
#' 
#' @param ylab Label for y axis
#'
#' @param cex.num Disjoint group number.
#' 
#' @param cex.x Scale factor for x axis tick labels.  Default 0.7.
#' 
#' @param cex.y Scale factor for y axis tick labels  Default 0.7.
#' 
#' @param col.regions Function for color regions. Default RedGrayBlue.
#'
#' @param cluster If "incidence", cluster rows and columns by the
#' incidence matrix. If FALSE, no clustering is performed.
#'
#' @param dropNA If TRUE, observed data that are `NA` will be dropped.
#' 
#' @param ... Other parameters passed to the levelplot() function.
#' 
#' @author Kevin Wright
#'
#' @return A lattice graphics object
#' 
#' @examples
#' require(lattice)
#' bar = transform(lattice::barley, env=factor(paste(site,year)))
#' set.seed(123)
#' bar <- bar[sample(1:nrow(bar), 70, replace=TRUE),]
#' con_view(bar, yield ~ variety * env, cex.x=1, cex.y=.3, cluster=FALSE)
#'
#' # Create a heatmap of cell counts
#' w2b = colorRampPalette(c('wheat','black'))
#' con_view(bar, yield ~ variety * env, fun.aggregate=length,
#'   cex.x=1, cex.y=.3, col.regions=w2b, cluster=FALSE)
#'
#' # Example from paper by Fernando et al. (1983).
#' set.seed(42)
#' data_fernando = transform(data_fernando,
#'   y=stats::rnorm(9, mean=100))
#' con_view(data_fernando, y ~ gen*herd, cluster=FALSE,
#'      main = "Fernando unsorted")
#' con_view(data_fernando, y ~ gen*herd, cluster=TRUE,
#'      main = "Fernando unsorted")
#'
#' # Example from Searle (1971), Linear Models, p. 325
#' dat2 = transform(data_searle,
#'   y=stats::rnorm(nrow(data_searle)) + 100)
#' 
#' con_view(dat2, y ~ f1*f2, cluster=FALSE, main="data_searle unsorted") 
#' con_view(dat2, y ~ f1*f2, main="data_searle clustered")
#'
#' @import lattice
#' @importFrom lfe compfactor
#' @importFrom reshape2 acast
#' @importFrom stats aggregate as.dendrogram as.dist
#' @importFrom stats dist hclust na.omit order.dendrogram
#' @export
#' 
con_view <- function(data, formula,
                     fun.aggregate=mean,
                     xlab="", ylab="",
                     cex.num=0.75,
                     cex.x=0.7, cex.y=0.7,
                     col.regions=RedGrayBlue,
                     cluster="incidence",
                     dropNA=TRUE,
                     ...){

  # Check that the formula has valid names (in the data)
  if(any(c('.resp','.fx','.fy') %in% names(data)))
    stop(".resp, .fx, .fy are reserved names")

  formula <- deparse(formula)
  .resp <- trimws( strsplit(formula, "~")[[1]][1] ) # left of tilde
  rhs <- strsplit(formula, "~")[[1]][2] # right of tilde
  leftstar <- strsplit(rhs, "\\*")[[1]][1]
  rightstar <- strsplit(rhs, "\\*")[[1]][2]
  if(is.na(leftstar) | is.na(rightstar) )
    stop("Incorrect formula")
  

  if( grepl("[:+]", leftstar) & grepl("[:+]", rightstar) ) {
    # Example: f1:f2 / f3:f4
    stop("Only 3 terms can be included in the formula")
  }
  
  if( grepl("[:+]", leftstar) & !grepl("[:+]", rightstar) ) {
    # Ex: f1:f2 / f3
    method="21"
    f1a <- trimws( strsplit(leftstar, ":+")[[1]][1] )
    f1b <- trimws( strsplit(leftstar, ":+")[[1]][2] )
    f2 <- trimws( rightstar )
  }
  if( !grepl("[:+]", leftstar) & grepl("[:+]", rightstar) ) {
    # Ex: f1 / f2:f3
    method="12"
    f1 <- trimws(leftstar)
    f2a <- trimws( strsplit(rightstar, ":+")[[1]][1] )
    f2b <- trimws( strsplit(rightstar, ":+")[[1]][2] )
  }
  if( !grepl("[:+]", leftstar) & !grepl("[:+]", rightstar) ) {
    # Ex: f1 / f2
    method="11"
    f1 <- trimws(leftstar)
    f2 <- trimws(rightstar)
  }

  # Drop rows that have missing values in response
  if(dropNA) data = data[ !is.na(data[[.resp]]), ]

  # Merge f1a & f1b or else f2a & f2b into a single item
  if(method=="11") {
    .fx <- f1
    .fy <- f2
  }
  if(method=="12") {
    .fx <- f1
    .fy <- paste0(f2a, "_", f2b)
    data[[.fy]] <- paste0(data[[f2a]], "_", data[[f2b]])
  }
  if(method=="21") {
    .fx <- paste0(f1a, "_", f1b)
    .fy <- f2
    data[[.fx]] <- paste0(data[[f1a]], "_", data[[f1b]])
  }

  
  # Guarantee fx and fy to be factors so that 'levels' will work later.
  data[[.fx]] <- factor(data[[.fx]])
  data[[.fy]] <- factor(data[[.fy]])

  # flag for fun.aggregate=length
  agglen <- FALSE
  if( "fun.aggregate" %in% names(as.list(sys.call())) &&
        as.character(as.list(sys.call())$fun.aggregate)=="length" )
    agglen <- TRUE
  
  # Aggregate reps together
  # Syntax found at http://stackoverflow.com/questions/27235088
  if(agglen) {
    # 'length' does not accept na.rm argument so use an anonymous function
    data <- aggregate(data[.resp], by=data[c(.fx,.fy)],
                      FUN=function(x) length(na.omit(x)))
  } else {
    # aggregate by mean
    data <- aggregate(data[.resp], by=data[c(.fx,.fy)], mean, na.rm=TRUE)
  }

  #browser()
  # Identify connected sets
  data$.grp <- lfe::compfactor(data[c(.fx,.fy)])
  #data$.grp <- lfe::compfactor(list(data[[.fx]], data[[.fy]]))
  data$.grp <- as.numeric(data$.grp)

  # Clustering needs a matrix. We have already aggregated.
  datm <- reshape2::acast(data, paste(.fx,"~",.fy), value.var=.resp)
  
  # If we aggregate by length (number of reps), then change
  # 0 to NA so that row/column counts of cells will be right
  if(agglen) {
    datm[datm==0L] <- NA
    datm[is.nan(datm)] <- NA
  }

  if(cluster=="incidence"){
    # use row and column clustering of the _incidence_ matrix
    tab <- !is.na(datm)
    class(tab) <- 'matrix'
    if(nrow(tab) > 2){
      hcr <- hclust(dist(tab))
      ddr <- as.dendrogram(hcr)
      ixr <- order.dendrogram(ddr)
      # need to re-order datm for cell counts
      datm <- datm[ixr, ]
      data[[.fx]] <- factor(data[[.fx]], levels=levels(data[[.fx]])[ixr])
    }
    if(ncol(tab)>2){
      hcc <- hclust(dist(t(tab)))
      ddc <- as.dendrogram(hcc)
      ixc <- order.dendrogram(ddc)
      datm <- datm[ , ixc]
      data[[.fy]] <- factor(data[[.fy]], levels=levels(data[[.fy]])[ixc])
    }
    # Convert to 0/1
    tab <- 1*tab
    n.common.env <- crossprod(tab)
    n.common.gen <- tcrossprod(tab)
  }

  # Change axis text size
  nc <- ncol(datm)
  nr <- nrow(datm)

  # Note, the 'par' function opens a graphics window, which causes
  # knitr to create an empty pdf file
  pctMiss <- sum(is.na(datm))/(nc * nr)
  pctMiss <- round(100*pctMiss, 1)
  subtitle <- paste("(", pctMiss, "% missing)", sep="")

  tab <- !is.na(datm)
  cells.per.row <- apply(tab, 1, sum)
  cells.per.col <- apply(tab, 2, sum)

  myxscale <- function(...) {
    ans <- xscale.components.default(...)
    ans$top <- ans$bottom
    ans$top$labels$labels <- as.character(cells.per.row)
    ans
  }
  myyscale <- function(...) {
    ans <- yscale.components.default(...)
    ans$right <- ans$left
    ans$right$labels$labels <- as.character(cells.per.col)
    ans
  }

  # if there is only 1 group, don't show it
  if( all(data$.grp<2) ) {
    data$.grp <-""
  }

  # if there are 2+ groups, warn the user
  max.grp.num = max(data$.grp)
  if (max.grp.num > 1)
    warning("There are ", max.grp.num, " groups")

  formula2 <- eval(parse(text=paste0(.resp, "~", .fx, "*", .fy)))
  # We need "data" instead of "datm" for the group information
  levelplot(formula2, data=data,
            aspect="fill",
            xlab=xlab, ylab=ylab,
            col.regions=col.regions,
            # Note, relation=free is required for top/right axes!
            scales=list(
              x=list(cex=cex.x, relation="free", rot=90),
              y=list(cex=cex.y, relation="free", rot=0)),
            xscale.components = myxscale,
            yscale.components = myyscale,
            # more space between panel/legend
            par.settings=list(layout.heights=list(key.axis.padding=4),
                              layout.widths=list(axis.key.padding=5)),
            panel=function(x,y,z,...){
              panel.levelplot(x,y,z,...)
              panel.text(x,y,data$.grp, cex=cex.num)
            } ,
            ...)

}

## ---------------------------------------------------------------------------


#' @title Filter a dataframe using two-way criteria to increase connectedness
#'
#' @description
#' Traditional filtering (subsetting) of data is typically performed via
#' some criteria based on the *columns* of the data.
#'
#' In contrast, this function performs filtering of data based on the
#' *joint* rows and columns of a matrix-view of two factors.
#' 
#' Conceptually, the idea is to re-shape two or three columns of a dataframe
#' into a matrix, and then delete entire rows (or columns) of the matrix if
#' there are too many missing cells in a row (or column).
#'
#' The two most useful applications of two-way filtering are to:
#' 
#' 1. Remove a factor level that has few interactions with another factor.
#' This is especially useful in linear models to remove rare factor
#' combinations. 
#' 
#' 2. Remove a factor level that has any missing interactions with another
#' factor. This is especially useful with biplots of a matrix to remove
#' rows or columns that have missing values.
#' 
#' A formula syntax is used to specify the two-way filtering criteria.
#' 
#' Some examples may provide the easiest understanding.
#' 
#' dat <- data.frame(state=c("NE","NE", "IA", "NE", "IA"),
#'                   year=c(1,2,2,3,3), value=11:15)
#' 
#' When the 'value' column is re-shaped into a matrix it looks like:
#' 
#' state/year |  1 |  2 |  3 |
#'         NE | 11 | 12 | 14 |
#'         IA |    | 13 | 15 |
#'
#' Drop states with too much missing combinations.
#' Keep only states with "at least 3 years per state"
#' con_filter(dat, ~ 3 * year / state)
#'  NE    1    11
#'  NE    2    12
#'  NE    3    14
#'
#' Keep only years with "at least 2 states per year"
#' con_filter(dat, ~ 2 * state / year)
#'  NE    2    12
#'  IA    2    13
#'  NE    3    14
#'  IA    3    15
#'
#' If the constant number in the formula is less than 1.0, this is
#' interpreted as a *fraction*.
#' Keep only states with "at least 75% of years per state"
#' con_filter(dat, ~ .75 * year / state)
#'
#' It is possible to include another factor on either side of the slash "/".
#' Suppose the data had another factor for political party called "party".
#' Keep only states with "at least 2 combinations of party:year per state"
#' con_filter(dat, ~ 2 * party:year / state)
#' 
#' If the formula contains a response variable, missing values are dropped
#' first, then the two-way filtering is based on the factor combinations.
#' con_filter(dat, value ~ 2 * state / year)
#' 
#' @param data A dataframe
#' 
#' @param formula A formula with two factor names in the dataframe
#' that specifies the criteria for filtering,
#' like \code{y ~ 2 * f1 / f2}
#' 
#' @param verbose If TRUE, print some diagnostic information about what data
#' is being deleted. (Similar to the 'tidylog' package).
#' 
#' @param returndropped If TRUE, return the dropped rows instead of the
#' kept rows. Default is FALSE.
#' 
#' @return
#' The original dataframe is returned, minus rows that are filtered out.
#' 
#' @author Kevin Wright
#' 
#' @examples
#' dat <- data.frame(
#'   gen = c("G3", "G4", "G1", "G2", "G3", "G4", "G5",
#'           "G1", "G2", "G3", "G4", "G5",
#'           "G1", "G2", "G3", "G4", "G5",
#'           "G1", "G2", "G3", "G4", "G5"),
#'   env = c("E1", "E1", "E1", "E1", "E1", "E1", "E1",
#'           "E2", "E2", "E2", "E2", "E2",
#'           "E3", "E3", "E3", "E3", "E3",
#'           "E4", "E4", "E4", "E4", "E4"),
#'   yield = c(65, 50, NA, NA, 65, 50, 60,
#'             NA, 71, 76, 80, 82,
#'             90, 93, 95, 102, 97,
#'             98, 102, 105, 130, 135))
#' 
#' # How many observations are there for each combination of gen*env?
#' with( subset(dat, !is.na(yield)) , table(gen,env) )
#' 
#' # Note, if there is no response variable, the two-way filtering is based
#' # only on the presence of the factor combinations.
#' dat1 <- con_filter(dat, ~ 4*env / gen)
#' 
#' # If there is a response variable, missing values are dropped first,
#' # then the two-way filtering is based on the factor combinations.
#' 
#' dat1 <- con_filter(dat, yield ~ 4*env/gen)
#' dat1 <- con_filter(dat, yield ~ 5*env/ gen)
#' dat1 <- con_filter(dat, yield ~ 6*gen/ env)
#' dat1 <- con_filter(dat, yield ~ .8 *env / gen)
#' dat1 <- con_filter(dat, yield ~ .8* gen / env)
#' dat1 <- con_filter(dat, yield ~ 7 * env / gen)
#'
#' @references 
#' None.
#' 
#' @export 
con_filter <- function(data, formula, verbose=TRUE, returndropped=FALSE) {
  formula <- deparse(formula)

  .resp <- trimws( strsplit(formula, "~")[[1]][1] ) # response left of tilde
  if(.resp != "" &
       !(.resp %in% names(data))) stop(.resp, " not found in data")
  
  rhs <- strsplit(formula, "~")[[1]][2] # right of tilde
  leftslash <- strsplit(rhs, "/")[[1]][1]
  rightslash <- strsplit(rhs, "/")[[1]][2]

  # peel off the threshold
  thresh <- as.numeric(strsplit(leftslash, "\\*")[[1]][1])
  leftslash <- trimws(strsplit(leftslash, "\\*")[[1]][2])
  
  if( grepl("[:+]", leftslash) & grepl("[:+]", rightslash) ) {
    # Example: f1:f2 / f3:f4
    stop("Only 3 terms can be included in the formula")
  }
  if( grepl("[:+]", leftslash) & !grepl("[:+]", rightslash) ) {
    # Ex: f1:f2 / f3
    method="21"
    f1a <- trimws( strsplit(leftslash, ":+")[[1]][1] )
    f1b <- trimws( strsplit(leftslash, ":+")[[1]][2] )
    f2 <- trimws( rightslash )
  }
  if( !grepl("[:+]", leftslash) & grepl("[:+]", rightslash) ) {
    # Ex: f1 / f2:f3
    method="12"
    f1 <- trimws(leftslash)
    f2a <- trimws( strsplit(rightslash, ":+")[[1]][1] )
    f2b <- trimws( strsplit(rightslash, ":+")[[1]][2] )
  }
  if( !grepl("[:+]", leftslash) & !grepl("[:+]", rightslash) ) {
    # Ex: f1 / f2
    method="11"
    f1 <- trimws(leftslash)
    f2 <- trimws(rightslash)
  }

  # If we have a response variable, omit rows with missing values
  if(.resp != ""){
    dropix <- is.na(data[[.resp]])
    data <- data[!dropix,]
  }

  n0 <- nrow(data)

  # Merge f1a & f1b or else f2a & f2b into a single item
  if(method=="11") {
    f1vec <- data[[f1]]
    f2vec <- data[[f2]]
  }
  if(method=="12") {
    f1vec <- data[[f1]]
    f2vec <- paste0(data[[f2a]], data[[f2b]])
    f2 <- paste0(f2a,":",f2b)
  }
  if(method=="21") {
    f1vec <- paste0(data[[f1a]], data[[f1b]])
    f2vec <- data[[f2]]
    f1 <- paste0(f1a,":",f1b)
  }

  # Unique combinations of f1 f2 (in case there are multiple reps per combination)
  uni <- unique(data.frame(f1=f1vec, f2=f2vec))
  # table() output includes unused factor levels, so get rid of those
  uni <- droplevels(uni)
  f1.nlev <- length(unique( uni$f1 ))
  f2.nlev <- length(unique( uni$f2 ))
  
  # Delete levels of one factor
  f1.per.f2 <- table( uni$f2 )
  if(thresh < 1)
    f2.keep.levs <- names(f1.per.f2)[f1.per.f2/f1.nlev >= thresh]
  else
    f2.keep.levs <- names(f1.per.f2)[f1.per.f2 >= thresh]
  
  f2.drop.levs <- setdiff(names(f1.per.f2), f2.keep.levs)
  
  # Delete rows with the f2 drop levels
  if(length(f2.drop.levs)>0) {
    if(verbose){
      message("Dropping these ", length(f2.drop.levs),
          " of ", f2.nlev,
          " levels of ",f2,":")
      print(f2.drop.levs)
    }
    keepix <- f2vec %in% f2.keep.levs
    f1vec <- f1vec[keepix]
    f2vec <- f2vec[keepix]
    if(returndropped)
      data <- data[ !keepix, ]
    else
      data <- data[ keepix, ]
    data <- droplevels(data)
    uni <- droplevels(uni[uni$f2 %in% f2.keep.levs,])
  }

  n2 <- nrow(data)
  if(verbose)
    cat("Deleted", n0-n2, "of", n0, "rows of data.\n")
    
  if(verbose){
    tab1 <- table( uni$f1 )
    if(any(tab1 < 2))
      warning("Some ",f1," have only 1 ", f2, ".")
    tab2 <- table( uni$f2 )
    if(any(tab2 < 2))
      warning("Some ",f2," have only 1 ", f1,".")
  }  

  if(nrow(data)==0L) warning("No data remains.")
  data <- droplevels(data)
  return(data)

}

## ---------------------------------------------------------------------------

#' @title View concurrence of two factors in a dataframe using a matrix plot.
#'
#' @description Draws a concurrence plot of 2 factors in a dataframe.
#' For example, in a multi-environment yield trial (testing multiple crop
#' varieties in multple environments) it is interesting to examine the
#' balance of the testing pattern.
#' For each pair of environments, how many genotypes are tested
#' in both environments? The concurrence plot shows the amount of
#' connectedness (number of varieties) of the environments with each other.
#'
#' By default, missing values in the response are deleted.
#' 
#' Replicated combinations of the two factors are ignored.
#' (This could be changed if someone has a need.)
#'
#' @param data A dataframe
#' 
#' @param formula A formula with multiple factor names in the dataframe,
#' like \code{y ~ f1 / f2}.
#' 
#' @param dropNA If TRUE, observed data that are `NA` will be dropped.
#' 
#' @param xlab Label for x axis
#' 
#' @param ylab Label for y axis
#' 
#' @param cex.x Scale factor for x axis tick labels.  Default 0.7.
#' 
#' @param cex.y Scale factor for y axis tick labels  Default 0.7.
#' 
#' @param ... Other parameters passed to the levelplot() function.
#' 
#' @return A lattice graphics object
#' 
#' @author Kevin Wright
#' 
#' @examples 
#' require(lattice)
#' bar = transform(lattice::barley, env=factor(paste(site,year)))
#' set.seed(123)
#' bar <- bar[sample(1:nrow(bar), 70, replace=TRUE),]
#' con_concur(bar, yield ~ variety / env, cex.x=0.75, cex.y=.3)
#'
#' @references 
#' None
#' @export 
con_concur <- function(data, formula,
                       dropNA=TRUE,
                       xlab="", ylab="", cex.x=.7, cex.y = 0.7, ...){

  # Check that the formula has valid names (in the data)
  if(any(c('.resp','.fx','.fy') %in% names(data)))
    stop(".resp, .fx, .fy are reserved names")

  formula <- deparse(formula)
  .resp <- trimws( strsplit(formula, "~")[[1]][1] ) # left of tilde
  rhs <- strsplit(formula, "~")[[1]][2] # right of tilde
  leftslash <- strsplit(rhs, "\\/")[[1]][1]
  rightslash <- strsplit(rhs, "\\/")[[1]][2]
  if(is.na(leftslash) | is.na(rightslash) )
    stop("Incorrect formula")

  .fx <- trimws(leftslash)
  .fy <- trimws(rightslash)

  # If there is a response, drop rows that have missing values in response
  if(.resp != "" & dropNA) data = data[ !is.na(data[[.resp]]), ]

  if(missing(xlab)) xlab <- paste0("Number of concurrent ", .fx, " per ", .fy)
  # Need unique() in case there are multiple observations
  dat2 <- unique(data[c(.fx,.fy)])
  # Add indicator, convert to incidence matrix
  dat2 <- cbind(dat2, ind=1)
  dat2 <- acast(dat2, paste(.fx,"~",.fy), value.var="ind", fill=0)
  dat2 <- crossprod(dat2)
  levelplot(dat2, scales=list(x=list(cex=cex.x, rot=90), y=list(cex=cex.y)),
            xlab=xlab, ylab=ylab,
            ...)
}

## ---------------------------------------------------------------------------

if(FALSE){
  # Explore the syntax tree of the formula.
  # Note: I decided to parse the formula by myself.
  ff <- formula(y~ 2 * gen/site:yr)
  deparse(ff[[1]]) # ~
  deparse(ff[[2]]) # y
  deparse(ff[[3]]) # 2*gen/site:yr
  deparse(ff[[3]][[1]]) # /
  deparse(ff[[3]][[2]]) # 2*gen
  deparse(ff[[3]][[3]]) # site:yr
  deparse(ff[[3]][[3]][[1]]) # :
  deparse(ff[[3]][[3]][[2]]) # site
  deparse(ff[[3]][[3]][[3]]) # yr
  
  all.vars(ff)
  lobstr::ast(y~ 2 * gen/site:yr)
}

if(FALSE){
  libs(janitor)
  test1 <- matrix( c("G1", "IA", "2020", # gen has 1 state, 1 yr,
                     "G2", "IA", "2020", # gen has 1 state, 2 yr
                     "G2", "IA", "2021",
                     "G3", "NE", "2020", # 2 states, 1 yr
                     "G3", "IA", "2020",
                     "G4", "KS", "2020", # state has 1 gen, 1 yr
                     "G5", "MO", "2020", # state has 1 gen, 2yr
                     "G5", "MO", "2021",
                     "G6", "IL", "2020", # state has 2 gen, 1yr
                     "G7", "IL", "2020",
                     "G8", "AR", "2019", # year has 1 gen 1 state
                     "G9", "IN", "2018", # year has 1 gen, 2 state
                     "G9", "OH", "2018",
                     "G10", "MN", "2017", # year has 2 gen, 1 state
                     "G11", "MN", "2017",
                     "G12", "MD", "2010", # gen has 2 state, 2 yr, 2 reps
                     "G12", "MD", "2010",
                     "G12", "GA", "2011",
                     "G12", "GA", "2011"), byrow=TRUE, ncol=3)
  test1 <- as.data.frame(test1)
  colnames(test1) <- c("gen","state","year")
  
  set.seed(42)
  test1$y <- round( runif(nrow(test1)), 2)
  head(test1)

  con_filter(test1, y~ 2 * f1:f2 / f3:f4) # Errs, as it should

  #con_filter(test1, y ~ 2 * gen / state) |> with(data=_, table(gen,state))
  con_filter(test1, y ~ 2 * gen / state) |> tabyl(gen,state)
  con_filter(test1, y ~ 2 * gen / state, returndropped=TRUE)
  
  #con_filter(test1, y ~ 2 * gen / year) |> with(data=_, table(gen,year))
  con_filter(test1, y ~ 2 * gen / year) |> tabyl(gen,year)

  con_view(test1, y~ gen * state:year)
  con_filter(test1, y ~ 2 * gen / state:year) |> transform(stateyr=paste(state,year)) |> with(table(gen,stateyr))
  con_filter(test1, y ~ 2 * gen / state:year) |> arrange(state,year)
  con_filter(test1, y ~ 2 * gen / state:year, returndropped=TRUE)

  con_view(test1, y~ state:year * gen)
  con_filter(test1, y ~ 2 * state:year / gen) %>% arrange(gen)
  con_filter(test1, y ~ 2 * state:year / gen, returndropped=TRUE)

  library(agridat)
  dat <- lin.unbalanced
  head(dat)
  con_view(dat, yield ~ gen*loc)

  # Maybe this could be a function con_table
  with(dat2, table(gen,yl)) %>% apply(2, \(x) sum(x>0)) %>% sort()
  with(dat2, table(year,gl)) %>% apply(2, \(x) sum(x>0)) %>% sort() # Problem here!
  with(dat2, table(gen,yl)) %>% apply(2, \(x) sum(x>0)) %>% sort()


}

