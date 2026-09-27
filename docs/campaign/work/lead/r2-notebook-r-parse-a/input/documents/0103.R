normalize_text <- function(x) {
    # remove mentions of other users
    x <- gsub("@.+?(\\s|$)", "", x)
    # remove "RT"
    x <- gsub("RT", "", x)
    x <- trimws(tolower(x))
    return(x)
}