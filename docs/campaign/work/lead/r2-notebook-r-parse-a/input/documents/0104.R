remove_hashtags <- function(x) {
    # remove hashtags from text
    x <- gsub("#.+?(\\s|$)", "", x)
    return(x)
}