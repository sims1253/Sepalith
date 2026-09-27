find_ngrams <- function(seq, decoded_ngrams) {
  
  end_pos <- 5L:length(seq)
  start_pos <- end_pos - 4
  
  res <- binarize(do.call(rbind, lapply(1L:length(end_pos), function(ith_mer_id) {
    five_mer <- paste0(seq[start_pos[ith_mer_id]:end_pos[ith_mer_id]], collapse = "")
    stri_count(five_mer, regex = decoded_ngrams)
  })))
  
  res
}