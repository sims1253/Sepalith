probs_attrue <- function(probs_pred, y) {
  tp <- rep(0, nrow(probs_pred))
  names(tp) <- rownames(probs_pred)
  for (i in seq_len(nrow(probs_pred))) {
    tp[i] <- probs_pred[i, y[i]]
  }

  tp
}