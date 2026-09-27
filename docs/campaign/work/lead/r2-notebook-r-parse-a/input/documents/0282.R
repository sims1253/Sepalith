ttest_data <- function(x, cases_column_1, cases_column_n, controls_column_1, controls_column_n, ttest_cutoff) {
  x_ttest <- apply(x, 1, function(x) {t.test(x[cases_column_1:cases_column_n], x[controls_column_1:controls_column_n], "two.sided", var.equal = FALSE)$p.value})
  my_ttest_sorted <- sort(x_ttest, decreasing = FALSE)
  my_ttest_sorted_dtfm <- as.data.frame(my_ttest_sorted)
  my_ttest_candidate <- subset(my_ttest_sorted_dtfm, my_ttest_sorted_dtfm$my_ttest_sorted <= ttest_cutoff)
  return(my_ttest_candidate)
}