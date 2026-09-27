rank_search_results <- function(data, terms_list, rank_method = "weighted") {
  if (rank_method == "weighted") {
    data <- data %>%
      mutate(Total = rowSums(select(., -Gene))) %>%
      select(Gene, all_of(terms_list), Total) %>%
      arrange(desc(across(all_of(terms_list)))) %>%
      mutate(PubMed_Rank = row_number())
  } else if (rank_method == "total") {
    data <- data %>%
      mutate(Total = rowSums(select(., -Gene))) %>%
      arrange(desc(Total)) %>%
      mutate(PubMed_Rank = row_number())
  } else {
    stop("Invalid rank_method. Choose either 'weighted' or 'total'.")
  }
  return(data)
}