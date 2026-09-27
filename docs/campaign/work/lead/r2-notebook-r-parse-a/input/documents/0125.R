search_pubmed <- function(
  genes_list,
  terms_list,
  rank_method = "weighted",
  verbose = TRUE
) {
  single_search_results <- data.frame(
    Gene = character(),
    Term = character(),
    Count = integer()
  )

  for (gene in genes_list) {
    for (term in terms_list) {
      if (verbose) {
        message(paste("Searching PubMed for gene:", gene, "and term:", term))
      }
      count <- single_pubmed_search(gene, term)
      single_search_results <- rbind(
        single_search_results,
        data.frame(Gene = gene, Term = term, Count = count)
      )
    }
  }

  aggregated_results <- single_search_results %>%
    group_by(Gene, Term) %>%
    summarise(Count = sum(Count), .groups = 'drop')

  pubmed_search_results <- aggregated_results %>%
    pivot_wider(
      names_from = Term,
      values_from = Count,
      values_fill = list(Count = 0)
    )

  pubmed_search_results <- rank_search_results(
    pubmed_search_results,
    terms_list,
    rank_method
  )

  return(pubmed_search_results)
}