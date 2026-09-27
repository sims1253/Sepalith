update_otu_table <- function(physeq_obj, rarefied_otus, iteration = NULL) {
  suppressMessages(.phyloseq_class_check(physeq_obj))

  physeq_samples <- sample_names(physeq_obj)

  # Handle list input
  if (is.list(rarefied_otus) && !is.data.frame(rarefied_otus)) {
    if (is.null(iteration)) {
      cli::cli_abort(
        "{.arg iteration} must be specified when {.arg rarefied_otus} is a list."
      )
    }

    n_iter <- length(rarefied_otus)

    if (iteration < 1 || iteration > n_iter) {
      cli::cli_abort(
        "{.arg iteration} must be between 1 and {n_iter}."
      )
    }

    cli::cli_alert_info(
      "Extracting iteration {.val {iteration}} from list of {.val {n_iter}} iterations."
    )

    rarefied_otus <- as.data.frame(rarefied_otus[[iteration]])
  }

  # Handle 3D array input
  if (is.array(rarefied_otus) && length(dim(rarefied_otus)) == 3) {
    if (is.null(iteration)) {
      cli::cli_abort(
        "{.arg iteration} must be specified when {.arg rarefied_otus} is a 3D array."
      )
    }

    n_iter <- dim(rarefied_otus)[3]

    if (iteration < 1 || iteration > n_iter) {
      cli::cli_abort(
        "{.arg iteration} must be between 1 and {n_iter}."
      )
    }

    cli::cli_alert_info(
      "Extracting iteration {.val {iteration}} from array with {.val {n_iter}} iterations."
    )

    rarefied_otus <- as.data.frame(rarefied_otus[,, iteration])
  }

  # Now rarefied_otus is always a data frame
  otu_rare_samples <- rownames(rarefied_otus)
  shared_samples <- intersect(physeq_samples, otu_rare_samples)
  removed_samples <- setdiff(physeq_samples, otu_rare_samples)

  .report_sample_status(
    shared_samples,
    removed_samples,
    rarefied_otus,
    physeq_samples
  )

  otu_rare_ord <- rarefied_otus[shared_samples, , drop = FALSE]

  cli::cli_alert_info(
    "Building phyloseq object with {.val {nrow(otu_rare_ord)}} samples and {.val {ncol(otu_rare_ord)}} taxa"
  )

  # Build components
  phyloseq_components <- list(
    otu_table(
      t(otu_rare_ord) |> as.matrix() |> as.data.frame(),
      taxa_are_rows = TRUE
    ),
    sample_data(physeq_obj)[shared_samples, ]
  )

  # Add optional components if they exist
  .add_optional_component <- function(components, accessor) {
    obj <- accessor(physeq_obj, errorIfNULL = FALSE)
    if (!is.null(obj)) c(components, list(obj)) else components
  }

  phyloseq_components <- .add_optional_component(phyloseq_components, tax_table)
  phyloseq_components <- .add_optional_component(phyloseq_components, phy_tree)
  phyloseq_components <- .add_optional_component(phyloseq_components, refseq)

  # Build the new phyloseq object
  new_phyloseq <- do.call(phyloseq, phyloseq_components) %>%
    prune_taxa(taxa_sums(x = .) > 0, x = .) %>%
    prune_samples(sample_sums(x = .) > 0, x = .)

  cli::cli_alert_success("Update complete!")

  return(new_phyloseq)
}