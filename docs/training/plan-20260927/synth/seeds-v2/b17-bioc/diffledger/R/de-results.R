#' Standardise a differential-expression result table
#'
#' Renames tool-specific columns to the common `feature`, `log2fc`,
#' `mean_expr`, `stat`, `pvalue` columns used throughout the package.
#'
#' @param res A data frame of differential-expression results.
#' @param source Which tool produced `res`.
#' @param feature_col Column holding feature identifiers; row names are used
#'   when `NULL`.
#' @return A tibble with standardised column names.
#' @export
as_de_table <- function(res,
                        source = c("generic", "deseq", "edger", "limma"),
                        feature_col = NULL) {
  source <- match.arg(source)

  # Map our standard names (left) to each tool's column names (right).
  mapping <- switch(source,
    generic = c(log2fc = "log2fc", mean_expr = "mean_expr", stat = "stat", pvalue = "pvalue"),
    deseq = c(log2fc = "log2FoldChange", mean_expr = "baseMean", stat = "stat", pvalue = "pvalue"),
    edger = c(log2fc = "logFC", mean_expr = "logCPM", stat = "LR", pvalue = "PValue"),
    limma = c(log2fc = "logFC", mean_expr = "AveExpr", stat = "t", pvalue = "P.Value")
  )

  missing_cols <- setdiff(mapping, names(res))
  if (length(missing_cols) > 0L) {
    stop(
      sprintf("columns missing for source '%s': %s", source, paste(missing_cols, collapse = ", ")),
      call. = FALSE
    )
  }

  # Most tools put the feature IDs in the row names, which a tibble drops.
  feature <- if (is.null(feature_col)) rownames(res) else as.character(res[[feature_col]])
  res |>
    dplyr::as_tibble() |>
    dplyr::rename(dplyr::all_of(mapping)) |>
    dplyr::mutate(feature = feature, .before = 1L) |>
    dplyr::select(feature, log2fc, mean_expr, stat, pvalue)
}

#' Add adjusted p-values
#'
#' @param de A tibble of standardised differential-expression results.
#' @param method Multiple-testing correction passed to [stats::p.adjust()].
#' @return `de` with a `padj` column.
#' @export
add_padj <- function(de, method = "BH") {
  de |>
    dplyr::mutate(padj = stats::p.adjust(pvalue, method = method))
}

#' Classify features as up, down or unchanged
#'
#' @description Features without an adjusted p-value (for example those
#'   removed by independent filtering) are labelled `"untested"`.
#'
#' @param de A tibble of standardised differential-expression results.
#' @param alpha Significance threshold on adjusted p-values.
#' @param lfc_min Minimum absolute log2 fold change.
#' @return `de` with a factor column `direction`.
#' @export
classify_de <- function(de, alpha = 0.05, lfc_min = 1) {
  if (!"padj" %in% names(de)) {
    de <- add_padj(de)
  }

  de |>
    dplyr::mutate(
      direction = dplyr::case_when(
        is.na(padj) ~ "untested",
        padj < alpha & log2fc >= lfc_min ~ "up",
        padj < alpha & log2fc <= -lfc_min ~ "down",
        TRUE ~ "unchanged"
      ),
      direction = factor(direction, levels = c("up", "down", "unchanged", "untested"))
    )
}

#' Count features per direction
#'
#' @param de A tibble of standardised differential-expression results.
#' @param alpha Significance threshold on adjusted p-values.
#' @param lfc_min Minimum absolute log2 fold change.
#' @return A tibble with one row per direction and columns `n` and `pct`.
#' @export
summarise_de <- function(de, alpha = 0.05, lfc_min = 1) {
  de |>
    classify_de(alpha = alpha, lfc_min = lfc_min) |>
    dplyr::count(direction, .drop = FALSE) |>
    dplyr::mutate(pct = round(100 * n / sum(n), 1))
}

#' Top differentially expressed features
#'
#' Ranks features by the chosen criterion and returns the first `n`,
#' optionally restricted to one direction.
#'
#' @param de A tibble of standardised differential-expression results.
#' @param n Number of features to return.
#' @param by Ranking criterion.
#' @param direction Restrict to `"up"` or `"down"` features, or `"both"`.
#' @return A tibble with at most `n` rows.
#' @export
top_de <- function(de, n = 20, by = c("padj", "pvalue", "abs_lfc", "stat"),
                   direction = c("both", "up", "down")) {
  by <- match.arg(by)
  direction <- match.arg(direction)
  if (!"padj" %in% names(de)) {
    de <- add_padj(de)
  }

  if (direction == "up") {
    de <- dplyr::filter(de, log2fc > 0)
  } else if (direction == "down") {
    de <- dplyr::filter(de, log2fc < 0)
  }

  # Ties on padj are common after BH adjustment; break them by raw p-value.
  ranked <- switch(by,
    padj = dplyr::arrange(de, padj, pvalue),
    pvalue = dplyr::arrange(de, pvalue),
    abs_lfc = dplyr::arrange(de, dplyr::desc(abs(log2fc))),
    stat = dplyr::arrange(de, dplyr::desc(abs(stat)))
  )
  utils::head(ranked, n)
}

#' Prepare data for a volcano plot
#'
#' Adds `-log10` p-values, direction and a label column for the top hits.
#'
#' @param de A tibble of standardised differential-expression results.
#' @param alpha Significance threshold on adjusted p-values.
#' @param lfc_min Minimum absolute log2 fold change.
#' @param n_labels Number of top features to label.
#' @return A tibble with columns `neg_log10_p`, `direction` and `label`.
#' @export
volcano_data <- function(de, alpha = 0.05, lfc_min = 1, n_labels = 10) {
  classified <- classify_de(de, alpha = alpha, lfc_min = lfc_min)

  # Only significant features are candidates for a text label.
  labelled <- classified |>
    dplyr::filter(direction %in% c("up", "down")) |>
    top_de(n = n_labels, by = "padj") |>
    dplyr::pull(feature)

  # p-values of exactly zero would plot at infinity; clamp them to the
  # smallest positive p-value (or the machine minimum if there is none).
  positive_p <- classified$pvalue[!is.na(classified$pvalue) & classified$pvalue > 0]
  floor_p <- if (length(positive_p) > 0L) min(positive_p) else .Machine$double.xmin

  classified |>
    dplyr::mutate(
      neg_log10_p = -log10(pmax(pvalue, floor_p)),
      label = dplyr::if_else(feature %in% labelled, feature, NA_character_)
    )
}

#' Compare two contrasts
#'
#' Joins two result tables by feature and tabulates agreement between their
#' directions.
#'
#' @param de_a,de_b Tibbles of standardised differential-expression results.
#' @param alpha Significance threshold on adjusted p-values.
#' @param lfc_min Minimum absolute log2 fold change.
#' @return A list with the joined table `joined`, a contingency table
#'   `agreement` and the fold-change `correlation` (`NA` with fewer than
#'   three shared features).
#' @export
compare_contrasts <- function(de_a, de_b, alpha = 0.05, lfc_min = 1) {
  a <- classify_de(de_a, alpha = alpha, lfc_min = lfc_min) |>
    dplyr::select(feature, log2fc_a = log2fc, direction_a = direction)
  b <- classify_de(de_b, alpha = alpha, lfc_min = lfc_min) |>
    dplyr::select(feature, log2fc_b = log2fc, direction_b = direction)

  joined <- dplyr::inner_join(a, b, by = "feature")
  if (nrow(joined) == 0L) {
    warning("the two contrasts share no features")
  }

  complete <- stats::complete.cases(joined$log2fc_a, joined$log2fc_b)
  correlation <- if (sum(complete) >= 3L) {
    stats::cor(joined$log2fc_a[complete], joined$log2fc_b[complete])
  } else {
    NA_real_
  }

  list(
    joined = joined,
    agreement = table(a = joined$direction_a, b = joined$direction_b),
    correlation = correlation
  )
}
