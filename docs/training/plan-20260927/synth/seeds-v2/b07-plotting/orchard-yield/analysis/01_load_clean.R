# Load raw harvest records and plot metadata, clean them, write tidy data.

raw_dir <- file.path("data", "raw")
out_dir <- file.path("data", "derived")
fig_dir <- file.path("output", "figures", "qc")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

read_harvest_file <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE, na.strings = c("", "NA", "n/a", "-"))
  names(df) <- tolower(gsub("[^A-Za-z0-9]+", "_", names(df)))
  df$source_file <- basename(path)
  df
}

read_all_harvests <- function(dir) {
  files <- list.files(dir, pattern = "^harvest_.*\\.csv$", full.names = TRUE)
  if (length(files) == 0) {
    stop("No harvest files found in ", dir)
  }

  pieces <- vector("list", length(files))
  for (i in seq_along(files)) {
    pieces[[i]] <- read_harvest_file(files[i])
  }

  # Files from different seasons do not always carry the same columns
  all_cols <- unique(unlist(lapply(pieces, names)))
  pieces <- lapply(pieces, function(p) {
    p[setdiff(all_cols, names(p))] <- NA
    p[all_cols]
  })
  do.call(rbind, pieces)
}

parse_harvest_date <- function(x) {
  out <- as.Date(x, format = "%Y-%m-%d")
  # The 2019 and 2020 sheets were typed by hand in day/month/year
  alt <- is.na(out) & !is.na(x)
  out[alt] <- as.Date(x[alt], format = "%d/%m/%Y")
  out
}

clean_variety <- function(x) {
  x <- tolower(trimws(x))
  x <- gsub("\\s+", " ", x)

  lookup <- c(
    "gala" = "Gala", "royal gala" = "Gala",
    "braeburn" = "Braeburn",
    "elstar" = "Elstar",
    "jonagold" = "Jonagold", "jona gold" = "Jonagold",
    "boskoop" = "Boskoop", "schoner aus boskoop" = "Boskoop"
  )
  out <- unname(lookup[x])
  out[is.na(out)] <- "Other"

  factor(out, levels = c("Gala", "Braeburn", "Elstar", "Jonagold", "Boskoop", "Other"))
}

flag_outliers <- function(x, k = 3) {
  med <- median(x, na.rm = TRUE)
  spread <- mad(x, na.rm = TRUE)
  if (is.na(spread) || spread == 0) {
    return(rep(FALSE, length(x)))
  }
  !is.na(x) & abs(x - med) / spread > k
}

clean_harvest <- function(df, plots) {
  df$harvest_date <- parse_harvest_date(df$harvest_date)
  df$season <- as.integer(format(df$harvest_date, "%Y"))
  df$variety <- clean_variety(df$variety)
  df$yield_kg <- as.numeric(df$yield_kg)
  df$trees_picked <- as.integer(df$trees_picked)

  # Rows without a plot or a date cannot be placed anywhere
  df <- df[!is.na(df$plot_id) & !is.na(df$harvest_date), ]
  df <- df[is.na(df$yield_kg) | df$yield_kg >= 0, ]

  df$yield_per_tree <- df$yield_kg / df$trees_picked
  df$yield_per_tree[!is.finite(df$yield_per_tree)] <- NA
  # Outliers are judged within a variety, not against the whole orchard
  df$outlier <- ave(df$yield_per_tree, df$variety, FUN = flag_outliers) == 1

  merged <- merge(df, plots, by = "plot_id", all.x = TRUE)
  if (any(is.na(merged$block))) {
    warning(sum(is.na(merged$block)), " rows have no matching plot metadata")
  }

  merged[order(merged$season, merged$plot_id, merged$harvest_date), ]
}

plot_qc_overview <- function(df, path) {
  png(path, width = 1800, height = 900, res = 150)
  on.exit(dev.off(), add = TRUE)
  op <- par(mfrow = c(1, 2), mar = c(4, 4, 2, 1), las = 1)
  on.exit(par(op), add = TRUE, after = FALSE)

  boxplot(yield_per_tree ~ variety, data = df, col = "#a7b7c7", border = "grey30",
          xlab = "", ylab = "Yield per tree (kg)", main = "Yield by variety")
  with(df[df$outlier, ], points(as.integer(variety), yield_per_tree, pch = 4, col = "#c8553d"))

  counts <- table(df$season)
  barplot(counts, col = "#1f4e79", border = NA, ylab = "Records", main = "Records per season")

  invisible(path)
}

plots_meta <- read.csv(file.path(raw_dir, "plots.csv"), stringsAsFactors = FALSE)
plots_meta$block <- factor(plots_meta$block)
plots_meta$rootstock <- factor(plots_meta$rootstock)

harvest_raw <- read_all_harvests(raw_dir)
harvest <- clean_harvest(harvest_raw, plots_meta)

message("Rows read: ", nrow(harvest_raw), "; rows kept: ", nrow(harvest))
message("Flagged outliers: ", sum(harvest$outlier, na.rm = TRUE))

plot_qc_overview(harvest, file.path(fig_dir, "qc_overview.png"))
saveRDS(harvest, file.path(out_dir, "harvest_clean.rds"))
write.csv(harvest, file.path(out_dir, "harvest_clean.csv"), row.names = FALSE)
