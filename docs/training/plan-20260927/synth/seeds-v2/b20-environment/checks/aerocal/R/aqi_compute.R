# aqiLevels() and getBreakpoints() live in aqi_breakpoints.R; minimal stand-ins
if (!exists("aqiLevels", inherits = FALSE)) {
  aqiLevels <- function() {
    data.frame(
      indexLo = c(0, 51, 101, 151, 201, 301),
      indexHi = c(50, 100, 150, 200, 300, 500),
      category = c("Good", "Moderate", "USG", "Unhealthy", "Very unhealthy", "Hazardous")
    )
  }
}
if (!exists("getBreakpoints", inherits = FALSE)) {
  getBreakpoints <- function(pollutant) {
    lv <- aqiLevels()
    tbl <- switch(pollutant,
      pm25 = data.frame(
        concLo = c(0.0, 9.1, 35.5, 55.5, 125.5, 225.5),
        concHi = c(9.0, 35.4, 55.4, 125.4, 225.4, 500.4)
      ),
      o3 = data.frame(concLo = c(0, 55, 71, 86, 106), concHi = c(54, 70, 85, 105, 200))
    )
    n <- nrow(tbl)
    tbl$indexLo <- lv$indexLo[seq_len(n)]
    tbl$indexHi <- lv$indexHi[seq_len(n)]
    attr(tbl, "digits") <- if (pollutant == "pm25") 1 else 0
    tbl
  }
}

# truncateConc
stopifnot(truncateConc(0.29, 2) == 0.29, truncateConc(9.19, 1) == 9.1)
stopifnot(identical(truncateConc(c(1.99, NA), 0), c(1, NA)))

# subIndex / indexToConc
si <- subIndex(c(0, 9.0, 9.1, 35.4, NA, -1, 1000), "pm25")
stopifnot(identical(si, c(0L, 50L, 51L, 100L, NA, NA, NA)))
stopifnot(indexToConc(50, "pm25") == 9.0, is.na(indexToConc(600, "pm25")))

# indexCategory
cats <- indexCategory(c(10, 51, 250, -5))
stopifnot(identical(cats, c("Good", "Moderate", "Very unhealthy", NA)))

# weightedRecentAvg
stopifnot(weightedRecentAvg(rep(10, 12)) == 10)
stopifnot(is.na(weightedRecentAvg(c(rep(10, 10), NA, NA))))
wa <- weightedRecentAvg(c(rep(10, 11), 40))
stopifnot(wa > 10, wa < 40)

# overallIndex
d <- data.frame(pm25 = c(9.0, NA), o3 = c(60, NA))
oi <- overallIndex(d)
stopifnot(oi$aqi[1] > 50, oi$dominant[1] == "o3", is.na(oi$aqi[2]), oi$category[1] == "Moderate")

# categoryDays
cd <- categoryDays(data.frame(site = c("a", "a", "b"), pm25 = c(5, 20, 5)))
stopifnot(identical(cd$nValid, c(2, 1)), cd$Good[1] == 1, cd$Moderate[1] == 1)
stopifnot(inherits(try(categoryDays(d, siteCol = "station"), silent = TRUE), "try-error"))

# ozoneIndex
oz <- ozoneIndex(c(rep(50, 8), 130))
stopifnot(nrow(oz) == 9, is.na(oz$o3_8h[7]), oz$index8h[8] == 46L, oz$index1h[9] == 107L)
stopifnot(oz$index[9] == 107L)
