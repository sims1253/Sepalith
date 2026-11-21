#' Start an empty issue table
#'
#' @return A data frame with zero rows and columns `row`, `field`, `value`
#'   and `problem`.
#' @keywords internal
#' @noRd
emptyIssues <- function() {
  data.frame(row = integer(), field = character(), value = character(),
             problem = character(), stringsAsFactors = FALSE)
}

#' Append issues for the failing rows of one check
#'
#' @param issues An issue table.
#' @param field Field name.
#' @param values Field values.
#' @param bad Logical vector, `TRUE` where the check failed.
#' @param problem Description of the problem.
#' @return `issues` with one row added per failing value.
#' @keywords internal
#' @noRd
addIssues <- function(issues, field, values, bad, problem) {
  bad[is.na(bad)] <- FALSE
  if (!any(bad)) {
    return(issues)
  }
  rbind(issues, data.frame(row = which(bad), field = field,
                           value = as.character(values[bad]), problem = problem,
                           stringsAsFactors = FALSE))
}

#' Validate order lines
#'
#' Applies one check per field of an order table: `orderId`, `customerId`,
#' `sku`, `quantity`, `unitPrice`, `discount` and `currency`.
#'
#' @param orders A data frame of order lines.
#' @param currencies Allowed currency codes.
#' @param maxQuantity Largest plausible quantity per line.
#' @return An issue table with columns `row`, `field`, `value` and
#'   `problem`.
#' @family record validators
#' @export
validateOrderLines <- function(orders, currencies = c("EUR", "USD", "GBP"),
                               maxQuantity = 1000) {
  issues <- emptyIssues()

  # Check `orderId`
  bad <- is.na(orders$orderId) | !grepl("^ORD-[0-9]{6}$", orders$orderId)
  issues <- addIssues(issues, "orderId", orders$orderId, bad, "not of the form ORD-nnnnnn")

  # Check `customerId`
  bad <- is.na(orders$customerId) | !grepl("^C[0-9]{5}$", orders$customerId)
  issues <- addIssues(issues, "customerId", orders$customerId, bad, "not of the form Cnnnnn")

  # Check `sku`
  bad <- is.na(orders$sku) | !grepl("^[A-Z]{3}-[0-9]{4}$", orders$sku)
  issues <- addIssues(issues, "sku", orders$sku, bad, "not of the form AAA-nnnn")

  # Check `quantity`
  bad <- is.na(orders$quantity) | orders$quantity < 1 | orders$quantity > maxQuantity
  issues <- addIssues(issues, "quantity", orders$quantity, bad, "outside the plausible range")

  # Check `unitPrice`
  bad <- is.na(orders$unitPrice) | orders$unitPrice <= 0
  issues <- addIssues(issues, "unitPrice", orders$unitPrice, bad, "outside the plausible range")

  # Check `discount`
  bad <- !is.na(orders$discount) & (orders$discount < 0 | orders$discount > 1)
  issues <- addIssues(issues, "discount", orders$discount, bad, "outside the plausible range")

  # Check `currency`
  bad <- is.na(orders$currency) | !orders$currency %in% currencies
  issues <- addIssues(issues, "currency", orders$currency, bad, "not an allowed code")

  issues[order(issues$row), , drop = FALSE]
}

#' Validate the dates of orders
#'
#' Checks that `orderDate` is not in the future, that `shipDate` is not
#' before `orderDate`, and that `deliveryDate` is not before `shipDate`.
#'
#' @param orders A data frame of order lines.
#' @param today Reference date for "the future".
#' @return An issue table with columns `row`, `field`, `value` and
#'   `problem`.
#' @family record validators
#' @export
validateOrderDates <- function(orders, today = Sys.Date()) {
  orderDate <- as.Date(orders$orderDate)
  shipDate <- as.Date(orders$shipDate)
  deliveryDate <- as.Date(orders$deliveryDate)
  issues <- emptyIssues()

  bad <- is.na(orderDate) | orderDate > today
  issues <- addIssues(issues, "orderDate", orders$orderDate, bad, "missing or in the future")

  bad <- !is.na(shipDate) & shipDate < orderDate
  issues <- addIssues(issues, "shipDate", orders$shipDate, bad, "before orderDate")

  bad <- !is.na(deliveryDate) & (is.na(shipDate) | deliveryDate < shipDate)
  issues <- addIssues(issues, "deliveryDate", orders$deliveryDate, bad, "before shipDate")

  issues[order(issues$row), , drop = FALSE]
}

#' Validate order totals
#'
#' @param orders A data frame of order lines.
#' @param totals A data frame with columns `orderId` and `total`.
#' @param tolerance Largest accepted absolute difference.
#' @return A data frame with columns `orderId`, `expected`, `reported` and
#'   `difference` for orders whose totals do not match.
#' @family record validators
#' @export
validateOrderTotals <- function(orders, totals, tolerance = 0.01) {
  absent <- setdiff(c("orderId", "total"), names(totals))
  if (length(absent) > 0L) {
    stop("`totals` lacks column(s): ", paste(absent, collapse = ", "), call. = FALSE)
  }
  lineTotal <- orders$quantity * orders$unitPrice
  expected <- tapply(lineTotal, orders$orderId, sum)
  reported <- totals$total[match(names(expected), totals$orderId)]
  difference <- reported - as.vector(expected)
  bad <- is.na(difference) | abs(difference) > tolerance
  data.frame(
    orderId = names(expected)[bad],
    expected = as.vector(expected)[bad],
    reported = reported[bad],
    difference = difference[bad],
    stringsAsFactors = FALSE
  )
}

#' Validate a postal address
#'
#' Checks that every field is present, then checks the format of
#' `street`, `houseNumber`, `postcode`, `city` and `country` in turn.
#'
#' @param address A named list with elements `street`, `houseNumber`,
#'   `postcode`, `city` and `country`.
#' @return A character vector of problems, empty when the address is valid.
#' @family record validators
#' @export
validateAddress <- function(address) {
  problems <- character()
  fields <- c("street", "houseNumber", "postcode", "city", "country")
  for (field in fields) {
    value <- address[[field]]
    if (is.null(value) || length(value) != 1L || is.na(value) || !nzchar(trimws(value))) {
      problems <- c(problems, sprintf("`%s` is missing.", field))
    }
  }
  if (length(problems) > 0L) {
    return(problems)
  }

  # Format of `street`
  if (!grepl("[A-Za-z]", address$street)) {
    problems <- c(problems, sprintf("`%s` is malformed.", "street"))
  }

  # Format of `houseNumber`
  if (!grepl("^[0-9]+[a-zA-Z]?$", address$houseNumber)) {
    problems <- c(problems, sprintf("`%s` is malformed.", "houseNumber"))
  }

  # Format of `postcode`
  if (!grepl("^[A-Za-z0-9 -]{3,10}$", address$postcode)) {
    problems <- c(problems, sprintf("`%s` is malformed.", "postcode"))
  }

  # Format of `city`
  if (grepl("[0-9]", address$city)) {
    problems <- c(problems, sprintf("`%s` is malformed.", "city"))
  }

  # Format of `country`
  if (!grepl("^[A-Z]{2}$", address$country)) {
    problems <- c(problems, sprintf("`%s` is malformed.", "country"))
  }
  problems
}

#' Validate shipments
#'
#' Applies one check per field of a shipment table: `trackingNo`,
#' `carrier`, `weightKg`, `lengthCm`, `widthCm`, `heightCm` and
#' `insuredValue`. Parcel
#' dimensions must lie in `(0, maxCm]` and the weight in `(0, maxKg]`.
#'
#' @param shipments A data frame of shipments.
#' @param carriers Allowed carrier codes.
#' @param maxKg Heaviest accepted parcel in kilograms.
#' @param maxCm Longest accepted parcel side in centimetres.
#' @return An issue table with columns `row`, `field`, `value` and
#'   `problem`.
#' @family record validators
#' @export
validateShipments <- function(shipments, carriers = c("DHL", "UPS", "DPD"), maxKg = 31.5,
                              maxCm = 120) {
  issues <- emptyIssues()

  # Check `trackingNo`
  bad <- is.na(shipments$trackingNo) | duplicated(shipments$trackingNo)
  issues <- addIssues(issues, "trackingNo", shipments$trackingNo, bad, "missing or duplicated")

  # Check `carrier`
  bad <- is.na(shipments$carrier) | !shipments$carrier %in% carriers
  issues <- addIssues(issues, "carrier", shipments$carrier, bad, "not an allowed code")

  # Check `weightKg`
  bad <- is.na(shipments$weightKg) | shipments$weightKg <= 0 | shipments$weightKg > maxKg
  issues <- addIssues(issues, "weightKg", shipments$weightKg, bad, "outside the plausible range")

  # Check `lengthCm`
  bad <- is.na(shipments$lengthCm) | shipments$lengthCm <= 0 | shipments$lengthCm > maxCm
  issues <- addIssues(issues, "lengthCm", shipments$lengthCm, bad, "outside the plausible range")

  # Check `widthCm`
  bad <- is.na(shipments$widthCm) | shipments$widthCm <= 0 | shipments$widthCm > maxCm
  issues <- addIssues(issues, "widthCm", shipments$widthCm, bad, "outside the plausible range")

  # Check `heightCm`
  bad <- is.na(shipments$heightCm) | shipments$heightCm <= 0 | shipments$heightCm > maxCm
  issues <- addIssues(issues, "heightCm", shipments$heightCm, bad, "outside the plausible range")

  # Check `insuredValue`
  bad <- !is.na(shipments$insuredValue) & shipments$insuredValue < 0
  issues <- addIssues(issues, "insuredValue", shipments$insuredValue, bad,
                      "outside the plausible range")

  issues[order(issues$row), , drop = FALSE]
}
