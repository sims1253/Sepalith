#' Common type of two money vectors
#'
#' Two money vectors combine only if they share a currency.
#'
#' @param x,y Vectors to combine.
#' @param ... Ignored.
#' @return A zero-length `foliokit_money` prototype.
#' @keywords internal
#' @export
vec_ptype2.foliokit_money.foliokit_money <- function(x, y, ...) {
  new_money(currency = check_same_currency(x, y))
}

#' Common type of money and a double vector
#'
#' Plain doubles are treated as amounts in the currency of the money vector.
#'
#' @param x,y Vectors to combine.
#' @param ... Ignored.
#' @return A zero-length `foliokit_money` prototype.
#' @keywords internal
#' @export
vec_ptype2.foliokit_money.double <- function(x, y, ...) {
  new_money(currency = attr(x, "currency"))
}

#' Common type of a double vector and money
#'
#' Plain doubles are treated as amounts in the currency of the money vector.
#'
#' @param x,y Vectors to combine.
#' @param ... Ignored.
#' @return A zero-length `foliokit_money` prototype.
#' @keywords internal
#' @export
vec_ptype2.double.foliokit_money <- function(x, y, ...) {
  new_money(currency = attr(y, "currency"))
}

#' Cast money to money
#'
#' @param x Vector to cast.
#' @param to Prototype to cast to.
#' @param ... Ignored.
#' @return A `foliokit_money` vector in the currency of `to`.
#' @keywords internal
#' @export
vec_cast.foliokit_money.foliokit_money <- function(x, to, ...) {
  check_same_currency(x, to)
  x
}

#' Cast a double vector to money
#'
#' @param x Vector to cast.
#' @param to Prototype to cast to.
#' @param ... Ignored.
#' @return A `foliokit_money` vector in the currency of `to`.
#' @keywords internal
#' @export
vec_cast.foliokit_money.double <- function(x, to, ...) {
  money(x, currency = attr(to, "currency"))
}

#' Cast money to a double vector
#'
#' @param x Vector to cast.
#' @param to Prototype to cast to.
#' @param ... Ignored.
#' @return A double vector of amounts with the currency dropped.
#' @keywords internal
#' @export
vec_cast.double.foliokit_money <- function(x, to, ...) {
  vctrs::vec_data(x)
}

#' Arithmetic on money vectors
#'
#' Money can be added to or subtracted from money in the same currency, and
#' multiplied or divided by plain numbers. Dividing money by money gives a
#' plain ratio.
#'
#' @param op An arithmetic operator, as a string.
#' @param x,y Operands.
#' @param ... Ignored.
#' @return A `foliokit_money` vector, or a double vector for ratios.
#' @keywords internal
#' @export
vec_arith.foliokit_money <- function(op, x, y, ...) {
  UseMethod("vec_arith.foliokit_money", y)
}

#' @rdname vec_arith.foliokit_money
#' @export
vec_arith.foliokit_money.foliokit_money <- function(op, x, y, ...) {
  cur <- check_same_currency(x, y)
  switch(op,
    "+" = ,
    "-" = new_money(vctrs::vec_arith_base(op, x, y), currency = cur),
    "/" = vctrs::vec_arith_base(op, x, y),
    vctrs::stop_incompatible_op(op, x, y)
  )
}

#' @rdname vec_arith.foliokit_money
#' @export
vec_arith.foliokit_money.numeric <- function(op, x, y, ...) {
  cur <- attr(x, "currency")
  switch(op,
    "*" = ,
    "/" = money(vctrs::vec_arith_base(op, x, y), currency = cur),
    vctrs::stop_incompatible_op(op, x, y)
  )
}
