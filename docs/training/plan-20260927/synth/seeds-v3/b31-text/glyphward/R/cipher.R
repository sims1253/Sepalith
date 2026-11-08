#' @title Rotate an alphabet
#' @param alphabet A single string of distinct characters.
#' @param k Number of positions to rotate by; negative values rotate back.
#' @return A string of the same characters, starting at position `k + 1`.
#' @keywords internal
shift.alphabet <- function(alphabet, k) {
  chars <- strsplit(alphabet, "", fixed = TRUE)[[1]]
  if (anyDuplicated(chars) > 0L) {
    stop("`alphabet` must not repeat characters.", call. = FALSE)
  }
  n <- length(chars)
  idx <- (seq_len(n) - 1L + k) %% n + 1L
  paste(chars[idx], collapse = "")
}

#' @title Rotation cipher
#' @description Letters are rotated by `k` within their case and, when
#'   `digits` is `TRUE`, digits are rotated by `k` within `0-9`. All other
#'   characters pass through. `rot.decode()` undoes `rot.encode()`.
#' @param x A character vector.
#' @param k Number of positions to rotate by.
#' @param digits Rotate digits as well?
#' @return A character vector the same length as `x`.
#' @rdname rotation
#' @export
rot.encode <- function(x, k = 13L, digits = FALSE) {
  if (!is.numeric(k) || length(k) != 1L || is.na(k)) {
    stop("`k` must be a single number.", call. = FALSE)
  }

  # Lower-case letters
  lower <- paste(letters, collapse = "")
  x <- chartr(lower, shift.alphabet(lower, k), x)

  # Upper-case letters
  upper <- paste(LETTERS, collapse = "")
  x <- chartr(upper, shift.alphabet(upper, k), x)

  # Digits
  if (digits) {
    numerals <- paste(0:9, collapse = "")
    x <- chartr(numerals, shift.alphabet(numerals, k), x)
  }
  x
}

#' @rdname rotation
#' @export
rot.decode <- function(x, k = 13L, digits = FALSE) {
  if (!is.numeric(k) || length(k) != 1L || is.na(k)) {
    stop("`k` must be a single number.", call. = FALSE)
  }

  # Lower-case letters
  lower <- paste(letters, collapse = "")
  x <- chartr(shift.alphabet(lower, k), lower, x)

  # Upper-case letters
  upper <- paste(LETTERS, collapse = "")
  x <- chartr(shift.alphabet(upper, k), upper, x)

  # Digits
  if (digits) {
    numerals <- paste(0:9, collapse = "")
    x <- chartr(shift.alphabet(numerals, k), numerals, x)
  }
  x
}

#' @title Encode or decode with a named cipher
#' @param x A character vector.
#' @param method The cipher to use: `"rot"` rotates letters, `"digits"`
#'   rotates letters and digits, `"atbash"` reverses the alphabet, `"mirror"`
#'   reverses the text and swaps brackets, and `"vowels"` cycles the vowels.
#' @param decode Decode instead of encode?
#' @param k Rotation used by `"rot"` and `"digits"`.
#' @return A character vector the same length as `x`.
#' @details `"atbash"` and `"mirror"` are their own inverses, so `decode` has
#'   no effect for them.
#' @export
cipher.text <- function(x, method = c("rot", "digits", "atbash", "mirror", "vowels"),
                        decode = FALSE, k = 13L) {
  method <- match.arg(method)

  switch(method,
    rot = {
      if (decode) {
        rot.decode(x, k = k)
      } else {
        rot.encode(x, k = k)
      }
    },
    digits = {
      if (decode) {
        rot.decode(x, k = k, digits = TRUE)
      } else {
        rot.encode(x, k = k, digits = TRUE)
      }
    },
    atbash = {
      plain <- paste(c(letters, LETTERS), collapse = "")
      cipher <- paste(c(rev(letters), rev(LETTERS)), collapse = "")
      chartr(plain, cipher, x)
    },
    mirror = {
      reversed <- vapply(strsplit(x, "", fixed = TRUE), function(ch) {
        paste(rev(ch), collapse = "")
      }, character(1))
      reversed[is.na(x)] <- NA_character_
      chartr("()[]{}<>", ")(][}{><", reversed)
    },
    vowels = {
      plain <- "aeiouAEIOU"
      cipher <- "eiouaEIOUA"
      if (decode) {
        chartr(cipher, plain, x)
      } else {
        chartr(plain, cipher, x)
      }
    }
  )
}

#' @title Letter frequencies
#' @param x A character vector; case is ignored.
#' @return A named numeric vector over `letters` giving the share of each
#'   letter among all letters in `x`, or all zeros when there are none.
#' @export
letter.frequency <- function(x) {
  chars <- unlist(strsplit(tolower(x[!is.na(x)]), "", fixed = TRUE), use.names = FALSE)
  chars <- chars[chars %in% letters]
  counts <- table(factor(chars, levels = letters))
  total <- sum(counts)
  if (total == 0L) {
    return(stats::setNames(numeric(26L), letters))
  }
  stats::setNames(as.numeric(counts) / total, letters)
}

#' @title Guess the rotation of a rotation cipher
#' @description Tries every rotation and scores the decoded text by the share
#'   of the letters `"etaoinshr"`, the most frequent letters in English.
#' @param x A character vector of encoded text.
#' @param common Letters expected to be frequent in the plain text.
#' @return A list with elements `k` (the best rotation), `score` (its share of
#'   common letters) and `scores` (the share for every rotation `0:25`).
#' @export
guess.rotation <- function(x, common = "etaoinshr") {
  wanted <- strsplit(common, "", fixed = TRUE)[[1]]
  candidates <- 0:25
  scores <- numeric(length(candidates))
  for (i in seq_along(candidates)) {
    decoded <- rot.decode(x, k = candidates[i])
    freq <- letter.frequency(decoded)
    scores[i] <- sum(freq[wanted])
  }

  names(scores) <- candidates
  best <- which.max(scores)
  list(
    k = candidates[best],
    score = scores[[best]],
    scores = scores
  )
}

#' @title Check that ciphers round-trip
#' @param x A character vector of sample texts.
#' @param methods The ciphers to check.
#' @param k Rotation used by `"rot"` and `"digits"`.
#' @return A named logical vector with one element per method, `TRUE` when
#'   decoding the encoded `x` gives back `x`.
#' @keywords internal
cipher.roundtrip <- function(x, methods = c("rot", "digits", "atbash", "mirror", "vowels"),
                             k = 13L) {
  methods <- match.arg(methods, several.ok = TRUE)
  ok <- logical(length(methods))
  names(ok) <- methods
  for (m in methods) {
    encoded <- cipher.text(x, method = m, k = k)
    decoded <- cipher.text(encoded, method = m, decode = TRUE, k = k)
    ok[[m]] <- identical(decoded, x)
  }
  ok
}
