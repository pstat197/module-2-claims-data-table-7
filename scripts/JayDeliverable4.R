## Deliverable 4 script for group 7

## 1. Ensure saved models exist (train + save once if needed)

if (!file.exists("results/models-group7.RData")) {
  message("Saved models not found. Training models via QuinlanPrimaryTask.R ...")
  source("scripts/QuinlanPrimaryTask.R")

  save(
    bin_class,
    multi_model,
    train_terms,
    bclass_levels,
    mclass_levels,
    file = "results/models-group7.RData"
  )

  message("Models trained and saved to results/models-group7.RData")
}

## 2. Load libraries needed for preprocessing and prediction

library(tidyverse)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)
library(tidyr)
library(glmnet)
library(xgboost)

## 3. Preprocessing functions (HTML -> cleaned text -> TF-IDF matrix)

parse_fn <- function(.html) {
  html <- tryCatch(
    read_html(.html),
    error = function(e) return(NA)
  )

  if (is.na(html)) {
    return(tibble(header = NA, text_clean = NA))
  }

  header <- html %>%
    html_elements("h1, h2") %>%
    html_text2() %>%
    str_c(collapse = " ") %>%
    str_squish()

  body <- html %>%
    html_elements("p") %>%
    html_text2() %>%
    str_c(collapse = " ") %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all("'") %>%
    str_replace_all(
      paste(
        c(
          "\n",
          "[[:punct:]]",
          "nbsp",
          "[[:digit:]]",
          "[[:symbol:]]"
        ),
        collapse = "|"
      ),
      " "
    ) %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")

  tibble(
    header = header,
    text_clean = body
  )
}

parse_data <- function(.df) {
  if (!"bclass" %in% names(.df)) .df$bclass <- NA
  if (!"mclass" %in% names(.df)) .df$mclass <- NA

  out <- .df %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest_wider(text_clean)

  return(out)
}

nlp_fn <- function(parse_data.out) {
  if (!"bclass" %in% names(parse_data.out)) parse_data.out$bclass <- NA
  if (!"mclass" %in% names(parse_data.out)) parse_data.out$mclass <- NA

  out <- parse_data.out %>%
    unnest_tokens(
      output = token,
      input = text_clean,
      token = "words",
      stopwords = str_remove_all(stop_words$word, "[[:punct:]]")
    ) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, mclass, token.lem, name = "n") %>%
    bind_tf_idf(
      term = token.lem,
      document = .id,
      n = n
    ) %>%
    pivot_wider(
      id_cols = c(".id", "bclass", "mclass"),
      names_from = "token.lem",
      values_from = "tf_idf",
      values_fill = 0
    )

  return(out)
}

## 4. Load saved models and generate predictions on claims-test

load("results/models-group7.RData")
load("data/claims-test.RData")

## Apply preprocessing pipeline to test data
claims_test_parsed <- parse_data(claims_test)
claims_test_nlp <- nlp_fn(claims_test_parsed)

## Align test features to training features (train_terms)
test_terms <- colnames(claims_test_nlp %>% select(-.id, -bclass, -mclass))

missing_cols <- setdiff(train_terms, test_terms)
claims_test_nlp[missing_cols] <- 0

extra_cols <- setdiff(test_terms, train_terms)
if (length(extra_cols) > 0) {
  claims_test_nlp <- claims_test_nlp %>%
    select(-all_of(extra_cols))
}

claims_test_nlp <- claims_test_nlp %>%
  select(.id, bclass, mclass, all_of(train_terms))

X_test <- claims_test_nlp %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()

## Binary predictions (glmnet)
pred_prob_test <- predict(bin_class, newx = X_test)
pred_label_test <- ifelse(pred_prob_test > 0.5, 1, 0)

## Multiclass predictions (xgboost)
pred_multi_test <- predict(multi_model, X_test)

pred_df <- tibble(
  .id = claims_test_nlp$.id,
  bclass.pred = factor(
    ifelse(
      pred_label_test == 1,
      "Relevant claim content",
      "N/A: No relevant content."
    ),
    levels = bclass_levels
  ),
  mclass.pred = factor(
    pred_multi_test + 1,
    labels = mclass_levels
  )
)

save(pred_df, file = "results/preds-group7-from-saved-models.RData")

print(head(pred_df))
message("Saved predictions to results/preds-group7-from-saved-models.RData")

