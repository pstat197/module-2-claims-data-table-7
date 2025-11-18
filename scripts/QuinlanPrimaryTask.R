library(glmnet)
library(xgboost)
library(dplyr)
library(caret)

####-Add Headers-############

## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)
library(tidyr)

# function to parse html and clean text
parse_fn <- function(.html){
  html <- tryCatch(
    read_html(.html),
    error = function(e) return(NA)
  )
  
  if (is.na(html)) {
    return(tibble(header = NA, text_clean = NA))
  }
  
  header <- html %>%
    html_elements('h1, h2') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    str_squish()
  
  
  body <- html %>%  
    html_elements('p') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
  
  tibble(
    header = header,
    text_clean = body
  )
}

# function to apply to claims data
parse_data <- function(.df){
  if (!"bclass" %in% names(.df)) .df$bclass <- NA
  if (!"mclass" %in% names(.df)) .df$mclass <- NA
  
  out <- .df %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest_wider(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  if (!"bclass" %in% names(parse_data.out)) parse_data.out$bclass <- NA
  if (!"mclass" %in% names(parse_data.out)) parse_data.out$mclass <- NA
  
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, mclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass', 'mclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

load("data/claims-raw.RData")
load("data/claims-test.RData")

claims_parsed <- parse_data(claims_raw)
claims_nlp <- nlp_fn(claims_parsed)

unique(claims_nlp$mclass)

####-Binary Classifier-############

X <- claims_nlp %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()

Y <- as.numeric(claims_nlp$bclass) - 1

bin_class <- cv.glmnet(X, Y)

pred_prob <- predict(bin_class, newx = X, s = "lambda.min", type = "response")
pred_label <- ifelse(pred_prob > 0.5, 1, 0)



####-Multiclass Classifier-############

X <- claims_nlp %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()

Y <- as.numeric(claims_nlp$mclass) - 1

train_multi <- xgb.DMatrix(data = X, label = Y)

multi_model <- xgb.train(
  params = list(
    objective = "multi:softmax",
    num_class = length(unique(Y))),
  data = train_multi,
  nrounds = 100
)

pred_multi <- predict(multi_model, X)



bclass_levels <- levels(claims_nlp$bclass)
bclass_pred_factor <- factor(
  ifelse(pred_label == 1,
         "Relevant claim content",
         "N/A: No relevant content."),
  levels = bclass_levels
)

mclass_levels <- levels(claims_nlp$mclass)
mclass_pred_factor <- factor(pred_multi + 1,
                             labels = mclass_levels)

pred_df <- tibble(.id = claims_nlp$.id,
                  bclass_pred = bclass_pred_factor,
                  mclass_pred = mclass_pred_factor,
                  bclass_actual = claims_nlp$bclass,
                  mclass_actual = claims_nlp$mclass)

multi_confusion <- confusionMatrix(
  pred_df$mclass_pred,
  pred_df$mclass_actual
)

binary_confusion <- confusionMatrix(
  pred_df$bclass_pred,
  pred_df$bclass_actual
)

####- Make Predictions on Test ############

claims_test_parsed <- parse_data(claims_test)
claims_test_nlp <- nlp_fn(claims_test_parsed)

train_terms <- colnames(claims_nlp %>% select(-.id, -bclass, -mclass))
test_terms <- colnames(claims_test_nlp %>% select(-.id, -bclass, -mclass))

#add extra terms in train to test 
missing_cols <- setdiff(train_terms, test_terms)
claims_test_nlp[missing_cols] <- 0

#remove extra cols in test
extra_cols <- setdiff(test_terms, train_terms)
claims_test_nlp <- claims_test_nlp %>%
  select(-all_of(extra_cols))

#reorder test cols to train col order 
claims_test_nlp <- claims_test_nlp %>% select(.id, bclass, mclass, all_of(train_terms))

X_test <- claims_test_nlp %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()
  

# Binary 

pred_prob_test <- predict(bin_class, newx = X_test)
pred_label_test <- ifelse(pred_prob_test > 0.5, 1, 0)

# Multiclass

pred_multi_test <- predict(multi_model, X_test)
mclass_pred_test <- factor(pred_multi_test + 1, labels = mclass_levels)

pred_df <- tibble(
  .id = claims_test_nlp$.id,
  bclass.pred = factor(
    ifelse(pred_label_test == 1,
           "Relevant claim content",
           "N/A: No relevant content."),
    levels = levels(claims_nlp$bclass)
  ),
  mclass.pred = factor(
    pred_multi_test + 1,
    labels = levels(claims_nlp$mclass)
  )
)
pred_df
