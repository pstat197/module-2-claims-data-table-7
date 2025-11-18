## Deliverable 3 script for group 7
## Runs the primary-task pipeline and saves predictions
source("scripts/QuinlanPrimaryTask.R")

save(pred_df, file = "results/preds-group7.RData")