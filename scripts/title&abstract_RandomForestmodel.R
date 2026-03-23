################################################################################
### Title:            Participatory modeling scoping review - machine learning
### Author:           Nancy Tahmo, PhD student
### Creation date:    January 14, 2025
### Description - Automated title/abstract screening
################################################################################

# ---------------------- Load Libraries --------------------------------------
library(tidyverse)    # For data manipulation
library(tm)           # For text mining
library(caret)        # For machine learning utilities
library(randomForest) # For random forest classification
library(Matrix)       # For handling sparse matrices
library(cld3)         # To check for non-English papers
library(deeplr)       # Translate non-en texts
library(doParallel)   # Leverage parallel processing
library(pROC)         # For displaying ROC plots


#------ STEP 1. Load datasets --------
manually_included <- read.csv("manually_included.csv", stringsAsFactors = FALSE)
manually_excluded <- read.csv("manually_excluded.csv", stringsAsFactors = FALSE)
unlabeled_set <- read.csv("non_screened.csv", stringsAsFactors = FALSE) # Papers yet to be screened
nrow(unlabeled_set)

#------ STEP 2. Preprocess the data for modeling  --------
##* Add labels to labeled set
manually_included$label <- 1
manually_excluded$label <- 0

##* Combine labeled datasets
labeled_set <- bind_rows(manually_included, manually_excluded)

##* Combine title and abstract texts for analysis
labeled_set$text <- paste(labeled_set$Title, labeled_set$Abstract, sep = " ")
unlabeled_set$text <- paste(unlabeled_set$Title, unlabeled_set$Abstract, sep = " ")

# Check for duplicates between labeled and unlabeled sets
duplicates <- intersect(labeled_set$Title, unlabeled_set$Title)  

# 5 duplicates found. Remove duplicates from unlabeled set to eliminate data leakage and performance inflation
unlabeled_set <- unlabeled_set[!unlabeled_set$Title %in% labeled_set$Title, ]

##* Detect language for labeled + unlabeled sets
labeled_set <- labeled_set %>%
  mutate(lang_iso = cld3::detect_language(text),
         needs_translation = if_else(lang_iso != "en" & !is.na(lang_iso), TRUE, FALSE))

unlabeled_set <- unlabeled_set %>%
  mutate(lang_iso = cld3::detect_language(text),
         needs_translation = if_else(lang_iso != "en" & !is.na(lang_iso), TRUE, FALSE))

# Quick counts
table(labeled_set$lang_iso)
table(unlabeled_set$lang_iso)

## See non-English papers
labeled_non_en <- labeled_set %>% filter(needs_translation)

unlabeled_non_en <- unlabeled_set %>% filter(needs_translation)

##* Translate non-English citations
# Authenticate (store your DeepL API key as an env variable for safety)
DEEPL_KEY <- "YOUR_DEEPL_KEY_HERE"  # 

## Translate the non-EN in UNLABELED
idx_unlab <- which(unlabeled_non_en$needs_translation)
unlabeled_non_en$text_en   <- unlabeled_non_en$text
unlabeled_non_en$text_en[idx_unlab] <- deeplr::translate2(
  unlabeled_non_en$text[idx_unlab],
  target_lang = "EN",
  auth_key    = DEEPL_KEY)

unlabeled_non_en$text_en

## Update text column to English
unlabeled_non_en$Covidence..

unlabeled_set <- unlabeled_set %>%
  mutate(text_org = text) # Save the original text

unlabeled_set$text[unlabeled_set$Covidence.. == "#7934"] <- 
  unlabeled_non_en$text_en[unlabeled_non_en$Covidence.. == "#7934"]

# Verify translation
unlabeled_set <- unlabeled_set %>%
  mutate(lang_iso2 = cld3::detect_language(text),
         needs_translation2 = if_else(lang_iso2 != "en" & !is.na(lang_iso2), TRUE, FALSE))

table(unlabeled_set$lang_iso2)

##* Define a text preprocessing function for both datasets
clean_text <- function(text) {
  text <- tolower(text)  # Convert to lowercase
  text <- removePunctuation(text)  # Remove punctuations
  text <- removeNumbers(text)  # Remove numbers
  text <- removeWords(text, stopwords("en"))  # Remove common stopwords
  text <- stripWhitespace(text)  # Remove extra whitespace
  return(text)
}

# Apply text cleaning to the datasets
labeled_set$text    <- vapply(labeled_set$text, clean_text, "", USE.NAMES = FALSE)
unlabeled_set$text    <- vapply(unlabeled_set$text, clean_text, "", USE.NAMES = FALSE)

##* Split the labeled set into train and test sets
y_all <- factor(labeled_set$label, levels = c(1,0), labels = c("include","exclude"))

set.seed(416)

idx_tr <- createDataPartition(y_all, p = 0.70, list = FALSE)

train_text <- labeled_set$text[idx_tr]
test_text  <- labeled_set$text[-idx_tr]
y_tr <- y_all[idx_tr]
y_te <- y_all[-idx_tr]

##* Build Corpora for modeling and prediction 
# Train set and constrained on test and unlabeled
# Custom tokenizer for unigrams + bigrams
bigramTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 1:2), paste, collapse = " "), use.names = FALSE)
}

## Train DTM with unigrams + bigrams
dtm_train <- DocumentTermMatrix(
  Corpus(VectorSource(train_text)),
  control = list(tokenize = bigramTokenizer,
                 weighting = weightTf)   # raw counts (you can use weightTfIdf)
)

## Apply train dictionary to test/unlabeled
dtm_test <- DocumentTermMatrix(
  Corpus(VectorSource(test_text)),
  control = list(tokenize = bigramTokenizer,
                 dictionary = Terms(dtm_train),
                 weighting = weightTf)
)

dtm_unlabeled <- DocumentTermMatrix(
  Corpus(VectorSource(unlabeled_set$text)),
  control = list(tokenize = bigramTokenizer,
                 dictionary = Terms(dtm_train),
                 weighting = weightTf)
)

# Convert to matrices
x_train <- as.matrix(dtm_train)
x_test  <- as.matrix(dtm_test)
x_unlab <- as.matrix(dtm_unlabeled)


#------ STEP 3. Model training with 5-fold cross validation on train set only ------
set.seed(416)

# Prepare to run via parallel processing
n_cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(n_cores)
registerDoParallel(cl)

ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5, allowParallel = F, classProbs = TRUE)

rf_fit <- train(
  x = x_train,
  y = y_tr,
  method = "rf",
  trControl = ctrl,
  tuneLength = 5,    
  ntree = 250,
  importance = TRUE)


# Save model 
saveRDS(rf_fit, file = "rf_fit_model.rds")

print(rf_fit$results)
print(rf_fit$bestTune)

# Read model
rf_fit <- readRDS("rf_fit_model_unigrams & bigrams.rds")

#------ STEP 4. Evaluate model performance on untouched test set -------
pred_te <- predict(rf_fit, x_test)

confusionMatrix(pred_te, y_te, positive = "include")

##* Compute 95% CI

## --- Copy model to evaluate ---
model <- rf_fit

## --- Predictions on TEST set ---
pred_cls  <- predict(model, x_test)
pred_prob <- predict(model, x_test, type = "prob")[, "include"]

## --- Point estimates (positive="include") ---
cm <- confusionMatrix(pred_cls, y_te, positive = "include")
acc  <- as.numeric(cm$overall["Accuracy"])
sens <- as.numeric(cm$byClass["Sensitivity"])
spec <- as.numeric(cm$byClass["Specificity"])

roc_obj <- roc(y_te, pred_prob, levels = c("exclude","include"))
auc_pt  <- as.numeric(auc(roc_obj))

## --- 95% CIs via nonparametric bootstrap ---
set.seed(416)
B <- 1000
n <- length(y_te)

boot_mat <- replicate(B, {
  idx <- sample.int(n, n, replace = TRUE)
  cm_b <- confusionMatrix(
    factor(ifelse(pred_prob[idx] >= 0.5, "include","exclude"),
           levels = c("exclude","include")),
    y_te[idx],
    positive = "include"
  )
  c(
    acc  = as.numeric(cm_b$overall["Accuracy"]),
    sens = as.numeric(cm_b$byClass["Sensitivity"]),
    spec = as.numeric(cm_b$byClass["Specificity"])
  )
})

ci_acc  <- quantile(boot_mat["acc",],  probs = c(0.025, 0.975), na.rm = TRUE)
ci_sens <- quantile(boot_mat["sens",], probs = c(0.025, 0.975), na.rm = TRUE)
ci_spec <- quantile(boot_mat["spec",], probs = c(0.025, 0.975), na.rm = TRUE)

## --- AUC + 95% CI (DeLong) ---
auc_ci <- ci.auc(roc_obj)  # returns 2.5%, 50%, 97.5%

## --- Neat printout ---
cat("\nPoint estimates:\n")
cat(sprintf("Accuracy   = %.3f\nSensitivity= %.3f\nSpecificity= %.3f\nAUC        = %.3f\n",
            acc, sens, spec, auc_pt))

cat("\n95%% Bootstrap CIs:\n")
cat(sprintf("Accuracy   95%% CI: [%.3f, %.3f]\n", ci_acc[1],  ci_acc[2]))
cat(sprintf("Sensitivity 95%% CI: [%.3f, %.3f]\n", ci_sens[1], ci_sens[2]))
cat(sprintf("Specificity 95%% CI: [%.3f, %.3f]\n", ci_spec[1], ci_spec[2]))

cat("\nAUC 95% CI (DeLong):\n")
cat(sprintf("[%.3f, %.3f]\n", auc_ci[1], auc_ci[3]))


#------ STEP 5. Predict inclusion/exclusion of unlabeled set ------
unlabeled_set$predicted_label <- predict(rf_fit, x_unlab)

unique(unlabeled_set$predicted_label)

# Export model predictions
write.csv(unlabeled_set, "unlabeled_set_predictions.csv", row.names = FALSE)
