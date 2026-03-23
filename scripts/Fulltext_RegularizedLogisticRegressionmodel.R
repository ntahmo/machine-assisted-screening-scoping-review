################################################################################
### Title:         Participatory modeling scoping review - machine learning
### Author:        Nancy Tahmo, PhD Student
### Creation date: August 07, 2025
### Description:   Automates full-text inclusion and exclusion classification
################################################################################

# ---------------------- Load Libraries --------------------------------------

library(tidyverse)   # Data manipulation, wrangling, and visualization (dplyr, ggplot2, etc.)
library(pdftools)    # Extracts text and metadata from PDF files
library(tesseract)   # Optical Character Recognition (OCR) for image-based or poorly encoded PDFs
library(cld3)        # Automatic language detection (Google Compact Language Detector v3)
library(furrr)       # Parallelized functional programming (map_* with multicore backends)
library(tidymodels)  # Unified framework for machine learning modeling and evaluation
library(textrecipes) # Preprocessing recipes for text data (tokenization, TF–IDF, etc.)
library(stopwords)   # Provides multilingual stopword lists for text cleaning
library(digest)      # Hashing utility (used to create reproducible cache keys for PDFs)
library(readr)       # Fast reading and writing of CSV/text files
library(purrr)       # Functional programming (map, safely, etc.)
library(Hmisc)       # Descriptive statistics, summaries, and labeling utilities
library(text2vec)    # Efficient text vectorization and modeling (tokenization, TF–IDF)
library(dplyr)       # Grammar of data manipulation (filter, mutate, select, joins)
library(tidyr)       # Reshaping and tidying data frames
library(yardstick)   # Model performance metrics (accuracy, recall, F1, specificity, etc.)
library(glmnet)      # Regularized regression (Ridge) used for logistic models
library(stringr)     # Manipulate text
library(stringdist)  # Jaro Winkler matching

#* Parallel Processing Setup
plan(multisession, workers = 4)  # Enable parallel processing with 4 worker sessions


# ---------------------- STEP 1. Load and label data -------------------------
manually_included <- read_csv("manually_included") %>%
  mutate(Label = "Included") %>%
  mutate(across(everything(), as.character))


manually_excluded <- read_csv("manually_excluded") %>%
  mutate(across(everything(), as.character)) %>%
  mutate(Label = "Excluded") %>%
  mutate(Clean_Reason = str_extract(Notes, "(?<=Exclusion reason:\\s)(.*?)(?=;)") %>% str_trim()) %>%
  
# Edit exclusion reasons as necessary to facilitate model training  
  mutate(Reason = case_when(
    Clean_Reason == "No knowledge user participation" ~ "No knowledge user participation",
    Clean_Reason == "Preprint" ~ "Preprint",
    Clean_Reason == "Review" ~ "Review",
    TRUE ~ "Other"
  ))

unlabeled_set <- read_csv("unlabeled_set") %>%
  mutate(Label = "Unlabeled") %>%
  mutate(across(everything(), as.character))

un.labeled_sets <- bind_rows(included, excluded, unlabeled) # combine all datasets, labeled and unlabeled

# ---------------------- STEP 2. Match citations to PDF Files --------------------------
# 1. Read in the Covidence file containing all the citations + column containing the PDF location
pdf_index <- read_csv("YOUR_COVIDENCE_FILE") %>%
  mutate(
    PDF_File = File_clean,  # Rename for clarity
    # Extract author portion before the first " - YYYY - "
    Authors_Extracted = str_extract(PDF_File, "^[^-]+(?= - \\d{4} - )"),
    
    # Extract title portion after " - YYYY - "
    Title_Extracted = PDF_File %>%
      str_remove("^.*? - \\d{4} - ") %>%   # Remove everything before title
      str_remove("\\.pdf$") %>%            # Remove file extension
      str_to_lower() %>%
      str_replace_all("[[:punct:][:space:]]+", ""))

# 2. Prepare `un.labeled_sets` titles for matching with the Covidence file
un.labeled_sets <- un.labeled_sets %>%
  mutate(
    norm_title = Title %>%
      str_to_lower() %>%
      str_replace_all("[[:punct:][:space:]]+", ""))

# Check for duplicates
un.labeled_sets %>%
  count(norm_title) %>%
  filter(n > 1)

pdf_index %>%
  count(Title_Extracted) %>%
  filter(n > 1)

# 13 & 15 duplicates found in the normalized, respectively 
# Double-check author names
options(dplyr.print_min = Inf, dplyr.print_max = Inf)

pdf_index %>%
  count(Title_Extracted) %>%
  filter(n > 1) %>%
  left_join(pdf_index, by = "Title_Extracted") %>%
  select(Title_Extracted, Authors_Extracted, PDF_File) %>%
  arrange(Title_Extracted)

# Remove the second of each duplicate citation
pdf_index_cln <- pdf_index %>%
  distinct(Title_Extracted, .keep_all = TRUE)

un.labeled_sets_cln <- un.labeled_sets %>%
  distinct(norm_title, .keep_all = TRUE)
  
# Inspect
pdf_index_cln %>%
  count(Title_Extracted) %>%
  filter(n > 1)

pdf_index_cln$Title_Extracted
un.labeled_sets_cln$norm_title

# 3. Join using cleaned titles
un.labeled_sets_cln2 <- un.labeled_sets_cln %>%
  left_join(pdf_index_cln %>% select(Title_Extracted, PDF_File),
            by = c("norm_title" = "Title_Extracted")) %>%
  mutate(PDF_Path = file.path("YOUR_FOLDER_DIRECTORY_HERE/All_PDF_Files", PDF_File))

# Diagnostics
cat("✅ Matching complete. Unmatched rows:", sum(is.na(un.labeled_sets_cln2$PDF_File)), "\n")

# 516 rows unmatched

# Step 3.5 - Attempt fuzzy matching on unmatched rows
unmatched <- un.labeled_sets_cln2 %>%
  filter(is.na(PDF_File))

# Function to find closest match in pdf_index
find_closest_match <- function(x, pdf_titles, pdf_files) {
  distances <- stringdist::stringdist(x, pdf_titles, method = "jw")  # Jaro-Winkler distance
  best <- which.min(distances)
  if (distances[best] < 0.2) {
    return(pdf_files[best])
  } else {
    return(NA_character_)
  }
}


# Apply it to all unmatched rows
fuzzy_matched <- unmatched %>%
  mutate(
    PDF_File = map_chr(norm_title, ~ find_closest_match(.x, pdf_index_cln$Title_Extracted, pdf_index_cln$PDF_File)),
    PDF_Path = ifelse(!is.na(PDF_File), file.path("YOUR_FOLDER_DIRECTORY_HERE/All_PDF_Files", PDF_File), NA_character_)
  )

# Check duplicate matches
# Identify duplicated PDF_Path values
dup_paths <- fuzzy_matched %>%
  count(PDF_Path) %>%
  filter(n > 1) %>%
  pull(PDF_Path)


# Display the full rows with duplicated PDF_Path
fuzzy_matched %>%
  filter(PDF_Path %in% dup_paths) %>%
  arrange(PDF_Path)

# Checking real duplicates
fuzzy_matched %>%
  group_by(PDF_Path) %>%
  filter(n() > 1) %>%
  select(DOI, Title, Authors, PDF_Path, `Covidence #`)

# Remove the only duplicates
fuzzy_matched <- fuzzy_matched %>%
  distinct(DOI, PDF_Path, .keep_all = TRUE)

# Manually correct the path for the incorrect match
fuzzy_matched <- fuzzy_matched %>%
  mutate(PDF_Path = case_when(
    `Covidence #` == "#1120" ~ "YOUR_FOLDER_DIRECTORY_HERE/All_PDF_Files/Kolaye et al. - 2019 - Mathematical assessment of the role of environmental factors on the dynamical transmission of choler.pdf",
    TRUE ~ PDF_Path
  ))

# Inspect
fuzzy_matched %>%
  count(PDF_Path) %>%
  filter(n > 1)

# Merge the fuzzy-matched rows back
un.labeled_sets_cln2 <- un.labeled_sets_cln2 %>%
  filter(!is.na(PDF_File)) %>%
  bind_rows(fuzzy_matched)

# Final check
cat("✅ Final matching complete. Remaining unmatched rows:", sum(is.na(un.labeled_sets_cln2$PDF_File)), "\n")

# Check for existence of files at the given paths
un.labeled_sets_cln2 <- un.labeled_sets_cln2 %>%
  mutate(File_Exists = file.exists(PDF_Path))

# Diagnostics
missing_files <- sum(!un.labeled_sets_cln2$File_Exists, na.rm = TRUE)
cat("✅ File existence check complete. Missing files:", missing_files, "\n")

# Print missing file names for manual checking
if (missing_files > 0) {
  print(un.labeled_sets_cln2 %>% filter(!File_Exists) %>% select(`Covidence #`, PDF_Path, norm_title))
}

# Manually correct the path for the missing file matches
un.labeled_sets_cln2 <- un.labeled_sets_cln2 %>%
  mutate(PDF_Path = case_when(
    `Covidence #` == "#324" ~ "YOUR_FOLDER_DIRECTORY_HERE/All_PDF_Files/Madubueze et al. - 2023 - A deterministic mathematical model for optimal control of diphtheria disease with booster vaccination.pdf.pdf",
    `Covidence #` == "#355" ~ "YOUR_FOLDER_DIRECTORY_HERE/All_PDF_Files/All PDFs/Broekaert et al. - 2024 - A comparative cost assessment of coalescing epidemic control strategies in heterogeneous social-contact networks.pdf.pdf",
    TRUE ~ PDF_Path))


# Inspect
un.labeled_sets_cln2 <- un.labeled_sets_cln2 %>%
  mutate(File_Exists = file.exists(PDF_Path))

missing_files <- sum(!un.labeled_sets_cln2$File_Exists, na.rm = TRUE)
cat("✅ File existence check complete. Missing files:", missing_files, "\n")

write.csv(un.labeled_sets_cln2, "Unlabeled & labeled citations w'out text extracted.csv", row.names = FALSE)

# ---------------------- STEP 3. Detect Language -----------------------------
un.labeled_sets_cln2 <- un.labeled_sets_cln2 %>%
  mutate(Language = detect_language(Full_Text)) %>%
  filter(Language == "en")


# ---------------------- STEP 4. Build Full_Text from cache ------------------

# Set up parallel plan for text extraction
plan(multisession, workers = parallel::detectCores() - 1)

# Cache folder setup
cache_folder <- "YOUR_FOLDER_DIRECTORY_HERE/cached_texts"

#if (!dir.exists(cache_folder)) dir.create(cache_folder)

# Logging list
log_list <- list()

#Method 1 CACHE Approach: Function to extract text with caching and OCR fallback
extract_pdf_text <- function(path) {
  tryCatch({
    if (is.na(path) || !file.exists(path)) {
      log_list[[path]] <<- list(Status = "Missing file", Message = NA_character_)
      return(NA_character_)
    }
    
    file_hash  <- digest::digest(path, algo = "sha256")
    cache_file <- file.path(cache_folder, paste0(file_hash, ".txt"))
    
    if (file.exists(cache_file)) {
      txt <- readr::read_file(cache_file)
      log_list[[path]] <<- list(Status = "Cached", Message = NA_character_)
      return(txt)
    }
    
    raw_txt <- pdftools::pdf_text(path)
    txt     <- paste(raw_txt, collapse = " ")
    
    if (nchar(txt) < 500) {
      message("⚠️ OCR fallback for: ", basename(path))
      ocr_txt <- tesseract::ocr(path)
      txt     <- paste(ocr_txt, collapse = " ")
      log_list[[path]] <<- list(Status = "OCR Fallback", Message = NA_character_)
    } else {
      log_list[[path]] <<- list(Status = "Extracted", Message = NA_character_)
    }
    
    readr::write_file(txt, cache_file)
    return(txt)
    
  }, error = function(e) {
    message("❌ Failed on: ", basename(path), " | Reason: ", e$message)
    log_list[[path]] <<- list(Status = "Failed", Message = e$message)
    return(NA_character_)
  })
}

##Method 2 Cache approach
# 1. Positional‐based fallback using pdf_data()
extract_via_pdf_data <- function(path) {
  pages_data <- pdftools::pdf_data(path)
  page_texts <- map_chr(pages_data, function(df) {
    # arrange by y (desc) then x to approximate reading order
    df_sorted <- df %>% arrange(desc(y), x)
    paste(df_sorted$text, collapse = " ")
  })
  paste(page_texts, collapse = " ")
}

# 2. Smart extractor
extract_pdf_text_smart <- function(path,
                                   cache_folder = "YOUR_FOLDER_DIRECTORY_HERE/cached_texts") {
  # compute cache filename
  file_hash  <- digest(path, algo = "sha256")
  cache_file <- file.path(cache_folder, paste0(file_hash, ".txt"))
  
  # return cached if present
  if (file.exists(cache_file)) {
    return(readr::read_file(cache_file))
  }
  
  # read raw PDF text by page
  pages <- pdftools::pdf_text(path)
  
  # for each page: if very short, OCR it; else keep pdf_text
  pages_clean <- map_chr(seq_along(pages), function(i) {
    pg <- pages[i]
    if (nchar(pg) < 200) {
      # render page to bitmap then OCR
      img <- pdftools::pdf_render_page(path, page = i, dpi = 300)
      return(tesseract::ocr(img))
    }
    return(pg)
  })
  
  # collapse into one string
  full_text <- paste(pages_clean, collapse = " ")
  
  # if still too short, do positional fallback
  if (nchar(full_text) < 500) {
    full_text <- extract_via_pdf_data(path)
  }
  
  # write to cache and return
  dir.create(cache_folder, showWarnings = FALSE, recursive = TRUE)
  readr::write_file(full_text, cache_file)
  full_text
}


# 3. Compute cache filenames and read only existing ones
cache_folder <- "YOUR_FOLDER_DIRECTORY_HERE/cached_texts"

text_tbl <- un.labeled_sets_cln2 %>%
  transmute(
    id         = `Covidence #`,
    cache_file = file.path(
      cache_folder,
      paste0(map_chr(PDF_Path, ~ digest(.x, "sha256")), ".txt"))
  ) %>%
  filter(file.exists(cache_file)) %>%
  mutate(
    Full_Text = map_chr(cache_file, ~ {
      # 1) slurp the whole file as one string
      txt <- read_file(.x)
      # 2) split on any newline / return / Unicode line‐sep
      parts <- str_split(txt, "\\r?\\n|\\u2028|\\u2029|\\u0085")[[1]]
      # 3) paste back together with single spaces
      collapsed <- paste(parts, collapse = " ")
      # 4) trim & collapse any runs of spaces
      str_squish(collapsed)
    })
  ) %>%
  distinct(id, .keep_all = TRUE)

# QUICK CHECK: absolutely no line-break codes remaining
stopifnot(!any(str_detect(text_tbl$Full_Text, "\\r|\\n")))

glimpse(text_tbl)

# Now `text_tbl` has one row for each citation ID, and a clean, single-string

# 4. Optionally report missing cache entries
missing_cache <- un.labeled_sets_cln2 %>%
  transmute(
    id         = `Covidence #`,
    file_hash  = map_chr(PDF_Path, ~ digest(.x, algo = "sha256")),
    cache_file = file.path(cache_folder, paste0(file_hash, ".txt"))
  ) %>%
  filter(!file.exists(cache_file))

cat("No cache for", nrow(missing_cache), "papers:\n")
print(missing_cache$id)

## Check for citations missing cache files
# A) Identify the rows in your master frame for those missing IDs
missing_tbl <- un.labeled_sets_cln2 %>%
  filter(`Covidence #` %in% missing_cache$id) %>%
  transmute(
    id       = `Covidence #`,
    PDF_Path
  )

# B) Re-extract (and re-cache) their text, using any extractor
recovered_texts <- missing_tbl %>%
  mutate(
    Full_Text = map_chr(
      PDF_Path,
      extract_pdf_text_smart,        # this writes to cache too
      cache_folder = cache_folder
    )
  )

# C) Append these back onto your existing text_tbl
text_tbl <- bind_rows(
  text_tbl,
  recovered_texts
)

#RERUN: 4. Optionally report missing cache entries

# ---------------------- STEP 5. Prepare Modeling data (pre-processing) ----------------------
##* Metadata: only ID + Label
names(un.labeled_sets_cln2)

meta_tbl <- un.labeled_sets_cln2 %>%
  transmute(
    Title = `norm_title`,
    id    = `Covidence #`,
    Label = factor(Label, levels = c("Excluded","Included","Unlabeled"))
  )

##* Join with text_tbl (id + Full_Text)
# Full_Text that you can immediately join back to your metadata:
modeling_tbl <- meta_tbl %>%
  inner_join(text_tbl, by = "id")

glimpse(modeling_tbl)

modeling_tbl <- modeling_tbl %>%
  select(-cache_file)

# Export extracted data
#library(readr)
#write_csv(text_tbl, "Unlabeled & labeled citations with text extracted.csv")

# ---------------------- STEP 6. KU signal (permissive) + trim Results/Refs ----
# Helpers
normalize_text <- function(x) {
  x %>%
    stringr::str_replace_all("\\r|\\n|\\u2028|\\u2029|\\u0085", " ") %>%
    stringr::str_replace_all("[\\u2010\\u2011\\u2012\\u2013\\u2014\\u2212]", "-") %>%
    stringr::str_squish() %>% stringr::str_to_lower()
}
cut_refs <- function(txt) {
  pat <- "(?i)\\b(references|bibliography|works\\s+cited|appendix|supplementary\\s+material)\\b"
  loc <- stringr::str_locate(txt, stringr::regex(pat))
  if (!is.na(loc[1,1])) substr(txt, 1, max(1L, loc[1,1] - 1L)) else txt
}
extract_acknowledgment_section <- function(text, num_chars = 2500) {
  pat <- "(?i)\\backnowledg(e)?ment(s)?\\b"
  loc <- stringr::str_locate(text, stringr::regex(pat))
  if (is.na(loc[1,1])) "" else substr(text, loc[1,1], min(nchar(text), loc[1,1] + num_chars))
}
cut_before_results <- function(text) {
  rx <- "(?im)^\\s*(results?|main\\s+results|key\\s+findings|findings|outcomes?)\\b"
  loc <- stringr::str_locate(text, stringr::regex(rx))
  if (!is.na(loc[1,1])) substr(text, 1, max(1L, loc[1,1] - 1L)) else text
}
rxify <- function(terms) stringr::regex(
  paste0("\\b(", paste(stringr::str_replace_all(terms, "\\s+", "\\\\s+"), collapse="|"), ")\\b"),
  ignore_case = TRUE
)

# Lexicons
ku_phrases <- c(
  "participatory modeling","participatory modelling","participatory simulation modeling",
  "participatory dynamic modeling","group model building","group-based model building",
  "stakeholder-driven modeling","stakeholder-engaged modeling","stakeholder-informed modeling",
  "community-based modeling","co-developed model","co-created model",
  "stakeholder engagement","stakeholder consultation","stakeholder workshops","expert elicitation",
  "community stakeholder engagement","policy-maker engagement","end-user engagement",
  "local expert consultation","in-country consultation","in-country modeling team",
  "decision-maker input","community partner input","provided data","commissioned",
  "co-production of knowledge","co-produced knowledge","co-design process","co-interpretation of results",
  "consensus-building workshop","workshop","consultation","model validation workshop","data validation meeting",
  "scenario co-development","participatory scenario analysis","participatory policy analysis",
  "collaborative interpretation of model outputs","shared decision-making process",
  "collaborative modeling","international participatory approach","participatory approach"
)
affil_terms <- c("ministry of health","department of health","public health","community","advisory",
                 "ngo","civil society","community partner","government")

# Regexes
rx_ku    <- rxify(ku_phrases)
rx_affil <- rxify(affil_terms)
ku_alt   <- paste(stringr::str_replace_all(ku_phrases, "\\s+", "\\\\s+"), collapse="|")
rx_act   <- "(we\\s+(conduct|held|facilitat|organiz|organis|ran)|conducted|held|facilitated|organized|ran|session[s]?|workshop[s]?|stakeholder[s]?|participant[s]?|consult|engag|co-?design|co-?develop)"
rx_prox  <- stringr::regex(
  paste0("(", ku_alt, ").{0,100}(", rx_act, ")|(", rx_act, ").{0,100}(", ku_alt, ")"),
  ignore_case = TRUE
)
rx_ack_particip <- stringr::regex(
  "\\b(provided\\s+data|data\\s+collection|data\\s+sharing|stakeholder\\s+(input|engagement|consultation)|workshop[s]?|session[s]?|co-?design(ed)?|co-?develop(ed)?|advis(or|ory)|participants?)\\b",
  ignore_case = TRUE
)

# Build signals on trimmed body; feed trimmed text forward as Full_Text
modeling_tbl <- modeling_tbl %>%
  mutate(
    txt_l   = normalize_text(coalesce(Full_Text, "")),
    body0   = purrr::map_chr(txt_l, cut_refs),
    body    = purrr::map_chr(body0, cut_before_results),
    
    abstract_win = stringr::str_sub(body, 1, 2500),
    methods_win  = { n <- nchar(body); ifelse(n > 1000, substr(body, floor(0.40*n), floor(0.80*n)), body) },
    ack          = purrr::map_chr(body0, extract_acknowledgment_section, num_chars = 2500),
    
    hit_front = stringr::str_detect(abstract_win, rx_ku),
    hit_meth  = stringr::str_detect(methods_win,  rx_ku),
    hit_ack   = stringr::str_detect(ack, rx_ku) | stringr::str_detect(ack, rx_ack_particip),
    hit_affil = stringr::str_detect(stringr::str_sub(body, 1, 1500), rx_affil),
    hit_prox  = stringr::str_detect(body, rx_prox),
    
    ku_signal = as.integer(hit_front | hit_meth | hit_ack | hit_affil | hit_prox),
    ku_score  = as.integer(hit_front) + as.integer(hit_meth) +
      as.integer(hit_ack)   + as.integer(hit_affil) + as.integer(hit_prox)
  ) %>%
  select(id, Title, Label, Full_Text = body, ku_signal, ku_score)

# ---------------------- STEP 7. Relabel pre-found papers (used in validating search strategy) -----------
fp_gold <- "YOUR_FOLDER_DIRECTORY_HERE/Pre-foundPapers.csv"
if (file.exists(fp_gold)) {
  gold_set <- readr::read_csv(fp_gold, show_col_types = FALSE) %>%
    transmute(Title = File %>% stringr::str_remove("\\.pdf$") %>%
                stringr::str_to_lower() %>% stringr::str_replace_all("[[:punct:][:space:]]+", ""))
  corpus_titles <- modeling_tbl %>%
    mutate(Title = Title %>% stringr::str_to_lower() %>% stringr::str_replace_all("[[:punct:][:space:]]+", "")) %>%
    select(Title, id, Label)
  overlaps <- dplyr::inner_join(corpus_titles, gold_set, by = "Title")
  message(sprintf("Found %d overlaps between gold set and corpus.", nrow(overlaps)))
  if (nrow(overlaps)) print(overlaps, n = min(20, nrow(overlaps)))
  modeling_tbl_cln <- modeling_tbl %>%
    mutate(Label = ifelse(id %in% overlaps$id & Label == "Unlabeled", "Included", as.character(Label)))
  readr::write_csv(overlaps, "gold_corpus_overlaps_present.csv")
} else {
  modeling_tbl_cln <- modeling_tbl
}
cat("\nRows in modeling_tbl_cln:", nrow(modeling_tbl_cln), "\n")
Hmisc::describe(modeling_tbl_cln$Label)

Hmisc::describe(modeling_tbl$Label)

# ---------------------- STEP 8. Split labeled/unlabeled (70/30, stratified) --
suppressPackageStartupMessages({
  library(text2vec); library(glmnet); library(yardstick); library(rsample)
  library(Matrix); library(dplyr); library(tidyr); library(purrr); library(tibble)
})
set.seed(416)

labeled <- modeling_tbl_cln %>%
  filter(Label %in% c("Excluded","Included")) %>%
  mutate(Label = factor(Label, levels = c("Excluded","Included"))) %>%
  distinct(id, .keep_all = TRUE)

to_predict <- modeling_tbl_cln %>%
  filter(Label == "Unlabeled") %>%
  distinct(id, .keep_all = TRUE)

split_obj  <- rsample::initial_split(labeled, prop = 0.70, strata = Label)
train_data <- rsample::training(split_obj)
test_data  <- rsample::testing(split_obj)

train_data <- train_data %>%
  mutate(Label = forcats::fct_relevel(Label, "Included","Excluded"),
         ku_signal = as.numeric(coalesce(ku_signal, 0)),
         ku_score  = as.numeric(coalesce(ku_score,  0)))
test_data <- test_data %>%
  mutate(Label = forcats::fct_relevel(Label, "Included","Excluded"),
         ku_signal = as.numeric(coalesce(ku_signal, 0)),
         ku_score  = as.numeric(coalesce(ku_score,  0)))
to_predict <- to_predict %>%
  mutate(ku_signal = as.numeric(coalesce(ku_signal, 0)),
         ku_score  = as.numeric(coalesce(ku_score,  0)))

y_train <- ifelse(train_data$Label == "Included", 1L, 0L)
y_test  <- ifelse(test_data$Label  == "Included", 1L, 0L)

# ---------------------- STEP 9. Text vectors (TF–IDF, train-fit only) -------
tok_fun <- function(x) {
  x <- tolower(gsub("[\r\n\u2028\u2029\u0085]", " ", x))
  x <- gsub("([[:alnum:]])-([[:alnum:]])", "\\1_\\2", x)
  text2vec::word_tokenizer(x)
}
it_train <- itoken(train_data$Full_Text, tokenizer = tok_fun, progressbar = FALSE)
vocab <- create_vocabulary(it_train, ngram = c(1L, 2L)) %>%
  prune_vocabulary(term_count_min = 2, vocab_term_max = 15000)
vectorizer <- vocab_vectorizer(vocab)
dtm_train  <- create_dtm(it_train, vectorizer)
tfidf      <- TfIdf$new()
X_train_tf <- tfidf$fit_transform(dtm_train)

it_test <- itoken(test_data$Full_Text, tokenizer = tok_fun, progressbar = FALSE)
X_test_tf <- tfidf$transform(create_dtm(it_test, vectorizer))
it_ul <- itoken(to_predict$Full_Text, tokenizer = tok_fun, progressbar = FALSE)
X_ul_tf <- tfidf$transform(create_dtm(it_ul, vectorizer))

add_num_feat <- function(X, v, name) {
  Matrix::cbind2(X, Matrix::Matrix(as.numeric(v), ncol = 1, sparse = TRUE, dimnames = list(NULL, name)))
}
X_train <- X_train_tf %>% add_num_feat(train_data$ku_signal, "ku_signal") %>% add_num_feat(train_data$ku_score, "ku_score")
X_test  <- X_test_tf  %>% add_num_feat(test_data$ku_signal,  "ku_signal") %>% add_num_feat(test_data$ku_score,  "ku_score")
X_ul    <- X_ul_tf    %>% add_num_feat(to_predict$ku_signal, "ku_signal") %>% add_num_feat(to_predict$ku_score, "ku_score")


# ---------------------- STEP 10. Ridge logistic (Repeated CV; tune penalty & τ) ------

wts <- ifelse(y_train == 1L,
              sum(y_train == 0L) / length(y_train),
              sum(y_train == 1L) / length(y_train))

##* 1. Repeated CV for lambda selection (5×5)
set.seed(416)
n_repeats <- 5
v_folds   <- 5

lambda_1se_vec <- numeric(n_repeats)

for (r in seq_len(n_repeats)) {
  set.seed(416 + r)
  foldid <- sample(rep(1:v_folds, length.out = nrow(X_train)))
  cvfit_r <- cv.glmnet(
    x = X_train, y = y_train, family = "binomial",
    alpha = 0, weights = wts, type.measure = "deviance",
    foldid = foldid, parallel = FALSE
  )
  lambda_1se_vec[r] <- cvfit_r$lambda.1se
}
lambda_star <- median(lambda_1se_vec)
cat(sprintf("[Repeated cv.glmnet] median λ.1se over %d repeats = %.6f\n",
            n_repeats, lambda_star))

##* 2. Repeated 5-fold OOF probabilities (for diagnostics only) 
set.seed(416)

oof_prob_mat <- matrix(NA_real_, nrow = nrow(X_train), ncol = n_repeats)

# Build a data frame that contains the indices and the label column for stratification
cv_df <- tibble(
  idx = 1:nrow(X_train),
  y   = factor(y_train, levels = c(0L, 1L))  # ensure factor; 1L = Included
)

for (r in seq_len(n_repeats)) {
  set.seed(900 + r)
  folds_r <- rsample::vfold_cv(cv_df, v = v_folds, strata = y)
  oof_prob_r <- rep(NA_real_, length(y_train))
  
  for (i in seq_len(nrow(folds_r))) {
    tr <- rsample::analysis(folds_r$splits[[i]])$idx
    va <- rsample::assessment(folds_r$splits[[i]])$idx
    
    fit_i <- glmnet(
      X_train[tr, ], y_train[tr],
      family = "binomial", alpha = 0,
      lambda = lambda_star, weights = wts[tr], standardize = TRUE
    )
    oof_prob_r[va] <- as.numeric(predict(fit_i, X_train[va, ], type = "response"))
  }
  
  stopifnot(!any(is.na(oof_prob_r)))
  oof_prob_mat[, r] <- oof_prob_r
}

# Average OOF probabilities across repeats (you can inspect these vs. labels)
oof_prob <- rowMeans(oof_prob_mat)

##* 3. (C) Final model fit (no τ tuning here) 
final_glmnet <- glmnet(x = X_train, y = y_train, family = "binomial",
                       alpha = 0, lambda = lambda_star, weights = wts)

# Save the fitted model object for reuse
saveRDS(final_glmnet, file = "reglr_glmnet_model.rds")

##* 4. Test-set scoring + manual τ 
p_test <- as.numeric(predict(final_glmnet, newx = X_test, type = "response"))

# Set your own threshold here (manually adjust as you like)
tau_user <- 0.466

test_eval <- tibble(
  id    = test_data$id,
  Label = factor(ifelse(y_test == 1L, "Included","Excluded"),
                 levels = c("Included","Excluded")),
  .pred_Included = p_test
) %>%
  mutate(.pred_class = factor(ifelse(.pred_Included >= tau_user, "Included","Excluded"),
                              levels = c("Included","Excluded")))

cat(sprintf("\n=== TEST @ τ=%.2f (manual) ===\n", tau_user))
print(metric_set(accuracy, recall, specificity, precision, f_meas)(test_eval, truth = Label, estimate = .pred_class))
print(roc_auc(test_eval, truth = Label, .pred_Included))
print(pr_auc( test_eval, truth = Label, .pred_Included))
print(conf_mat(test_eval, truth = Label, estimate = .pred_class))


# ---------------------- STEP 11. 95% CIs on TEST set ----------------------
# (Requires pROC for DeLong AUC CI)
if (!requireNamespace("pROC", quietly = TRUE)) {
  install.packages("pROC")
}
library(pROC)

# Point estimates at τ* on the held-out test set
acc  <- yardstick::accuracy_vec(truth = test_eval$Label, estimate = test_eval$.pred_class)
sens <- yardstick::recall_vec(  truth = test_eval$Label, estimate = test_eval$.pred_class,
                                event_level = "first")      # "Included" is the 1st level
spec <- yardstick::spec_vec(    truth = test_eval$Label, estimate = test_eval$.pred_class,
                                event_level = "first")

# AUC (point) and DeLong 95% CI
truth_bin <- as.numeric(test_eval$Label == "Included")
roc_obj   <- pROC::roc(response = truth_bin, predictor = test_eval$.pred_Included, quiet = TRUE)
auc_pt    <- as.numeric(pROC::auc(roc_obj))
auc_ci    <- pROC::ci.auc(roc_obj)  # 2.5%, 50%, 97.5%

# --- Bootstrap CIs for Accuracy / Sensitivity / Specificity at τ* ---
set.seed(416)
B <- 1000
n <- nrow(test_eval)
prob <- test_eval$.pred_Included
truth <- test_eval$Label  # factor levels: c("Included","Excluded"); ensure order below

# Ensure consistent level order for yardstick ("Included" is positive, set as first level)
truth <- factor(truth, levels = c("Included", "Excluded"))

boot_mat <- replicate(B, {
  idx <- sample.int(n, n, replace = TRUE)
  prob_b  <- prob[idx]
  truth_b <- truth[idx]
  pred_b  <- factor(ifelse(prob_b >= tau_user, "Included", "Excluded"),
                    levels = c("Included", "Excluded"))
  c(
    acc  = yardstick::accuracy_vec(truth = truth_b, estimate = pred_b),
    sens = yardstick::recall_vec(  truth = truth_b, estimate = pred_b, event_level = "first"),
    spec = yardstick::spec_vec(    truth = truth_b, estimate = pred_b, event_level = "first")
  )
})

ci_acc  <- stats::quantile(boot_mat["acc", ],  probs = c(0.025, 0.975), na.rm = TRUE)
ci_sens <- stats::quantile(boot_mat["sens", ], probs = c(0.025, 0.975), na.rm = TRUE)
ci_spec <- stats::quantile(boot_mat["spec", ], probs = c(0.025, 0.975), na.rm = TRUE)

# --- Neat printout ---
cat("\nPoint estimates (TEST @ τ*):\n")
cat(sprintf("Accuracy   = %.3f\nSensitivity= %.3f\nSpecificity= %.3f\nAUC        = %.3f\n",
            acc, sens, spec, auc_pt))

cat("\n95% Bootstrap CIs (TEST @ τ*):\n")
cat(sprintf("Accuracy    95%% CI: [%.3f, %.3f]\n", ci_acc[1],  ci_acc[2]))
cat(sprintf("Sensitivity 95%% CI: [%.3f, %.3f]\n", ci_sens[1], ci_sens[2]))
cat(sprintf("Specificity 95%% CI: [%.3f, %.3f]\n", ci_spec[1], ci_spec[2]))

cat("\nAUC 95% CI (DeLong):\n")
cat(sprintf("[%.3f, %.3f]\n", as.numeric(auc_ci[1]), as.numeric(auc_ci[3])))



# ---------------------- STEP 12. Predict unlabeled & export ---------
p_ul   <- as.numeric(predict(final_glmnet, newx = X_ul, type = "response"))
ul_pred <- factor(ifelse(p_ul >= 0.466, "Included","Excluded"),
                  levels = c("Included","Excluded"))

ul_out <- tibble(
  id              = to_predict$id,
  Title           = to_predict$Title,
  .pred_Included  = p_ul,
  Label           = ul_pred,
  tau_used        = 0.466,
  lambda_used     = as.numeric(lambda_star)
) %>% arrange(desc(.pred_Included))  # prioritize highest probabilities first

# Export
readr::write_csv(ul_out, "unlabeled_set_predictions.csv")
cat("\nWrote predictions for unlabeled to 'unlabeled_predictions.csv' (sorted by probability desc).\n")


