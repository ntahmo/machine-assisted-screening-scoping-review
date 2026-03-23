################################################################################
### Title:            Participatory modeling scoping review - machine learning
### Author:           Nancy Tahmo, PhD student
### Creation date:    March 17, 2025
### Description:      Full-text algorithmic screening
################################################################################

# ---------------------- Load Libraries --------------------------------------

library(pdftools)      # Extract text from PDFs
library(tesseract)     # OCR for image-based PDFs
library(tidyverse)     # Data manipulation and I/O
library(future.apply)  # Parallel processing
library(furrr)         # Parallel mapping with tidy output

# Set the folder containing the PDFs
pdf_folder <- "YOUR_FOLDER_DIRECTORY_HERE/All_PDF_Files"
pdf_files <- list.files(pdf_folder, pattern = "\\.pdf$", full.names = TRUE)

# Setup parallel plan (adjust workers if memory permits)
plan(multisession, workers = 1)

# ------- STEP 1. Define words and phrases for screening -------- 
# Pre-compile regex patterns for efficiency
infectiousdisease_regex <- paste(c("HIV", "AIDS", "tuberculosis", "TB", "hepatitis A", "hepatitis B", "hepatitis C",
                           "influenza", "COVID-19", "SARS-CoV-2", "Ebola", "dengue", "Zika", "chikungunya",
                           "measles", "mumps", "rubella", "varicella", "chickenpox", "herpes", "HPV",
                           "cytomegalovirus", "RSV", "hantavirus", "Lassa fever", "West Nile virus",
                           "rabies", "polio", "smallpox", "yellow fever", "Japanese encephalitis",
                           "gonorrhea", "syphilis", "chlamydia", "bacterial vaginosis", "leptospirosis",
                           "pertussis", "diphtheria", "tetanus", "Legionnaires' disease", "cholera",
                           "typhoid fever", "malaria", "toxoplasmosis", "leishmaniasis", "Chagas disease",
                           "schistosomiasis", "giardiasis", "cryptosporidiosis", "candidiasis",
                           "aspergillosis", "cryptococcosis", "Pneumocystis pneumonia", "prion disease",
                           "leprosy", "Creutzfeldt-Jakob disease"), collapse = "|")

modeling_regex <- paste(c("model\\w*", "simulation", "theoretical model\\w*", "epidemiological model\\w*",
                          "patient-specific model\\w*", "mathematical model\\w*", "stochastic model\\w*",
                          "outbreak model\\w*", "compartmental model\\w*", "deterministic model\\w*",
                          "individual-based model\\w*", "agent-based model\\w*", "network model\\w*",
                          "dynamic model\\w*", "Markov model\\w*", "SIR model\\w*", "SEIR model\\w*"), collapse = "|")

community_regex <- paste(c("community-based", "participatory", "collaborative", "stakeholder", 
                           "policymaker\\w*", "decision maker\\w*", "knowledge user\\w*",
                           "lived experience\\w*", "human\\w* judgment",  
                           "co-design", "citizen science", "group model building", 
                           "participatory system dynamics", "advisory board\\w*"), collapse = "|")

supplemental_regex <- paste(c("supplementary", "supplemental", "supporting information",
                              "appendix", "appendices", "online supplement", "additional file"), collapse = "|")

# ------- STEP 2. Set up function to process every single PDF ------
process_pdf <- function(pdf) {
  tryCatch({
    message("Processing: ", basename(pdf))
    # Extract text from the PDF
    pdf_text_raw <- pdf_text(pdf)
    pdf_text <- paste(pdf_text_raw, collapse = " ")
    
    # Apply OCR if text is too short (likely an image-based PDF)
    if (nchar(pdf_text) < 500) {
      message("  [OCR] Applying OCR to ", basename(pdf))
      ocr_text <- ocr(pdf)
      pdf_text <- paste(ocr_text, collapse = " ")
    }
    
    # Check if both ID and Modeling terms are present using the precompiled regex
    if (!grepl(id_regex, pdf_text, ignore.case = TRUE) ||
        !grepl(modeling_regex, pdf_text, ignore.case = TRUE)) {
      return(NULL)
    }
    
    # Check for community and supplemental indicators
    has_community <- grepl(community_regex, pdf_text, ignore.case = TRUE)
    has_supplemental <- grepl(supplemental_regex, pdf_text, ignore.case = TRUE)
    
    # Final classification
    final_inclusion <- if (has_community) {
      "Included (Community)"
    } else if (has_supplemental) {
      "Other (Supplemental)"
    } else {
      "Excluded"
    }
    
    tibble(File = basename(pdf),
           ID_Modeling_Match = TRUE,
           Community_Match = has_community,
           Supplemental_Match = has_supplemental,
           Final_Inclusion = final_inclusion)
    
  }, error = function(e) {
    message("Error processing ", basename(pdf), ": ", e$message)
    return(NULL)
  })
}

##* Process PDFs in batches to help manage memory and get progress feedback
batch_size <- 10  # Adjust this number based on your system's memory and performance
pdf_batches <- split(pdf_files, ceiling(seq_along(pdf_files) / batch_size))

results_list <- list()

for (i in seq_along(pdf_batches)) {
  message("Processing batch ", i, " of ", length(pdf_batches))
  batch_results <- future_map_dfr(pdf_batches[[i]], process_pdf, .progress = TRUE)
  results_list[[i]] <- batch_results
  # Clean up memory after each batch
  gc()
}

# Combine all batch results into one dataframe
results <- bind_rows(results_list)

# ------- STEP 3. Save results to CSV ------
output_path <- "Fulltext_algorithmicscreening_results"
write_csv(results, output_path)

cat("✅ Full-text review complete. Results saved as '", output_path, "'.\n", sep = "")
cat("Total PDFs processed: ", length(pdf_files), "\n")
cat("Total citations found: ", nrow(results), "\n")
