#header ----
rm(list=ls())
gc()

library(tidyverse)
library(readxl)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# analysis of bills
test_claude <- read_csv("../data/processed/ncsl_2025_test.csv")
old_data <- read_excel('/Users/ianmayhood/Desktop/State Legislation Taxonomy/State Legislation Taxonomy.xlsx', sheet = 11)
