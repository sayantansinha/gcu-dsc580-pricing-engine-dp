locals {
  # AWS region
  aws_region = "us-west-2"

  # Project naming prefix
  app_prefix = "ppe"

  # Buckets (Free-Tier eligible)
  raw_bucket       = "${local.app_prefix}-poc-raw"
  processed_bucket = "${local.app_prefix}-poc-processed"
  profiles_bucket  = "${local.app_prefix}-poc-profiles"
  models_bucket    = "${local.app_prefix}-poc-models"
  figures_bucket   = "${local.app_prefix}-poc-figures"
  reports_bucket   = "${local.app_prefix}-poc-reports"

}
