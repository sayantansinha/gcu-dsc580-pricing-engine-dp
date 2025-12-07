#############################################
# identity
#############################################
data "aws_caller_identity" "current" {}

#############################################
# derived locals
#############################################
locals {
  buckets_all = [
    local.raw_bucket,
    local.processed_bucket,
    local.profiles_bucket,
    local.models_bucket,
    local.figures_bucket,
    local.reports_bucket,
    local.deploy_artifact_bucket
  ]

  tags = {
    Project = local.app_prefix
    Owner   = data.aws_caller_identity.current.account_id
  }
}
