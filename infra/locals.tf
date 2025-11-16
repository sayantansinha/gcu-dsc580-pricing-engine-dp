locals {
  # AWS region
  aws_region = "us-west-2"

  # Project naming prefix
  app_prefix = "ppe"

  # AMI
  ppe_ami = "ami-00dd21db9cf92ef84"

  # # Domain
  # root_domain = "mipuba.com"
  # ppe_domain = "ppe.${local.root_domain}"

  # Buckets (Free-Tier eligible)
  raw_bucket             = "${local.app_prefix}-poc-raw"
  processed_bucket       = "${local.app_prefix}-poc-processed"
  profiles_bucket        = "${local.app_prefix}-poc-profiles"
  models_bucket          = "${local.app_prefix}-poc-models"
  figures_bucket         = "${local.app_prefix}-poc-figures"
  reports_bucket         = "${local.app_prefix}-poc-reports"
  deploy_artifact_bucket = "${local.app_prefix}-poc-deploy-artifacts"

  # CloudWatch Agent
  ppe_cloudwatch_agent_config = jsonencode({
    logs = {
      logs_collected = {
        files = {
          collect_list = [
            {
              file_path       = "/var/log/ppe-app/ppe-app.log"
              log_group_name  = aws_cloudwatch_log_group.ppe-app-lg.name
              log_stream_name = "{instance_id}"
              timezone        = "UTC"
            }
          ]
        }
      }
    }
  })
}
