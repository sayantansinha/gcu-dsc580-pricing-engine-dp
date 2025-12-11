locals {
  # AWS region
  aws_region = "us-west-2"

  # Project naming prefix
  app_prefix = "ppe"

  # AMI
  ppe_ami = "ami-0ebda4230423eb2b4"

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

  # CloudWatch agent configuration (metrics + logs)
  cwagent_config = {
    metrics = {
      metrics_collected = {
        mem = {
          measurement                 = ["mem_used_percent"]
          metrics_collection_interval = 60
        }
        swap = {
          measurement                 = ["swap_used_percent"]
          metrics_collection_interval = 60
        }
      }
      append_dimensions = {
        InstanceId = "$${aws:InstanceId}"
      }
    }

    logs = {
      logs_collected = {
        files = {
          collect_list = [
            {
              file_path       = "/var/log/app/app.log"
              log_group_name  = aws_cloudwatch_log_group.ppe-app-lg.name
              log_stream_name = "{instance_id}"
              timezone        = "LOCAL"
            }
          ]
        }
      }
    }
  }

  # Base64 so we can write the JSON in one clean shell command
  cwagent_config_b64 = base64encode(jsonencode(local.cwagent_config))
}

