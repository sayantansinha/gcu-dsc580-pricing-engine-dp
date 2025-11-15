resource "aws_cloudwatch_log_group" "app" {
  name              = "/${local.app_prefix}/app"
  retention_in_days = 7
  tags              = local.tags
}
