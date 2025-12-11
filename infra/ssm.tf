resource "aws_ssm_parameter" "ppe_admin_password" {
  name  = "${local.app_prefix}_admin_password"
  type  = "SecureString"
  value = var.ppe_admin_password
}
