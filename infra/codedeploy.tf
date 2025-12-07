# App & Deployment Group
resource "aws_codedeploy_app" "ppe" {
  name             = local.app_prefix
  compute_platform = "Server"
  tags             = local.tags
}

# Service role for CodeDeploy (managed policy)
resource "aws_iam_role" "codedeploy_service" {
  name = "${local.app_prefix}-codedeploy-svc"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect    = "Allow",
        Principal = { Service = "codedeploy.amazonaws.com" },
        Action    = "sts:AssumeRole"
      }
    ]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "codedeploy_managed" {
  role       = aws_iam_role.codedeploy_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSCodeDeployRole"
}

resource "aws_codedeploy_deployment_group" "ppe_blue" {
  app_name              = aws_codedeploy_app.ppe.name
  deployment_group_name = "${local.app_prefix}-blue"
  service_role_arn      = aws_iam_role.codedeploy_service.arn

  # target EC2 instances by tags
  ec2_tag_set {
    ec2_tag_filter {
      key   = "CodeDeploy:App"
      type  = "KEY_AND_VALUE"
      value = local.app_prefix
    }

    ec2_tag_filter {
      key   = "CodeDeploy:Group"
      type  = "KEY_AND_VALUE"
      value = "${local.app_prefix}-blue"
    }
  }

  deployment_style {
    deployment_option = "WITHOUT_TRAFFIC_CONTROL"
    deployment_type   = "IN_PLACE"
  }

  auto_rollback_configuration {
    enabled = true
    events  = ["DEPLOYMENT_FAILURE"]
  }

  tags = local.tags
}
