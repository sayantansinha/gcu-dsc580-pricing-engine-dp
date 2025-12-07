resource "aws_instance" "ppe_ec2" {
  ami                         = local.ppe_ami
  instance_type               = "t3.small"
  subnet_id                   = data.aws_subnets.default.ids[0]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  vpc_security_group_ids      = [aws_security_group.ppe_web_sg.id]

  tags = merge(local.tags, {
    "CodeDeploy:App"   = local.app_prefix
    "CodeDeploy:Group" = "${local.app_prefix}-blue"
    Name               = "${local.app_prefix}-ec2"
  })
}
