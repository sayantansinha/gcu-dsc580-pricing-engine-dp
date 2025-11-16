resource "aws_instance" "ppe_ec2" {
  ami                         = local.ppe_ami
  instance_type               = "t2.micro"
  subnet_id                   = data.aws_subnets.default.ids[0]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  vpc_security_group_ids = [aws_security_group.ppe_web_sg.id]

  # Minimal bootstrap only; app deploy comes later via CI/CD or SSM
  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail
    dnf -y update || true
    dnf -y install python3.11 python3.11-pip amazon-ssm-agent
    systemctl enable --now amazon-ssm-agent
  EOF

  tags = merge(local.tags, {
    "CodeDeploy:App"   = local.app_prefix
    "CodeDeploy:Group" = "${local.app_prefix}-blue"
    Name               = "${local.app_prefix}-ec2"
  })
}

output "ec2_public_ip" {
  value = aws_instance.ppe_ec2.public_ip
}
