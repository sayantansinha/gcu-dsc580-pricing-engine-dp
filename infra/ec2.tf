# AMI: Amazon Linux 2023 (x86_64)
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

# Default VPC + subnets (AWS provider v5 style)
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Security group (open 8501 for Streamlit later)
resource "aws_security_group" "ppe_web" {
  name        = "${local.app_prefix}-web-sg"
  description = "Allow Streamlit 8501"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "Streamlit"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # tighten later
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

# Free-Tier EC2
resource "aws_instance" "ppe_ec2" {
  ami                         = data.aws_ami.al2023.id
  instance_type               = "t2.micro"
  subnet_id                   = data.aws_subnets.default.ids[0]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  vpc_security_group_ids      = [aws_security_group.ppe_web.id]

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
  })
}

output "ec2_public_ip" {
  value = aws_instance.ppe_ec2.public_ip
}

output "app_url" {
  value = "http://${aws_instance.ppe_ec2.public_ip}:8501"
}
