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

    # Basic updates and SSM agent
    dnf -y update || true
    dnf -y install python3.11 python3.11-pip amazon-ssm-agent amazon-cloudwatch-agent
    systemctl enable --now amazon-ssm-agent

    # Write CloudWatch Agent config
    mkdir -p /opt/aws/amazon-cloudwatch-agent/etc
    cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'CONFIGEOF'
    ${local.ppe_cloudwatch_agent_config}
    CONFIGEOF

    # Start CloudWatch Agent
    /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a stop || true
    /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
      -a start \
      -m ec2 \
      -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
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
