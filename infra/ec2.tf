resource "aws_instance" "ppe_ec2" {
  ami                         = local.ppe_ami
  instance_type               = "t2.micro"
  subnet_id                   = data.aws_subnets.default.ids[0]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  vpc_security_group_ids = [aws_security_group.ppe_web_sg.id]

  tags = merge(local.tags, {
    "CodeDeploy:App"   = local.app_prefix
    "CodeDeploy:Group" = "${local.app_prefix}-blue"
    Name               = "${local.app_prefix}-ec2"
  })
}

resource "aws_ssm_association" "ppe_cloudwatch_setup" {
  name = "AWS-RunShellScript"

  targets {
    key = "InstanceIds"
    values = [aws_instance.ppe_ec2.id]
  }

  parameters = {
    commands = <<-EOC
          #!/bin/bash
          set -x

          # Install SSM + CW agent
          dnf -y update || true
          dnf -y install python3.11 python3.11-pip amazon-ssm-agent amazon-cloudwatch-agent || true
          systemctl enable --now amazon-ssm-agent || true

          # Set timezone
          timedatectl set-timezone America/Los_Angeles || true

          # Write CloudWatch Agent config
          cat << 'EOF' > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
          {
            "logs": {
              "logs_collected": {
                "files": {
                  "collect_list": [
                    {
                      "file_path": "/var/log/ppe-app/ppe-app.log",
                      "log_group_name": "ppe-app-lg",
                      "log_stream_name": "{instance_id}",
                      "timezone": "LOCAL"
                    }
                  ]
                }
              }
            }
          }
          EOF

          # Restart CW Agent
          /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a stop || true
          /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a start -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json || true

      EOC

  }
}

output "ec2_public_ip" {
  value = aws_instance.ppe_ec2.public_ip
}
