# Allow CloudWatch to run a one-time install command after apply
resource "aws_ssm_document" "install_cloudwatch_agent" {
  name          = "${local.app_prefix}-install-cwagent"
  document_type = "Command"

  content = jsonencode({
    schemaVersion = "2.2",
    description   = "Install and configure CloudWatch Agent on AL2023",
    mainSteps = [
      {
        action = "aws:runShellScript",
        name   = "installAndConfigure",
        inputs = {
          runCommand = [
            # Install the CloudWatch agent (AL2023 uses dnf, fallback to yum)
            "dnf -y install amazon-cloudwatch-agent || yum -y install amazon-cloudwatch-agent",

            "mkdir -p /opt/aws/amazon-cloudwatch-agent/etc",

            # Minimal metrics config: memory + swap usage
            "cat >/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<'CONFIG'",
            "{",
            "  \"metrics\": {",
            "    \"metrics_collected\": {",
            "      \"mem\": {",
            "        \"measurement\": [\"mem_used_percent\"],",
            "        \"metrics_collection_interval\": 60",
            "      },",
            "      \"swap\": {",
            "        \"measurement\": [\"swap_used_percent\"],",
            "        \"metrics_collection_interval\": 60",
            "      }",
            "    },",
            "    \"append_dimensions\": {",
            "      \"InstanceId\": \"$${aws:InstanceId}\"",
            "    }",
            "  },",
            " \"logs\": {",
            "    \"logs_collected\": {",
            "      \"files\": {",
            "        \"collect_list\": [",
            "            {",
            "              \"file_path\": \"/var/log/ppe-app/ppe-app.log\",",
            "              \"log_group_name\": \"ppe-app-lg\",",
            "              \"log_stream_name\": \"{instance_id}\",",
            "              \"timezone\": \"LOCAL\"",
            "            }",
            "         ]",
            "       }",
            "     }",
            "  }",
            "}",
            "CONFIG",

            # Start CloudWatch agent with that config
            "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl ",
            "-a fetch-config -m ec2 ",
            "-c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json ",
            "-s"
          ]
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_ssm_association" "install_cloudwatch_agent_to_instance" {
  name = aws_ssm_document.install_cloudwatch_agent.name

  targets {
    key    = "InstanceIds"
    values = [aws_instance.ppe_ec2.id]
  }
}

