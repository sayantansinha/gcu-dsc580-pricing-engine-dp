# Application Load Balancer
resource "aws_lb" "ppe_alb" {
  name               = "${local.app_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups = [aws_security_group.ppe_alb_sg.id]

  # Public subnets for the ALB
  subnets = data.aws_subnets.default.ids

  tags = {
    Name = "${local.app_prefix}-alb"
  }
}

# Target Group (PPE app on port 8501)
resource "aws_lb_target_group" "ppe_tg" {
  name        = "${local.app_prefix}-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "instance"

  health_check {
    protocol            = "HTTP"
    port                = "8501"
    path = "/"          # change to /health later if you add a health endpoint
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    matcher             = "200-399"
  }

  tags = {
    Name = "${local.app_prefix}-tg"
  }
}

# Attach EC2 instance to Target Group
resource "aws_lb_target_group_attachment" "ppe_tg_attachment" {
  target_group_arn = aws_lb_target_group.ppe_tg.arn
  target_id        = aws_instance.ppe_ec2.id
  port             = 8501
}

# HTTP Listener (port 80)
resource "aws_lb_listener" "ppe_http" {
  load_balancer_arn = aws_lb.ppe_alb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "forward"
    target_group_arn = aws_lb_target_group.ppe_tg.arn
  }
}

# HTTPS Listener (port 443)
# resource "aws_lb_listener" "ppe_https" {
#   load_balancer_arn = aws_lb.ppe_alb.arn
#   port              = 443
#   protocol          = "HTTPS"
#   ssl_policy        = "ELBSecurityPolicy-2016-08"
#   certificate_arn   = aws_acm_certificate_validation.ppe_cert_validation.certificate_arn
#
#   default_action {
#     type             = "forward"
#     target_group_arn = aws_lb_target_group.ppe_tg.arn
#   }
# }

output "ppe_alb_dns_name" {
  description = "Public DNS name of the PPE ALB"
  value       = aws_lb.ppe_alb.dns_name
}