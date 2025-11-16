# # Public Hosted Zone for the root domain
# resource "aws_route53_zone" "root_zone" {
#   name = local.root_domain
#
#   # This creates a public hosted zone
#   comment = "Public hosted zone for ${local.root_domain}"
#
#   tags = {
#     Name = "${local.app_prefix}-root-zone"
#   }
# }
#
# # DNS record for ppe domain on ALB
# resource "aws_route53_record" "ppe_app" {
#   zone_id = aws_route53_zone.root_zone.zone_id
#   name    = local.ppe_domain
#   type    = "A"
#
#   alias {
#     name                   = aws_lb.ppe_alb.dns_name
#     zone_id                = aws_lb.ppe_alb.zone_id
#     evaluate_target_health = false
#   }
# }
#
# # DNS validation record required by ACM
# resource "aws_route53_record" "ppe_cert_validation" {
#   for_each = {
#     for dvo in aws_acm_certificate.ppe_cert.domain_validation_options :
#     dvo.domain_name => {
#       name   = dvo.resource_record_name
#       type   = dvo.resource_record_type
#       record = dvo.resource_record_value
#     }
#   }
#
#   zone_id = aws_route53_zone.root_zone.zone_id
#   name    = each.value.name
#   type    = each.value.type
#   ttl     = 60
#   records = [each.value.record]
# }