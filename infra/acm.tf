# ########################################
# # ACM Certificate for HTTPS
# ########################################
#
# # Request an ACM certificate for ppe domain
# resource "aws_acm_certificate" "ppe_cert" {
#   domain_name       = local.ppe_domain
#   validation_method = "DNS"
#
#   tags = {
#     Name = "${local.app_prefix}-cert"
#   }
# }
#
# # Validate ACM is using the DNS record
# resource "aws_acm_certificate_validation" "ppe_cert_validation" {
#   certificate_arn         = aws_acm_certificate.ppe_cert.arn
#   validation_record_fqdns = [for record in aws_route53_record.ppe_cert_validation : record.fqdn]
# }
