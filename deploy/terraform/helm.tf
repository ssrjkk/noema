locals {
  db_url  = "postgresql+asyncpg://noema:${var.db_password}@${aws_db_instance.this.address}:5432/noema"
  redis_url = "redis://default:${var.redis_password}@${aws_elasticache_replication_group.this.primary_endpoint_address}:6379/0"
}

resource "helm_release" "nginx_ingress" {
  name       = "nginx-ingress"
  namespace  = "ingress-nginx"
  create_namespace = true

  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  version    = "~> 4.11"

  set {
    name  = "controller.publishService.enabled"
    value = "true"
  }
}

resource "helm_release" "noema" {
  name       = "${var.app_name}"
  namespace  = "${var.app_name}"
  create_namespace = true

  chart      = "../../helm/noema"
  version    = var.image_tag == "latest" ? null : var.image_tag

  values = [
    jsonencode({
      image = {
        tag = var.image_tag
      }
      postgresql = {
        enabled = false
      }
      redis = {
        enabled = false
      }
      ingress = {
        className = "nginx"
        hosts = [
          {
            host = var.domain_name
            paths = [
              {
                path = "/"
                pathType = "Prefix"
              }
            ]
          }
        ]
        tls = [
          {
            secretName = "noema-tls"
            hosts = [var.domain_name]
          }
        ]
      }
      env = {
        NOEMA_DB_URL  = local.db_url
        NOEMA_REDIS_URL = local.redis_url
      }
    })
  ]

  depends_on = [
    aws_eks_cluster.this,
    aws_db_instance.this,
    aws_elasticache_replication_group.this,
  ]
}
