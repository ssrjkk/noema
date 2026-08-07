resource "aws_elasticache_subnet_group" "this" {
  name        = "${var.app_name}-${var.environment}"
  description = "ElastiCache subnet group for ${var.app_name}"
  subnet_ids  = data.aws_subnets.private.ids
}

resource "aws_security_group" "redis" {
  name        = "${var.app_name}-${var.environment}-redis"
  description = "Security group for ElastiCache Redis"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    description = "Redis from allowed CIDRs"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.app_name}-${var.environment}-redis"
  }
}

resource "aws_elasticache_replication_group" "this" {
  replication_group_id          = "${var.app_name}-${var.environment}"
  description                   = "Redis replication group for ${var.app_name}"
  node_type                     = var.redis_node_type
  num_cache_clusters            = var.environment == "production" ? 2 : 1
  port                          = 6379
  parameter_group_name          = "default.redis7"

  engine                       = "redis"
  engine_version               = "7.1"

  automatic_failover_enabled   = var.environment == "production" ? true : false
  multi_az_enabled             = var.environment == "production" ? true : false

  at_rest_encryption_enabled   = true
  transit_encryption_enabled   = true
  auth_token                   = var.redis_password

  subnet_group_name            = aws_elasticache_subnet_group.this.name
  security_group_ids           = [aws_security_group.redis.id]

  auto_minor_version_upgrade   = true
  maintenance_window           = "sun:06:00-sun:07:00"

  tags = {
    Name = "${var.app_name}-${var.environment}"
  }
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.this.primary_endpoint_address
}

output "redis_address" {
  value = aws_elasticache_replication_group.this.primary_endpoint_address
}
