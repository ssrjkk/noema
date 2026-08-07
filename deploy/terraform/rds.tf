resource "aws_db_subnet_group" "this" {
  name        = "${var.app_name}-${var.environment}"
  description = "Database subnet group for ${var.app_name}"
  subnet_ids  = data.aws_subnets.private.ids

  tags = {
    Name = "${var.app_name}-${var.environment}"
  }
}

resource "aws_security_group" "rds" {
  name        = "${var.app_name}-${var.environment}-rds"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    description = "PostgreSQL from allowed CIDRs"
    from_port   = 5432
    to_port     = 5432
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
    Name = "${var.app_name}-${var.environment}-rds"
  }
}

resource "aws_db_instance" "this" {
  identifier     = "${var.app_name}-${var.environment}"
  engine         = "postgres"
  engine_version = "16.4"
  instance_class = var.db_instance_class

  allocated_storage     = 100
  storage_type          = "gp3"
  storage_encrypted     = true
  delete_automated_backups = false

  db_name  = "noema"
  username = "noema"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:05:00-sun:06:00"

  multi_az               = var.environment == "production" ? true : false
  deletion_protection    = true
  skip_final_snapshot    = false
  final_snapshot_identifier = "${var.app_name}-${var.environment}-final"

  auto_minor_version_upgrade = true

  tags = {
    Name = "${var.app_name}-${var.environment}"
  }
}

output "rds_endpoint" {
  value = aws_db_instance.this.endpoint
}

output "rds_address" {
  value = aws_db_instance.this.address
}
