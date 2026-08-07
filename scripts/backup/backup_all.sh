#!/bin/bash
set -euo pipefail

# Run all backups
# Usage: ./backup_all.sh [s3_bucket]

S3_BUCKET="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  Noema — Full Backup"
echo "  $(date +%Y-%m-%dT%H:%M:%S)"
echo "=========================================="

echo ""
echo "--- PostgreSQL Backup ---"
bash "$SCRIPT_DIR/pg_backup.sh" "noema" "$S3_BUCKET"

echo ""
echo "--- Redis Backup ---"
bash "$SCRIPT_DIR/redis_backup.sh" "$S3_BUCKET"

echo ""
echo "=========================================="
echo "  All backups completed successfully!"
echo "=========================================="
