#!/bin/bash
set -euo pipefail

# Redis Backup Script
# Usage: ./redis_backup.sh [s3_bucket]

S3_BUCKET="${1:-}"
BACKUP_DIR="${BACKUP_DIR:-/tmp/noema-backups/redis}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/redis_dump_${TIMESTAMP}.rdb"
RETENTION_DAYS="${RETENTION_DAYS:-3}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

mkdir -p "$BACKUP_DIR"

echo "[$(date +%Y-%m-%dT%H:%M:%S)] Starting Redis backup..."

# Trigger SAVE
if [ -n "$REDIS_PASSWORD" ]; then
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" SAVE
else
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" SAVE
fi

echo "[$(date +%Y-%m-%dT%H:%M:%S)] SAVE triggered"

# Copy RDB file
RDB_PATH=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG GET dir | tail -1)
cp "${RDB_PATH}/dump.rdb" "$BACKUP_FILE"
gzip "$BACKUP_FILE"

BACKUP_SIZE=$(du -h "${BACKUP_FILE}.gz" | cut -f1)
echo "[$(date +%Y-%m-%dT%H:%M:%S)] Backup completed: ${BACKUP_FILE}.gz (${BACKUP_SIZE})"

# Upload to S3 if bucket specified
if [ -n "$S3_BUCKET" ]; then
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] Uploading to S3: s3://${S3_BUCKET}/redis-backups/"
    aws s3 cp "${BACKUP_FILE}.gz" "s3://${S3_BUCKET}/redis-backups/$(basename ${BACKUP_FILE}).gz" \
        --storage-class STANDARD_IA
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] S3 upload completed"
fi

# Cleanup old backups
echo "[$(date +%Y-%m-%dT%H:%M:%S)] Cleaning backups older than ${RETENTION_DAYS} days..."
find "$BACKUP_DIR" -name "redis_dump_*.rdb.gz" -mtime "+${RETENTION_DAYS}" -delete

echo "[$(date +%Y-%m-%dT%H:%M:%S)] Redis backup finished."
