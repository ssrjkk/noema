#!/bin/bash
set -euo pipefail

# PostgreSQL Backup Script
# Usage: ./pg_backup.sh [db_name] [s3_bucket]

DB_NAME="${1:-noema}"
S3_BUCKET="${2:-}"
BACKUP_DIR="${BACKUP_DIR:-/tmp/noema-backups/pg}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_${TIMESTAMP}.sql.gz"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
PG_HOST="${PG_HOST:-localhost}"
PG_PORT="${PG_PORT:-5432}"
PG_USER="${PG_USER:-noema}"
PG_PASSWORD="${PG_PASSWORD:-noema}"

mkdir -p "$BACKUP_DIR"

export PGPASSWORD="$PG_PASSWORD"

echo "[$(date +%Y-%m-%dT%H:%M:%S)] Starting PostgreSQL backup of ${DB_NAME}..."

pg_dump \
    -h "$PG_HOST" \
    -p "$PG_PORT" \
    -U "$PG_USER" \
    -d "$DB_NAME" \
    --format=custom \
    --compress=9 \
    --verbose \
    --no-owner \
    --clean \
    --if-exists \
    -f "$BACKUP_FILE" 2>&1

BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "[$(date +%Y-%m-%dT%H:%M:%S)] Backup completed: ${BACKUP_FILE} (${BACKUP_SIZE})"

# Upload to S3 if bucket specified
if [ -n "$S3_BUCKET" ]; then
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] Uploading to S3: s3://${S3_BUCKET}/pg-backups/"
    aws s3 cp "$BACKUP_FILE" "s3://${S3_BUCKET}/pg-backups/$(basename ${BACKUP_FILE})" \
        --storage-class STANDARD_IA
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] S3 upload completed"
fi

# Cleanup old backups
echo "[$(date +%Y-%m-%dT%H:%M:%S)] Cleaning backups older than ${RETENTION_DAYS} days..."
find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime "+${RETENTION_DAYS}" -delete
if [ -n "$S3_BUCKET" ]; then
    aws s3 ls "s3://${S3_BUCKET}/pg-backups/" | while read -r line; do
        FILE_DATE=$(echo "$line" | awk '{print $1" "$2}')
        FILE_NAME=$(echo "$line" | awk '{print $4}')
        FILE_TIMESTAMP=$(date -d "$FILE_DATE" +%s 2>/dev/null || echo 0)
        CUTOFF=$(date -d "-${RETENTION_DAYS} days" +%s)
        if [ "$FILE_TIMESTAMP" -lt "$CUTOFF" ] 2>/dev/null; then
            aws s3 rm "s3://${S3_BUCKET}/pg-backups/${FILE_NAME}"
        fi
    done
fi

echo "[$(date +%Y-%m-%dT%H:%M:%S)] PostgreSQL backup finished."
export PGPASSWORD=
