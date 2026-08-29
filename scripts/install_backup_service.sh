#!/bin/sh
set -eu

project_root=${1:-/opt/quantlab}
ops_root=${QUANTLAB_OPS_ROOT:-/opt/quantlab-ops}
backup_root=${QUANTLAB_BACKUP_ROOT:-/opt/quantlab-backups}
python_bin=${PYTHON_BIN:-python3}

if [ "$(id -u)" -ne 0 ]; then
    echo "install_backup_service.sh must run as root" >&2
    exit 1
fi
if [ ! -f "$project_root/deploy/backup-ops-requirements.txt" ]; then
    echo "backup ops requirements are missing from $project_root" >&2
    exit 1
fi
if [ ! -f "$project_root/deploy/systemd/quantlab-backup.service" ]; then
    echo "backup systemd unit is missing from $project_root" >&2
    exit 1
fi

install -d -m 0750 "$ops_root" "$backup_root" /var/lib/quantlab-ops
if [ ! -x "$ops_root/venv/bin/python" ]; then
    "$python_bin" -m venv "$ops_root/venv"
fi
"$ops_root/venv/bin/python" -m pip install \
    --disable-pip-version-check \
    --requirement "$project_root/deploy/backup-ops-requirements.txt"

install -m 0644 \
    "$project_root/deploy/systemd/quantlab-backup.service" \
    /etc/systemd/system/quantlab-backup.service
install -m 0644 \
    "$project_root/deploy/systemd/quantlab-backup.timer" \
    /etc/systemd/system/quantlab-backup.timer
systemctl daemon-reload
systemctl enable --now quantlab-backup.timer
systemctl status --no-pager quantlab-backup.timer
