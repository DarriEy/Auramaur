# Secrets, encrypted bundles, and the age identity

This repository is **public**. Nothing secret is ever committed — not even
encrypted. Private state lives on the host and, encrypted, in off-host object
storage. Git history on a protected branch cannot be scrubbed, so the rule has
no exceptions.

## What is private state

Two pieces, with different cadences:

1. **The database** — `deploy/backup.sh` (local, verified) and
   `deploy/offsite-backup.sh` (age-encrypted, uploaded via rclone). Runs daily
   via the supplied systemd timer.
2. **Everything else a bare host needs** — `deploy/offsite-bundle.sh` bundles
   `.env`, `secrets/`, `config/defaults.local.yaml`, `runtime/config/`,
   `runtime/ibgateway/` (minus logs), and `runtime/claude/` into one
   age-encrypted tarball and uploads it. Run it **after any credential or
   config change**, and at least monthly.

Restore is the mirror image: `deploy/restore-bundle.sh BUNDLE.tar.gz.age`
stages into a fresh directory (never over a live tree), then
`deploy/restore.sh BACKUP.db.gz` for the database.

Deliberately **not** in the bundle: the age identity itself (see below), the
rclone configuration (recreate it from the storage provider's console), the IB
Gateway installer (re-download from IBKR), and anything outside this repo
(Straummaur data, `~/.hermes`).

## The age identity

Encryption uses public recipients only — hosts that take backups never hold a
decryption key. `AURAMAUR_BACKUP_AGE_RECIPIENT` accepts **multiple
space-separated recipients**; every bundle should be encrypted to at least two:

- **Primary identity** — lives in a password manager or hardware-backed key
  store. Used for routine restores.
- **Recovery identity** — generated once, stored **offline** (printed or on a
  USB stick in a separate physical location). Never typed into a cloud
  service.
- Optional per-machine identity added during provisioning, removed when the
  machine is decommissioned.

Rules:

- The identity never touches GitHub — not the repository, not Actions
  secrets. A passphrase held only in Actions secrets is a single point of
  failure owned by someone else; do not use it as a recovery path.
- Losing every identity means the backups are noise. The recovery identity
  exists so that losing the password manager is survivable.
- Rotating: generate a new pair, add the new recipient, re-run
  `offsite-bundle.sh` and `offsite-backup.sh`, then retire the old identity
  once fresh backups exist under the new one.

## Restore drill

A backup that has never been restored is a hope, not a backup. Quarterly, on
any machine with `age` installed:

```sh
AURAMAUR_BACKUP_AGE_IDENTITY=path/to/identity.txt \
  deploy/restore-bundle.sh path/to/auramaur-bundle-*.tar.gz.age /tmp/drill
```

Check the staged tree contains `.env`, `secrets/kalshi-private.pem`, and the
Gateway/Claude state, then delete the staging directory. Do the same for the
newest database backup with `deploy/restore.sh` on a scratch path
(`AURAMAUR_DB_PATH=/tmp/drill.db`). Exercise the **recovery** identity at
least once a year so it is known-good, not assumed-good.

## Environment schema

`deploy/env.manifest.yaml` classifies every environment variable (secret vs
config, where it is required). `deploy/check-env.py` compares the manifest,
`.env.example`, and the local `.env` **by name only — values are never read
or printed** — and fails on drift. Run it before cutover and whenever `.env`
changes shape.
