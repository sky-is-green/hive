"""Static guards for the dedicated AI-user deployment.

The actual destructive test needs root and a deployed account; these checks
keep the unit and setup script from drifting back to GUARD's `hive` account, a
shared Docker socket, or a human-writable app tree.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNIT = (ROOT / "deploy" / "hive.service").read_text(encoding="utf-8")
SETUP = (ROOT / "scripts" / "setup-hive-account.sh").read_text(encoding="utf-8")
BLAST = (ROOT / "scripts" / "test-blast-radius.sh").read_text(encoding="utf-8")
DOCS = (ROOT / "docs" / "AI-USER.md").read_text(encoding="utf-8")


def test_unit_runs_the_whole_app_as_hive_ai_with_hardening():
    assert "User=hive-ai" in UNIT
    assert "Group=hive-ai" in UNIT
    # GUARD owns the `hive` account; a substring check is not enough.
    assert "\nUser=hive\n" not in UNIT
    assert "WorkingDirectory=/srv/hive-ai/app" in UNIT
    assert "HIVE_HOME=/srv/hive-ai" in UNIT
    assert "HIVE_STATE_DIR=/srv/hive-ai/state" in UNIT
    assert "DSH_HOME=/srv/hive-ai/dsh" in UNIT
    assert "HIVE_DSH_SESSION_ROOT=/srv/hive-ai/dsh/sessions" in UNIT
    # Relative defaults would resolve under the read-only app tree.
    assert "--state-dir /srv/hive-ai/state" in UNIT
    assert "--log-dir /srv/hive-ai/logs" in UNIT
    assert "--providers-file /srv/hive-ai/state/providers.local.json" in UNIT
    assert "ProtectSystem=strict" in UNIT
    # ProtectHome=yes/read-only would hide or freeze /run/user and break the
    # rootless socket; the homes are hidden with InaccessiblePaths instead.
    assert "\nProtectHome=no\n" in UNIT
    assert "\nProtectHome=yes\n" not in UNIT
    assert "\nProtectHome=read-only\n" not in UNIT
    assert "InaccessiblePaths=/home -/root" in UNIT
    assert "PrivateTmp=yes" in UNIT
    assert "NoNewPrivileges=yes" in UNIT
    assert "CapabilityBoundingSet=" in UNIT
    assert "ReadWritePaths=/srv/hive-ai" in UNIT
    assert "EnvironmentFile=-/etc/hive/hive.env" in UNIT
    assert "UMask=0077" in UNIT


def test_unit_uses_the_ai_rootless_socket_not_the_human_socket():
    assert "/var/run/docker.sock" not in UNIT
    assert "DOCKER_HOST" in SETUP
    assert "/run/user/${HIVE_UID}/docker.sock" in SETUP
    assert 'HIVE_USER="${HIVE_USER:-hive-ai}"' in SETUP
    assert 'HIVE_BASE="${HIVE_BASE:-/srv/hive-ai}"' in SETUP
    assert "/etc/hive/hive.env" in SETUP
    # The AI-writable base would allow renaming the read-only app/venv.
    assert 'install -d -o root -g "$HIVE_GROUP" -m 0750 "$HIVE_BASE"' in SETUP
    # The Studio imports cortex/sieve/backend from the absorbed splinter tree.
    assert '"$HIVE_BASE/app/splinter"' in SETUP
    assert 'chown -R root:"$HIVE_GROUP" "$HIVE_BASE/app" "$HIVE_BASE/venv"' in SETUP
    assert "shares GUARD's uid" in SETUP
    assert "is in the human docker group" in SETUP
    assert "usermod -aG docker" not in SETUP
    assert "gpasswd -a" not in SETUP


def test_setup_does_not_touch_guard_or_require_usermod_home():
    assert 'if [[ "$HIVE_USER" == "hive" ]]' in SETUP
    assert "owned by GUARD" in SETUP
    # GUARD's live process makes `usermod -d` fail; the unit sets HOME.
    assert "usermod -d" not in SETUP
    assert 'HIVE_USER" == "hive-ai" && "$HIVE_BASE" == "/srv/hive-ai"' in SETUP


def test_blast_radius_test_is_explicit_and_sentinel_only():
    assert "--yes" in BLAST
    assert 'HIVE_USER="${HIVE_USER:-hive-ai}"' in BLAST
    assert 'HIVE_BASE="${HIVE_BASE:-/srv/hive-ai}"' in BLAST
    assert "HIVE-BLAST-HUMAN" in BLAST
    assert "HIVE-BLAST-AI" in BLAST
    assert "human_docker_rc" in BLAST
    assert "guard_list_rc" in BLAST
    assert "guard_touch_rc" in BLAST
    assert "systemd-run" in BLAST
    assert "InaccessiblePaths=\"/home -/root\"" in BLAST
    assert "ReadWritePaths=\"$HIVE_BASE\"" in BLAST
    assert "runtime_write_rc" in BLAST
    assert "home_visible_rc" in BLAST
    # The app-tree unlink is bounded and non-interactive.
    assert 'timeout 5 rm -f "$APP"' in BLAST
    assert 'test -e "$HUMAN_HOME"' in BLAST
    assert "/var/run/docker.sock" in BLAST
    assert "id -nG \"$HIVE_USER\"" in BLAST
    # It must not contain a broad destructive command against a real path.
    for forbidden in ("rm -rf /", "rm -rf /home", "rm -rf /srv"):
        assert forbidden not in BLAST


def test_docs_name_the_new_account_and_the_guard_boundary():
    assert "hive-ai" in DOCS
    assert "/srv/hive-ai" in DOCS
    assert "GUARD" in DOCS
    assert "InaccessiblePaths=/home" in DOCS
    assert "usermod -d" in DOCS
