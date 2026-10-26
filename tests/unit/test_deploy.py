"""Static guards for the dedicated AI-user deployment.

The actual destructive test needs root and a deployed account; these checks
keep the unit and setup script from drifting back to a shared Docker socket or
a human-writable app tree.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNIT = (ROOT / "deploy" / "hive.service").read_text(encoding="utf-8")
SETUP = (ROOT / "scripts" / "setup-hive-account.sh").read_text(encoding="utf-8")
BLAST = (ROOT / "scripts" / "test-blast-radius.sh").read_text(encoding="utf-8")


def test_unit_runs_the_whole_app_as_hive_with_hardening():
    assert "User=hive" in UNIT
    assert "WorkingDirectory=/srv/hive/app" in UNIT
    assert "HIVE_HOME=/srv/hive" in UNIT
    assert "HIVE_STATE_DIR=/srv/hive/state" in UNIT
    assert "DSH_HOME=/srv/hive/dsh" in UNIT
    assert "ProtectSystem=strict" in UNIT
    # ProtectHome=yes would hide /run/user and break the rootless socket.
    assert "ProtectHome=read-only" in UNIT
    assert "ProtectHome=yes" not in UNIT
    assert "PrivateTmp=yes" in UNIT
    assert "NoNewPrivileges=yes" in UNIT
    assert "CapabilityBoundingSet=" in UNIT
    assert "ReadWritePaths=/srv/hive" in UNIT
    assert "UMask=0077" in UNIT


def test_unit_uses_the_ai_rootless_socket_not_the_human_socket():
    assert "/var/run/docker.sock" not in UNIT
    assert "DOCKER_HOST" in SETUP
    assert "/run/user/${HIVE_UID}/docker.sock" in SETUP
    assert "usermod -aG docker" not in SETUP
    assert "gpasswd -a" not in SETUP


def test_blast_radius_test_is_explicit_and_sentinel_only():
    assert "--yes" in BLAST
    assert "HIVE-BLAST-HUMAN" in BLAST
    assert "HIVE-BLAST-AI" in BLAST
    assert "human_docker_rc" in BLAST
    assert "systemd-run" in BLAST
    assert "/var/run/docker.sock" in BLAST
    # It must not contain a broad destructive command against a real path.
    for forbidden in ("rm -rf /", "rm -rf /home", "rm -rf /srv"):
        assert forbidden not in BLAST
