# Release process

A release is a **citable, immutable** point. The goal is that a stranger can
point at a tag or a DOI and get the same bytes, the same contract, and the same
numbers later. Keep releases boring.

## 1. Pre-flight (must be green)

From the repo root:

```bash
venv/bin/python -c "import splinter"                 # facade imports clean
venv/bin/python -m pytest <owned tests> -q           # owned tests exit 0
```

And, from the sibling `hivebench` checkout, the offline suite:

```bash
pytest tests/unit tests/integration -q
```

A release is blocked on a red verification matrix. Do not "release anyway".

## 2. Version bump

- Edit `version` in `pyproject.toml`. This is a **hotspot**: commit it alone.
- Update `CHANGELOG.md`: move `[Unreleased]` entries under the new version and
  add today's date. Keep a Changelog / SemVer.

## 3. Commit, tag, push

```bash
git add -A
git commit -m "release: vX.Y.Z"
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin <default-branch>
git push origin vX.Y.Z
```

Never move or delete a released tag. If a release is wrong, publish the next
patch; do not rewrite history others may have fetched.

## 4. Freeze the contract anchors

Record, in the release notes, the identifiers a consumer can pin against:

- The wire contract file digest: `sha256sum docs/INTEGRATE.md`.
- The route/header/tool surface: `/v1/splinter/*`, `X-Splinter-Conversation`,
  `splinter_search`, `splinter_remember`.
- For the sibling forensics project, the canonical spec hash
  `0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b`
  (`python -m bonsai_forensics.spec_hash --check`).

A hash in the release notes turns "we did not change the contract" from a claim
into something a reader can check.

## 5. Publish a citable archive

1. Create the GitHub release for the tag.
2. Archive the tag to Zenodo (or an equivalent DOI mint) so the release gets a
   permanent DOI that carries no personal identity.
3. Add the DOI to `CITATION.cff` (`doi:` and, if applicable, `identifiers:`)
   and to the top of `README.md`.
4. Optionally attach a `git archive` tarball plus its `sha256` to the release
   as a byte-exact artifact.

Also attach the reproduction status of the release: which claims are
deterministic, which are live, and which remain recorded-only (see
`REPRODUCE.md`). Honest labels age better than optimistic ones.

## 6. Commit identity

This project is published under a pseudonym. Keep it that way:

```bash
git config user.name  hive-dev
git config user.email hive@local
```

and set every publishing machine to a GitHub **noreply** address
(`<id>+<handle>@users.noreply.github.com`), with "Keep my email addresses
private" enabled in account settings. Never let a personal name or address into
a commit: it is permanent and it defeats the pseudonym. Audit before each
release:

```bash
git log --format='%an <%ae>' | sort -u
git log --format='%ae' | grep -v 'users.noreply.github.com' | sort -u
```

## 7. External steps a release may depend on

These are not done by the release commit; track them explicitly:

- Rename the GitHub repository (`Strata-memory` → `splinter-memory`) so the
  README/`CITATION.cff` URLs resolve directly. GitHub keeps redirects from the
  old name, so existing links do not break.
- Rename the CI branch (`StrataMemory-test` → `SplinterMemory-test`) and update
  the `on.push.branches` filter in `.github/workflows/ci.yml` — the CI config
  and the branch must change together.
- Rename the checkout directory and the sibling env var
  (`STRATA_HOME` → `SPLINTER_HOME`) in the hivebench conftest once both
  checkouts move together.
