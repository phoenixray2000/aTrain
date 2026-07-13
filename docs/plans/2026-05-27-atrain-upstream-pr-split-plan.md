# aTrain Upstream PR Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏 `D:\Git\aTrain` 当前可用集成分支的前提下，把本地 CLI、外部词表、GUI 声纹、CLI 声纹能力拆成多个可审阅的上游 PR，逐个提交到 `JuergenFleiss/aTrain:develop`。

**Architecture:** `D:\Git\aTrain` 保持为本地使用分支，不直接改、不直接 rebase。PR 1 和 PR 3 可以并行从最新上游 `develop` 准备；PR 2 依赖 PR 1 的 CLI 基座；PR 4 依赖 PR 1 的 CLI 基座和 PR 3 的声纹核心，但不依赖 PR 2 的 hotwords/replace-map。上游已经包含 Flatpak、uv、ruff、CI 和安全文档，因此 PR 只重放本地功能，不覆盖上游基础设施。

**Tech Stack:** Git worktree, GitHub fork workflow, Python 3.11, Typer, NiceGUI, unittest, uv/ruff from upstream develop, PyInstaller specs.

---

## Current State

**Local usable branch:** `codex-speaker-voiceprint-identification`

**Local HEAD when this plan was written:** `d6b5def4`

**True upstream:** `https://github.com/JuergenFleiss/aTrain.git`

**Upstream target branch:** `develop`

**Divergence at plan time:** using `git rev-list --left-right --count HEAD...upstream/develop`, the local branch has `26` unique commits and upstream `develop` has `68` unique commits. All PR bases in this plan use `upstream/develop`, not `upstream/main`.

**Do not include in upstream PRs:**
- `docs/superpowers/plans/2026-05-16-atrain-external-vocabulary-consumption.md`
- `docs/superpowers/plans/2026-05-26-atrain-cli-voiceprint-enrollment.md`
- local version bump commits
- local dependency pins to old bugfix/release branches when upstream `develop` already uses `aTrain_core @ develop`
- local model-manager/storage experiments that conflict with upstream Flatpak model handling unless explicitly approved later

**Keep untouched for local use:**
- `D:\Git\aTrain`
- current branch `codex-speaker-voiceprint-identification`
- untracked local artifacts `.matplotlib-cache/`, `.model-backup/`, `dist_fixed/`

---

## Branch And Worktree Layout

Use sibling worktrees outside the main checkout:

```text
D:\Git\aTrain                  # current integrated local branch, keep usable
D:\Git\aTrain-prs\01-cli       # PR 1 worktree
D:\Git\aTrain-prs\02-vocab     # PR 2 worktree, stack on PR 1 while PR 1 is open
D:\Git\aTrain-prs\03-gui-vp    # PR 3 worktree, independent from PR 1/2
D:\Git\aTrain-prs\04-cli-vp    # PR 4 worktree, stack on PR 1 + PR 3 while they are open
```

Branch names pushed to the fork:

```text
codex-atrain-cli-transcribe
codex-atrain-cli-vocabulary
codex-atrain-gui-voiceprints
codex-atrain-cli-voiceprint-enrollment
```

Target every PR at:

```text
JuergenFleiss/aTrain:develop
```

Push every branch to:

```text
phoenixray2000/aTrain
```

Preferred dependency graph:

```text
upstream/develop
├─ PR 1: standalone CLI
│  └─ PR 2: CLI prompt/hotwords/replace-map
└─ PR 3: GUI speaker voiceprints
   └─ PR 4: CLI voiceprint enrollment
      also depends on PR 1's CLI base
```

Practical review strategy:

```text
Open early:
  PR 1 -> base upstream/develop
  PR 3 -> base upstream/develop

Open as stacked/draft while dependencies are under review:
  PR 2 -> base codex-atrain-cli-transcribe
  PR 4 -> base a temporary integration branch containing PR 1 + PR 3

Before asking upstream maintainers for final review:
  Retarget or rebase stacked PRs onto upstream/develop after their dependencies merge.
```

This avoids blocking CLI voiceprint work on PR 2, and lets reviewers inspect GUI voiceprint and CLI foundation independently.

---

## File Responsibility Map

### PR 1: Standalone CLI Transcription

**Purpose:** Add a scriptable `aTrain-cli transcribe` and `aTrain-cli init` without GUI startup.

**Create:**
- `aTrain/cli.py` - Typer CLI, input collection, output routing, model init, batch exit codes.
- `freeze_cli.py` - PyInstaller console entrypoint for packaged CLI.
- `pyi_runtime_model_paths.py` - packaged runtime hook only if still needed after adapting to upstream Flatpak/uv state.
- `tests/test_cli_paths.py` - regression for paths with spaces/unicode.

**Modify:**
- `pyproject.toml` - add `aTrain-cli = "aTrain.cli:cli"` under `[project.scripts]` while preserving upstream `uv` sections.
- `freeze.spec` - add CLI executable only if Windows packaging is included in this PR.
- `macos_freeze.spec` - add CLI executable only if macOS packaging is included in this PR.
- `README.md` - add minimal CLI usage section; do not remove upstream Flathub/uv/security documentation.

**Do not include:**
- `aTrain/cli_vocabulary.py`
- `aTrain/transcription_hotwords.py`
- `aTrain/voiceprints.py`
- `aTrain/voiceprint_cli.py`
- `aTrain/voiceprint_identification.py`
- `tests/test_cli_vocabulary.py`
- `tests/test_cli_voiceprints.py`
- any `docs/superpowers/plans/*`

### PR 2: CLI Prompt, Hotwords, And Replace Map

**Purpose:** Extend the CLI from PR 1 with upstream-generated vocabulary inputs.

**Create:**
- `aTrain/cli_vocabulary.py` - UTF-8 prompt/hotwords loading, JSON/YAML replacement map validation, transcript replacement helpers.
- `aTrain/transcription_hotwords.py` - local monkey patch around `aTrain_core.transcribe.run_transcription` to pass faster-whisper `hotwords`.
- `tests/test_cli_vocabulary.py`
- `tests/test_transcription_hotwords.py`

**Modify:**
- `aTrain/cli.py` - add `--prompt-file`, `--hotwords`, `--hotwords-file`, `--replace-map`; call vocabulary helpers; post-process staged JSON before copying outputs.
- `README.md` - document the new CLI inputs and one example.

**Do not include:**
- voiceprint files or voiceprint CLI flags
- GUI voiceprint page/sidebar changes
- local plan docs

### PR 3: GUI Speaker Voiceprints

**Purpose:** Add GUI enrollment and speaker relabeling during GUI transcription.

**Create:**
- `aTrain/voiceprints.py` - persistent voiceprint profile schema, validation, save/load/list/remove, cosine similarity and assignment.
- `aTrain/voiceprint_identification.py` - local embedding extraction and diarization embedding capture.
- `aTrain/utils/voiceprints.py` - NiceGUI async enrollment/removal helpers.
- `aTrain/components/voiceprints/enroll_dialog.py` - enrollment dialog and upload UI.
- `aTrain/pages/voiceprints.py` - voiceprint management page.
- `tests/test_voiceprints.py`
- `tests/test_voiceprint_identification.py`
- `tests/test_gui_voiceprint_identification.py`

**Modify:**
- `aTrain/app.py` - import `voiceprints` page while preserving upstream `FLATPAK` storage and wakepy behavior.
- `aTrain/components/layout/sidebar.py` - add Voiceprints navigation item.
- `aTrain/utils/transcription.py` - wrap GUI transcription with speaker capture and apply speaker map when profiles exist.
- `README.md` - document GUI voiceprint behavior briefly.

**Do not include:**
- `aTrain/voiceprint_cli.py`
- `voiceprint enroll` CLI command
- `--speaker-embeddings-output`
- `tests/test_voiceprint_cli.py`
- CLI voiceprint assertions in `tests/test_cli_voiceprints.py`

### PR 4: CLI Voiceprint Enrollment And Speaker Embedding Export

**Purpose:** Add CLI management for voiceprint profiles and allow transcription to export captured speaker embeddings.

**Create:**
- `aTrain/voiceprint_cli.py` - audio enrollment and `.npz` speaker embedding enrollment helpers.
- `tests/test_voiceprint_cli.py`
- `tests/test_cli_voiceprints.py`

**Modify:**
- `aTrain/cli.py` - add `voiceprint enroll`; add `--identify-speakers`, `--voiceprint-threshold`, `--voiceprint-margin`, `--speaker-embeddings-output`; enforce single-file export and `--speaker-detection` gates.
- `README.md` - document CLI voiceprint enrollment and export flow.

**Required fixes to preserve from local `d6b5def4`:**
- `voiceprint enroll --audio` checks `check_model_downloaded("speaker-detection")` before `get_model("speaker-detection")`.
- CLI enrollment maps `Device.CPU/GPU` to `torch.device("cpu"/"cuda")`.
- GPU enrollment fails before extraction when CUDA is unavailable.
- `_copy_speaker_embeddings()` respects `--no-overwrite`.

---

## Task 1: Preparation And Baseline Audit

**Files:**
- Read only: repository metadata and Git refs.
- Modify: none.

- [ ] **Step 1: Confirm current local branch remains the integration branch**

Run in `D:\Git\aTrain`:

```powershell
rtk git status --short --branch
```

Expected shape:

```text
## codex-speaker-voiceprint-identification...origin/codex-speaker-voiceprint-identification [ahead 6]
?? .matplotlib-cache/
?? .model-backup/
?? dist_fixed/
```

The exact ahead count may change if more local commits were added. Do not clean or delete the untracked artifact directories.

- [ ] **Step 2: Configure upstream remote if it is missing**

Run:

```powershell
rtk git remote -v
```

If no `upstream` remote exists, run:

```powershell
rtk git remote add upstream https://github.com/JuergenFleiss/aTrain.git
```

If `upstream` already exists with another URL, stop and inspect before changing it:

```powershell
rtk git remote get-url upstream
```

Expected URL:

```text
https://github.com/JuergenFleiss/aTrain.git
```

- [ ] **Step 3: Fetch upstream and fork**

Run:

```powershell
rtk git fetch upstream develop main
rtk git fetch origin
```

Expected: both commands exit `0`.

- [ ] **Step 4: Record divergence**

Run:

```powershell
rtk git rev-list --left-right --count HEAD...upstream/develop
rtk git log --oneline --max-count=8 upstream/develop
```

Expected: the first command prints two integers. The second command should show upstream `develop` commits including CI/ruff/uv work unless upstream has moved further.

- [ ] **Step 5: Create the worktree parent**

Run:

```powershell
rtk powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path 'D:\Git\aTrain-prs' | Out-Null"
```

Expected: command exits `0`.

- [ ] **Step 6: Commit nothing in the main checkout**

Run:

```powershell
rtk git diff --stat
rtk git status --short --branch
```

Expected: no tracked file modifications from preparation.

---

## Task 2: PR 1 - Standalone CLI Transcription

**Files:**
- Create: `aTrain/cli.py`
- Create: `freeze_cli.py`
- Create: `pyi_runtime_model_paths.py` only if packaging validation needs it
- Create: `tests/test_cli_paths.py`
- Modify: `pyproject.toml`
- Modify: `freeze.spec`
- Modify: `macos_freeze.spec`
- Modify: `README.md`

- [ ] **Step 1: Create a fresh PR 1 worktree from upstream develop**

Run in `D:\Git\aTrain`:

```powershell
rtk git worktree add -b codex-atrain-cli-transcribe D:\Git\aTrain-prs\01-cli upstream/develop
```

Expected:

```text
HEAD is now at <sha> <upstream develop commit>
```

- [ ] **Step 2: Apply only CLI transcription source commits**

Run in `D:\Git\aTrain-prs\01-cli`:

```powershell
rtk git cherry-pick 65550e00 1f34a6a2 8937fbb2 e8fce672
```

Expected: four commits apply or stop at conflicts. If conflicts happen, preserve upstream `pyproject.toml` uv/ruff sections and add only the CLI script entry and CLI packaging changes.

- [ ] **Step 3: Add the path-with-spaces regression without importing later voiceprint code**

Create `tests/test_cli_paths.py` from local commit `917fed7d`, but keep `_transcribe_one()` arguments matching PR 1 only:

```python
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aTrain.cli import InputFile, _transcribe_one
from aTrain_core.settings import ComputeType, Device


class CliPathTests(unittest.TestCase):
    def test_transcribe_one_preserves_original_file_path_with_unicode_spaces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "录音 (28).m4a"
            audio_path.write_bytes(b"placeholder")
            item = InputFile(audio_path, Path(audio_path.name), Path("."))

            def assert_original_path(path, *args):
                self.assertEqual(str(audio_path), path)

            def assert_original_settings(settings):
                self.assertEqual(audio_path, settings.file)
                self.assertEqual(audio_path.name, settings.file_name)

            with (
                mock.patch("aTrain.cli.check_inputs_transcribe", side_effect=assert_original_path),
                mock.patch("aTrain_core.transcribe.transcribe", side_effect=assert_original_settings),
            ):
                _transcribe_one(
                    item=item,
                    output_plan=[],
                    overwrite=True,
                    model="large-v3-turbo",
                    language="auto-detect",
                    speaker_detection=False,
                    speaker_count=0,
                    device=Device.CPU,
                    compute_type=ComputeType.FLOAT32,
                    temperature=None,
                    prompt=None,
                    cpu_threads=0,
                )
```

- [ ] **Step 4: Patch `_transcribe_one()` to keep the original path**

In `aTrain/cli.py`, the relevant block must use the original `item.path` after `prepare_transcription()`:

```python
_, file_id, timestamp = prepare_transcription(item.path)
# prepare_transcription sanitizes names for file_id; decoding must use the real path.
file = item.path
check_inputs_transcribe(str(file), model, language, device)
```

- [ ] **Step 5: Remove accidental later-feature imports and flags**

Run:

```powershell
rtk rg --encoding utf-8 -n "voiceprint|hotwords|replace_map|cli_vocabulary|transcription_hotwords|speaker_embeddings" aTrain tests README.md
```

Expected: no matches except unrelated prose already present upstream. If matches are inside `aTrain/cli.py` or PR 1 tests, remove them from this PR.

- [ ] **Step 6: Preserve upstream pyproject and add only CLI entrypoint**

In `pyproject.toml`, keep upstream `aTrain_core@git+https://github.com/JuergenFleiss/atrain_core.git@develop`, `[tool.uv]`, `[tool.uv.sources]`, and `[dependency-groups]`. The only script addition should be:

```toml
[project.scripts]
aTrain = "aTrain.app:cli"
aTrain-cli = "aTrain.cli:cli"
```

- [ ] **Step 7: Run PR 1 tests**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest tests.test_cli_paths -v
rtk .\.venv\Scripts\python.exe -m aTrain.cli --help
rtk .\.venv\Scripts\python.exe -m aTrain.cli transcribe --help
rtk git diff --check
```

Expected:
- unittest exits `0`
- both `--help` commands exit `0`
- `git diff --check` prints no output

- [ ] **Step 8: Commit PR 1**

Run:

```powershell
rtk git status --short
rtk git add aTrain/cli.py freeze_cli.py pyproject.toml freeze.spec macos_freeze.spec README.md tests/test_cli_paths.py
rtk git add pyi_runtime_model_paths.py
rtk git commit -m "Add standalone CLI transcription workflow"
```

If `pyi_runtime_model_paths.py` was not needed, omit that `git add` line.

- [ ] **Step 9: Push PR 1 branch**

Run:

```powershell
rtk git push -u origin codex-atrain-cli-transcribe
```

Open a draft PR:

```text
base: JuergenFleiss/aTrain:develop
compare: phoenixray2000:codex-atrain-cli-transcribe
title: Add standalone CLI transcription workflow
```

PR body checklist:

```markdown
## Summary
- adds `aTrain-cli` for non-GUI transcription and model initialization
- supports file/directory input, recursive scans, format-specific output dirs, overwrite protection, and deterministic exit codes
- keeps GUI behavior unchanged

## Tests
- `python -m unittest tests.test_cli_paths -v`
- `python -m aTrain.cli --help`
- `python -m aTrain.cli transcribe --help`
- `git diff --check`
```

---

## Task 3: PR 2 - CLI Prompt, Hotwords, And Replace Map

**Files:**
- Create: `aTrain/cli_vocabulary.py`
- Create: `aTrain/transcription_hotwords.py`
- Create: `tests/test_cli_vocabulary.py`
- Create: `tests/test_transcription_hotwords.py`
- Modify: `aTrain/cli.py`
- Modify: `README.md`

- [ ] **Step 1: Choose PR 2 base**

Recommended while PR 1 is still under review: stack PR 2 on the PR 1 branch so reviewers can see only vocabulary changes.

```text
base: phoenixray2000:codex-atrain-cli-transcribe
compare: phoenixray2000:codex-atrain-cli-vocabulary
```

If PR 1 has already merged upstream, base PR 2 directly on `upstream/develop`.

- [ ] **Step 2: Refresh upstream develop**

Run in `D:\Git\aTrain`:

```powershell
rtk git fetch upstream develop
```

- [ ] **Step 3: Create PR 2 worktree**

If PR 1 is still open, run:

```powershell
rtk git worktree add -b codex-atrain-cli-vocabulary D:\Git\aTrain-prs\02-vocab codex-atrain-cli-transcribe
```

If PR 1 has merged, run instead:

```powershell
rtk git worktree add -b codex-atrain-cli-vocabulary D:\Git\aTrain-prs\02-vocab upstream/develop
```

- [ ] **Step 4: Apply vocabulary commits**

Run in `D:\Git\aTrain-prs\02-vocab`:

```powershell
rtk git cherry-pick f829832d a02785c4 4ba15906 be198c7f 093a1905
```

If the README hunk conflicts, keep upstream Flathub/uv/security sections and add only CLI vocabulary documentation.

- [ ] **Step 5: Exclude local planning doc**

Run:

```powershell
rtk git status --short
```

If `docs/superpowers/plans/2026-05-16-atrain-external-vocabulary-consumption.md` appears, remove it from the PR:

```powershell
rtk git restore --staged docs/superpowers/plans/2026-05-16-atrain-external-vocabulary-consumption.md
rtk git rm --cached docs/superpowers/plans/2026-05-16-atrain-external-vocabulary-consumption.md
rtk powershell -NoProfile -Command "Remove-Item -LiteralPath 'docs\superpowers\plans\2026-05-16-atrain-external-vocabulary-consumption.md' -Force"
```

- [ ] **Step 6: Verify CLI vocabulary flags exist and voiceprint flags do not**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m aTrain.cli transcribe --help
rtk rg --encoding utf-8 -n "prompt-file|hotwords|replace-map" aTrain/cli.py README.md
rtk rg --encoding utf-8 -n "voiceprint|identify-speakers|speaker-embeddings-output" aTrain/cli.py tests README.md
```

Expected:
- first two commands find vocabulary options
- third command has no matches from this PR

- [ ] **Step 7: Run PR 2 tests**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest tests.test_cli_vocabulary tests.test_transcription_hotwords tests.test_cli_paths -v
rtk git diff --check
```

Expected: tests pass and `git diff --check` prints no output.

- [ ] **Step 8: Commit PR 2**

Run:

```powershell
rtk git add aTrain/cli.py aTrain/cli_vocabulary.py aTrain/transcription_hotwords.py README.md tests/test_cli_vocabulary.py tests/test_transcription_hotwords.py
rtk git commit -m "Add CLI vocabulary inputs"
```

- [ ] **Step 9: Push PR 2 branch**

Run:

```powershell
rtk git push -u origin codex-atrain-cli-vocabulary
```

Open a draft PR. If PR 1 is still open, target the PR 1 branch in the fork:

```text
base: phoenixray2000/aTrain:codex-atrain-cli-transcribe
compare: phoenixray2000:codex-atrain-cli-vocabulary
title: Add CLI prompt, hotwords, and replacement-map inputs
```

After PR 1 merges, rebase PR 2 onto `upstream/develop` and retarget it:

```text
base: JuergenFleiss/aTrain:develop
compare: phoenixray2000:codex-atrain-cli-vocabulary
title: Add CLI prompt, hotwords, and replacement-map inputs
```

PR body checklist:

```markdown
## Summary
- adds UTF-8 prompt and hotword file inputs to the CLI
- passes hotwords into faster-whisper through a narrow runtime patch
- validates JSON/YAML replacement maps and applies replacements to output transcripts

## Tests
- `python -m unittest tests.test_cli_vocabulary tests.test_transcription_hotwords tests.test_cli_paths -v`
- `git diff --check`
```

---

## Task 4: PR 3 - GUI Speaker Voiceprints

**Files:**
- Create: `aTrain/voiceprints.py`
- Create: `aTrain/voiceprint_identification.py`
- Create: `aTrain/utils/voiceprints.py`
- Create: `aTrain/components/voiceprints/enroll_dialog.py`
- Create: `aTrain/pages/voiceprints.py`
- Create: `tests/test_voiceprints.py`
- Create: `tests/test_voiceprint_identification.py`
- Create: `tests/test_gui_voiceprint_identification.py`
- Modify: `aTrain/app.py`
- Modify: `aTrain/components/layout/sidebar.py`
- Modify: `aTrain/utils/transcription.py`
- Modify: `README.md`

- [ ] **Step 1: Start PR 3 independently**

PR 3 does not depend on PR 1 or PR 2. It can be prepared and opened directly against `upstream/develop` while the CLI PR stack is under review.

- [ ] **Step 2: Refresh upstream develop**

Run in `D:\Git\aTrain`:

```powershell
rtk git fetch upstream develop
```

- [ ] **Step 3: Create PR 3 worktree**

Run:

```powershell
rtk git worktree add -b codex-atrain-gui-voiceprints D:\Git\aTrain-prs\03-gui-vp upstream/develop
```

- [ ] **Step 4: Copy GUI voiceprint files from the local integration branch**

Run in `D:\Git\aTrain-prs\03-gui-vp`:

```powershell
rtk git checkout d6b5def4 -- aTrain/voiceprints.py aTrain/voiceprint_identification.py aTrain/utils/voiceprints.py aTrain/components/voiceprints/enroll_dialog.py aTrain/pages/voiceprints.py tests/test_voiceprints.py tests/test_voiceprint_identification.py tests/test_gui_voiceprint_identification.py
```

- [ ] **Step 5: Manually apply only GUI integration hunks**

Edit these files:

```text
aTrain/app.py
aTrain/components/layout/sidebar.py
aTrain/utils/transcription.py
README.md
```

Required `aTrain/app.py` result: preserve upstream `FLATPAK` storage path and conditional `wakepy`, and add `voiceprints` to the page import:

```python
from aTrain.pages import about, archive, faq, models, transcribe, voiceprints  # noqa
```

Required sidebar result: add one navigation entry for the voiceprint page, matching existing sidebar style:

```python
("Voiceprints", "record_voice_over", "/voiceprints")
```

Required GUI transcription result: when `speaker_detection` is enabled and local voiceprint profiles exist, capture speaker embeddings, assign enrolled names, rewrite speaker labels, and recreate output files. When no profiles exist, the behavior must match upstream transcription.

- [ ] **Step 6: Keep CLI voiceprint out of PR 3**

Run:

```powershell
rtk rg --encoding utf-8 -n "voiceprint enroll|speaker-embeddings-output|identify-speakers|voiceprint-threshold|voiceprint-margin" aTrain tests README.md
```

Expected: no CLI command/option matches. GUI page labels and module names may appear.

- [ ] **Step 7: Run PR 3 tests**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest tests.test_voiceprints tests.test_voiceprint_identification tests.test_gui_voiceprint_identification -v
rtk git diff --check
```

Expected: tests pass and `git diff --check` prints no output.

- [ ] **Step 8: Commit PR 3**

Run:

```powershell
rtk git add aTrain/app.py aTrain/components/layout/sidebar.py aTrain/utils/transcription.py aTrain/voiceprints.py aTrain/voiceprint_identification.py aTrain/utils/voiceprints.py aTrain/components/voiceprints/enroll_dialog.py aTrain/pages/voiceprints.py README.md tests/test_voiceprints.py tests/test_voiceprint_identification.py tests/test_gui_voiceprint_identification.py
rtk git commit -m "Add GUI speaker voiceprints"
```

- [ ] **Step 9: Push PR 3 branch**

Run:

```powershell
rtk git push -u origin codex-atrain-gui-voiceprints
```

Open a draft PR:

```text
base: JuergenFleiss/aTrain:develop
compare: phoenixray2000:codex-atrain-gui-voiceprints
title: Add GUI speaker voiceprints
```

PR body checklist:

```markdown
## Summary
- adds local speaker voiceprint profiles
- adds a GUI page for enrolling/removing profiles
- relabels diarized GUI transcription output when a confident voiceprint match exists
- keeps transcription unchanged when no voiceprints are enrolled

## Tests
- `python -m unittest tests.test_voiceprints tests.test_voiceprint_identification tests.test_gui_voiceprint_identification -v`
- `git diff --check`
```

---

## Task 5: PR 4 - CLI Voiceprint Enrollment And Speaker Embedding Export

**Files:**
- Create: `aTrain/voiceprint_cli.py`
- Create: `tests/test_voiceprint_cli.py`
- Create: `tests/test_cli_voiceprints.py`
- Modify: `aTrain/cli.py`
- Modify: `README.md`

- [ ] **Step 1: Choose PR 4 strategy**

PR 4 depends on PR 1's CLI base and PR 3's voiceprint core. It does not depend on PR 2.

Recommended while PR 1 and PR 3 are still under review: create a temporary fork-only base branch that merges PR 1 and PR 3, then stack PR 4 on that branch. Retarget PR 4 to `JuergenFleiss/aTrain:develop` only after PR 1 and PR 3 merge upstream.

If PR 1 and PR 3 have already merged, base PR 4 directly on `upstream/develop`.

- [ ] **Step 2: Refresh upstream develop**

Run in `D:\Git\aTrain`:

```powershell
rtk git fetch upstream develop
```

- [ ] **Step 3: Create PR 4 worktree**

If PR 1 and PR 3 are still open, create the temporary combined base first:

```powershell
rtk git branch -f codex-atrain-cli-voiceprint-base codex-atrain-gui-voiceprints
rtk git worktree add D:\Git\aTrain-prs\04-cli-vp-base codex-atrain-cli-voiceprint-base
```

Run in `D:\Git\aTrain-prs\04-cli-vp-base`:

```powershell
rtk git merge --no-ff codex-atrain-cli-transcribe -m "Merge CLI base for CLI voiceprint stack"
rtk git push -u origin codex-atrain-cli-voiceprint-base
```

Then create PR 4 from the combined base:

```powershell
rtk git worktree add -b codex-atrain-cli-voiceprint-enrollment D:\Git\aTrain-prs\04-cli-vp codex-atrain-cli-voiceprint-base
```

If PR 1 and PR 3 have already merged, run instead:

```powershell
rtk git worktree add -b codex-atrain-cli-voiceprint-enrollment D:\Git\aTrain-prs\04-cli-vp upstream/develop
```

- [ ] **Step 4: Copy CLI voiceprint helper and tests from local integration branch**

Run in `D:\Git\aTrain-prs\04-cli-vp`:

```powershell
rtk git checkout d6b5def4 -- aTrain/voiceprint_cli.py tests/test_voiceprint_cli.py tests/test_cli_voiceprints.py
```

- [ ] **Step 5: Apply CLI voiceprint hunks to `aTrain/cli.py`**

Edit `aTrain/cli.py` so it includes:

```python
voiceprint_cli = typer.Typer(help="Manage speaker voiceprints.", no_args_is_help=True)
cli.add_typer(voiceprint_cli, name="voiceprint")
```

Add `voiceprint enroll` with these mutually exclusive sources:

```text
--audio PATH
--speaker-embeddings PATH --speaker SPEAKER_XX
```

Add transcription options:

```text
--identify-speakers / --no-identify-speakers
--voiceprint-threshold FLOAT
--voiceprint-margin FLOAT
--speaker-embeddings-output PATH
```

Required validation behavior:

```python
if identify_speakers and not speaker_detection:
    raise ValueError("--identify-speakers requires --speaker-detection.")
if speaker_embeddings_output and len(inputs) != 1:
    raise ValueError("--speaker-embeddings-output supports single-file input only.")
if speaker_embeddings_output and not speaker_detection:
    raise ValueError("--speaker-embeddings-output requires --speaker-detection.")
if speaker_embeddings_output and not identify_speakers:
    raise ValueError("--speaker-embeddings-output requires --identify-speakers.")
```

- [ ] **Step 6: Preserve review fixes from `d6b5def4`**

Check `aTrain/voiceprint_cli.py` has:

```python
check_model_downloaded("speaker-detection")
model_path = get_model("speaker-detection")
```

in that order for audio enrollment.

Check device mapping uses `torch.device`:

```python
return torch.device("cuda" if device == Device.GPU else "cpu")
```

Check unavailable GPU raises before extraction:

```python
if device == Device.GPU and not torch.cuda.is_available():
    raise RuntimeError("GPU was requested but CUDA is not available.")
```

Check `_copy_speaker_embeddings()` takes and uses `overwrite`:

```python
def _copy_speaker_embeddings(
    staging_dir: Path, file_id: str, output_path: Path, overwrite: bool = True
) -> Path:
    source = staging_dir / file_id / CAPTURE_FILENAME
    if not source.exists():
        raise FileNotFoundError(
            "Speaker embeddings were not captured. Use --speaker-detection and --identify-speakers."
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Target file exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    return output_path
```

- [ ] **Step 7: Update README CLI voiceprint docs**

Add one concise section with these examples:

```powershell
aTrain-cli voiceprint enroll --name "李想" --audio "D:\samples\li-xiang.wav" --update
```

```powershell
aTrain-cli transcribe "D:\input\meeting.wav" --speaker-detection --identify-speakers --speaker-embeddings-output "D:\out\meeting.speaker-embeddings.npz"
aTrain-cli voiceprint enroll --name "李想" --speaker-embeddings "D:\out\meeting.speaker-embeddings.npz" --speaker SPEAKER_01 --update
```

- [ ] **Step 8: Run PR 4 tests**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest tests.test_voiceprint_cli tests.test_cli_voiceprints tests.test_voiceprints tests.test_voiceprint_identification tests.test_cli_paths tests.test_cli_vocabulary tests.test_transcription_hotwords -v
rtk .\.venv\Scripts\python.exe -m aTrain.cli voiceprint --help
rtk .\.venv\Scripts\python.exe -m aTrain.cli voiceprint enroll --help
rtk git diff --check
```

Expected:
- unit tests pass
- both help commands exit `0`
- `git diff --check` prints no output

- [ ] **Step 9: Commit PR 4**

Run:

```powershell
rtk git add aTrain/cli.py aTrain/voiceprint_cli.py README.md tests/test_voiceprint_cli.py tests/test_cli_voiceprints.py
rtk git commit -m "Add CLI voiceprint enrollment"
```

- [ ] **Step 10: Push PR 4 branch**

Run:

```powershell
rtk git push -u origin codex-atrain-cli-voiceprint-enrollment
```

Open a draft PR. If PR 1 and PR 3 are still open, target the temporary fork-only base:

```text
base: phoenixray2000/aTrain:codex-atrain-cli-voiceprint-base
compare: phoenixray2000:codex-atrain-cli-voiceprint-enrollment
title: Add CLI voiceprint enrollment and speaker embedding export
```

After PR 1 and PR 3 merge, rebase PR 4 onto `upstream/develop` and retarget it:

```text
base: JuergenFleiss/aTrain:develop
compare: phoenixray2000:codex-atrain-cli-voiceprint-enrollment
title: Add CLI voiceprint enrollment and speaker embedding export
```

PR body checklist:

```markdown
## Summary
- adds `aTrain-cli voiceprint enroll`
- supports enrolling from audio or captured speaker embeddings
- lets CLI transcription identify diarized speakers using enrolled profiles
- lets single-file transcription export captured speaker embeddings for later enrollment

## Tests
- `python -m unittest tests.test_voiceprint_cli tests.test_cli_voiceprints tests.test_voiceprints tests.test_voiceprint_identification tests.test_cli_paths tests.test_cli_vocabulary tests.test_transcription_hotwords -v`
- `python -m aTrain.cli voiceprint --help`
- `python -m aTrain.cli voiceprint enroll --help`
- `git diff --check`
```

---

## Task 6: PR Review And Merge Discipline

**Files:**
- Modify only files in the active PR worktree.

- [ ] **Step 1: Never fix review comments in `D:\Git\aTrain`**

For each requested change, switch to the relevant worktree:

```powershell
rtk git -C D:\Git\aTrain-prs\01-cli status --short --branch
rtk git -C D:\Git\aTrain-prs\02-vocab status --short --branch
rtk git -C D:\Git\aTrain-prs\03-gui-vp status --short --branch
rtk git -C D:\Git\aTrain-prs\04-cli-vp status --short --branch
```

Only edit the matching worktree branch.

- [ ] **Step 2: Keep each PR branch rebased on latest upstream develop before final review**

Run in the active worktree:

```powershell
rtk git fetch upstream develop
rtk git rebase upstream/develop
```

Expected: rebase exits `0`. If conflicts occur, resolve only inside that PR's files and rerun that PR's test bundle.

- [ ] **Step 3: Force-push only with lease after rebase**

Run:

```powershell
rtk git push --force-with-lease origin <branch-name>
```

Replace `<branch-name>` with the active branch name from this plan.

- [ ] **Step 4: After a PR merges, refresh the main checkout refs without changing its branch**

Run in `D:\Git\aTrain`:

```powershell
rtk git fetch upstream develop
rtk git status --short --branch
```

Expected: current branch remains `codex-speaker-voiceprint-identification`.

---

## Task 7: Optional Local Integration Refresh After All PRs Merge

**Files:**
- Modify local integration branch only if the user explicitly asks to refresh local use branch.

- [ ] **Step 1: Confirm all PRs are merged**

Run:

```powershell
rtk git fetch upstream develop
rtk git log --oneline --max-count=20 upstream/develop
```

Expected: upstream history includes the four PR merge commits or equivalent squash commits.

- [ ] **Step 2: Create a safety branch before touching local integration**

Run in `D:\Git\aTrain`:

```powershell
rtk git branch codex-atrain-local-before-upstream-refresh
```

Expected: branch creation exits `0`.

- [ ] **Step 3: Decide refresh mode**

Use one of these modes:

```text
Mode A: Keep current integration branch exactly as-is for local use.
Mode B: Create a new local branch from upstream/develop and cherry-pick only unmerged private changes.
Mode C: Rebase codex-speaker-voiceprint-identification onto upstream/develop after all PRs merge.
```

Recommended mode: `Mode B`, because it leaves the current branch recoverable and avoids rewriting a branch used by local tools.

---

## Self-Review

**Spec coverage:** The plan covers upstream baseline handling, current branch preservation, PR splitting, PR order, branch/worktree names, files per PR, exclusion list, verification commands, push/PR metadata, and review workflow.

**Placeholder scan:** No `TBD`, `TODO`, `fill in later`, or unspecified test command remains. Conflict cases have concrete preservation rules.

**Type consistency:** Branch names, file names, helper names, and test module names match the current local code where those files exist. PR 1 intentionally uses a reduced `_transcribe_one()` test call because later vocabulary and voiceprint parameters do not exist in that PR.

---

## Execution Choice

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - execute tasks in this session using `superpowers:executing-plans`, with checkpoints before each PR push.
