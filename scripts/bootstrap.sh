#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
WORKTREE_DIR="${ROOT_DIR}/.worktree/AniSOAP"
ANISOAP_REPOSITORY="https://github.com/cersonsky-lab/AniSOAP.git"
ANISOAP_REVISION="02aa98a3d4f74c9f637ca166ebe8d6043e0e7b26"

rm -rf "${VENV_DIR}" "${WORKTREE_DIR}"
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}[test]"

mkdir -p "$(dirname "${WORKTREE_DIR}")"
git clone --filter=blob:none --no-checkout "${ANISOAP_REPOSITORY}" "${WORKTREE_DIR}"
git -C "${WORKTREE_DIR}" fetch --depth 1 origin "${ANISOAP_REVISION}"
git -C "${WORKTREE_DIR}" checkout --detach "${ANISOAP_REVISION}"

"${VENV_DIR}/bin/python" - "${WORKTREE_DIR}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]) / "anisoap/representations/ellipsoidal_density_projection.py"
text = source.read_text()
old = "blockidx = pair_ellip_feat.blocks_matching(selection=selection)"
new = "blockidx = [int(index) for index in pair_ellip_feat.keys.select(selection)]"
if text.count(old) != 1:
    raise SystemExit("Pinned AniSOAP compatibility target was not found exactly once.")
source.write_text(text.replace(old, new))
PY

export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
"${VENV_DIR}/bin/python" -m pip install -e "${WORKTREE_DIR}"

SITE_PACKAGES="$("${VENV_DIR}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${WORKTREE_DIR}" > "${SITE_PACKAGES}/anisoap_worktree.pth"

"${VENV_DIR}/bin/python" -c 'import anisoap, anisoap_rust_lib'

printf '\nEnvironment ready. Activate it with:\n  source .venv/bin/activate\n'
