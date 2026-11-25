"""Build native ComposePost binaries from a staged Cargo workspace."""

import argparse
from pathlib import Path
import shutil
import subprocess

LEAVES = (
    "compose-creator-with-userid",
    "media-service",
    "social-graph-get-followers",
    "store-post",
    "unique-id-service",
    "url-shorten-service",
    "user-mention-service",
    "write-user-timeline",
)
CALLERS = ("compose-post", "text-service", "write-home-timeline")
SDK_DEPENDENCY = 'OpenFaaSRPC = { path = "../OpenFaaSRPC" }'
LEAF_SDK_DEPENDENCY = 'OpenFaaSRPC = { path = "../OpenFaaSRPC", default-features = false }'


def disable_leaf_rpc(workspace):
    manifests = {}
    for name in LEAVES:
        path = workspace / name / "Cargo.toml"
        lines = path.read_text().splitlines(keepends=True)
        entries = [line for line in lines if line.startswith("OpenFaaSRPC =")]
        if len(entries) != 1 or entries[0].strip() not in (SDK_DEPENDENCY, LEAF_SDK_DEPENDENCY):
            raise ValueError(f"Unexpected OpenFaaSRPC dependency in {path}")
        manifests[path] = "".join(
            line.replace(SDK_DEPENDENCY, LEAF_SDK_DEPENDENCY) if line == entries[0] else line
            for line in lines
        )

    for path, contents in manifests.items():
        if path.read_text() != contents:
            path.write_text(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="disposable native workspace with a Cargo.lock")
    parser.add_argument("target", type=Path, help="Cargo target directory")
    parser.add_argument("output", type=Path, help="directory for native executables")
    args = parser.parse_args()
    workspace, target, output = (path.resolve() for path in (args.workspace, args.target, args.output))
    disable_leaf_rpc(workspace)

    # Cargo unifies features across packages selected in a single build.
    for name in (*LEAVES, *CALLERS):
        subprocess.run(
            ["cargo", "build", "--release", "--locked", "--target-dir", str(target), "-p", name],
            cwd=workspace,
            check=True,
        )

    output.mkdir(parents=True, exist_ok=True)
    for name in (*LEAVES, *CALLERS):
        shutil.copy2(target / "release" / name, output / name)


if __name__ == "__main__":
    main()
