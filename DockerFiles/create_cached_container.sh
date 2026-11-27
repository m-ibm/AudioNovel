#!/usr/bin/env bash
# Snapshot an exited VoxNovel container's writable layer, preserving downloaded
# TTS models, then create a separate idle container with the same project mount.
# The original container is NEVER started, renamed, or deleted.
set -Eeuo pipefail
SOURCE="${1:-voxnovel-audio-only}"
TARGET="${2:-voxnovel-modern}"
SNAPSHOT_IMAGE="${3:-voxnovel-ready:m1-cached}"
if ! docker inspect "$SOURCE" >/dev/null 2>&1; then echo "Source container not found: $SOURCE" >&2; exit 2; fi
if docker inspect "$TARGET" >/dev/null 2>&1; then echo "Target container already exists: $TARGET" >&2; exit 3; fi
if docker image inspect "$SNAPSHOT_IMAGE" >/dev/null 2>&1; then
  echo "Snapshot image tag already exists: $SNAPSHOT_IMAGE" >&2
  echo "Choose a fresh tag as the third argument; this script will not overwrite an image tag." >&2
  exit 3
fi
RUNNING="$(docker inspect -f '{{.State.Running}}' "$SOURCE")"
if [[ "$RUNNING" == "true" ]]; then
  echo "Source container is running. Stop it only after confirming no TTS job is active." >&2
  echo "This script intentionally snapshots stopped containers only." >&2
  exit 4
fi

MOUNTS="$(docker inspect -f '{{range .Mounts}}{{println .Destination}}{{end}}' "$SOURCE")"
CACHE_PATHS=(/root/.local/share/tts /root/booknlp_models /root/.cache/huggingface)
while IFS= read -r mount_path; do
  [[ -n "$mount_path" ]] || continue
  for cache_path in "${CACHE_PATHS[@]}"; do
    if [[ "$cache_path" == "$mount_path" || "$cache_path" == "$mount_path"/* || "$mount_path" == "$cache_path"/* ]]; then
      echo "WARNING: mount '$mount_path' overlaps model-cache path '$cache_path'." >&2
      echo "docker commit excludes mounted-volume contents, so this cache may not be preserved." >&2
      echo "Inspect/copy that volume before proceeding." >&2
      exit 5
    fi
  done
done <<< "$MOUNTS"

printf 'Snapshotting stopped container %s -> image %s\n' "$SOURCE" "$SNAPSHOT_IMAGE"
docker commit "$SOURCE" "$SNAPSHOT_IMAGE" >/dev/null
printf 'Creating idle container %s from snapshot (leaving %s untouched)\n' "$TARGET" "$SOURCE"
docker run -d \
  --name "$TARGET" \
  --cpus=4 \
  -e TORCH_NUM_THREADS=4 \
  -e OMP_NUM_THREADS=4 \
  -v "$HOME/VoxNovel:/VoxNovel" \
  -w /VoxNovel \
  --entrypoint bash \
  "$SNAPSHOT_IMAGE" \
  -lc 'sleep infinity' >/dev/null

echo "Created running container: $TARGET"
echo "Cached model files from the stopped container's writable layer were retained in $SNAPSHOT_IMAGE."
echo "Original container $SOURCE was not started, renamed, or deleted."
echo "Next: bash patched_audiobook/install_modernbooknlp_in_container.sh $TARGET"
