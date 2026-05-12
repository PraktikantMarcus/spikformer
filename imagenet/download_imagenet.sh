#!/bin/bash
# Download and prepare ImageNet (ILSVRC2012) to /nfsscratch/fritzsche/imagenet
#
# Usage:
#   sbash download_imagenet.sh [--user <imagenet_user> --password <imagenet_password>]
#
# Expected tar files:
#   ILSVRC2012_img_train.tar  (~138 GB)
#   ILSVRC2012_img_val.tar    (~6.3 GB)

set -euo pipefail

DEST_DIR="/nfsscratch/fritzsche/imagenet"
TRAIN_DIR="$DEST_DIR/train"
VAL_DIR="$DEST_DIR/val"

# IMAGENET_USER=""
# IMAGENET_PASS=""

# while [[ $# -gt 0 ]]; do
#     case "$1" in
#         --user)     IMAGENET_USER="$2"; shift 2 ;;
#         --password) IMAGENET_PASS="$2"; shift 2 ;;
#         *) echo "Unknown argument: $1"; exit 1 ;;
#     esac
# done

echo "=== ImageNet download & extraction ==="
echo "Destination: $DEST_DIR"
mkdir -p "$DEST_DIR" "$TRAIN_DIR" "$VAL_DIR"

# ── Download ─────────────────────────────────────────────────────────────────
download_tar() {
    local url="$1"
    local out="$2"
    if [[ -f "$out" ]]; then
        echo "  $out already exists, skipping download."
        return
    fi
    # if [[ -z "$IMAGENET_USER" || -z "$IMAGENET_PASS" ]]; then
    #     echo "ERROR: $out not found and no credentials supplied."
    #     echo "  Either place $out in $DEST_DIR or re-run with --user / --password."
    #     exit 1
    # fi
    echo "  Downloading $out ..."
    # wget --user="$IMAGENET_USER" --password="$IMAGENET_PASS" \
    wget --user="Fritzsche" --password="Save" \
         -c -q --show-progress \
         -O "$out" "$url"
}

TRAIN_TAR="$DEST_DIR/ILSVRC2012_img_train.tar"
VAL_TAR="$DEST_DIR/ILSVRC2012_img_val.tar"

download_tar \
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar" \
    "$TRAIN_TAR"

download_tar \
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar" \
    "$VAL_TAR"

# ── Extract training set ──────────────────────────────────────────────────────
# The train tar contains one tar per synset; each synset tar contains the images.
if [[ ! -f "$TRAIN_DIR/.extracted" ]]; then
    echo "Extracting training set (this takes ~30 min) ..."
    tar -xf "$TRAIN_TAR" -C "$TRAIN_DIR"
    for synset_tar in "$TRAIN_DIR"/*.tar; do
        synset=$(basename "$synset_tar" .tar)
        mkdir -p "$TRAIN_DIR/$synset"
        tar -xf "$synset_tar" -C "$TRAIN_DIR/$synset"
        rm "$synset_tar"
    done
    touch "$TRAIN_DIR/.extracted"
    echo "  Training set extraction done."
else
    echo "  Training set already extracted, skipping."
fi

# ── Extract validation set and organise by class ─────────────────────────────
# Requires the official valprep script to map flat filenames to synset folders.
if [[ ! -f "$VAL_DIR/.extracted" ]]; then
    echo "Extracting validation set ..."
    tar -xf "$VAL_TAR" -C "$VAL_DIR"

    echo "Organising validation images into class subdirectories ..."
    # Download the official mapping script from the Soumith mirror
    VALPREP="$DEST_DIR/valprep.sh"
    if [[ ! -f "$VALPREP" ]]; then
        wget -q -O "$VALPREP" \
            "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
    fi
    pushd "$VAL_DIR" > /dev/null
    bash "$VALPREP"
    popd > /dev/null
    touch "$VAL_DIR/.extracted"
    echo "  Validation set organisation done."
else
    echo "  Validation set already extracted, skipping."
fi

# ── Sanity check ─────────────────────────────────────────────────────────────
N_TRAIN=$(find "$TRAIN_DIR" -name "*.JPEG" | wc -l)
N_VAL=$(find "$VAL_DIR"   -name "*.JPEG" | wc -l)
echo ""
echo "=== Done ==="
echo "  Training images : $N_TRAIN  (expected ~1,281,167)"
echo "  Validation images: $N_VAL  (expected 50,000)"
echo "  Data path for training: $DEST_DIR"
