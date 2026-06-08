# clip_scene_classifier.py
#
# Full pipeline for CLIP-based hospital scene classification.
#
# Step 1 — Extract CLIP embeddings from your folder-organised dataset
# Step 2 — Evaluate with 5-fold cross-validation (reliable accuracy estimate)
# Step 3 — Train final classifier on ALL data for deployment
# Step 4 — Save the classifier for use in your agent pipeline
#
# Expected folder structure:
#   dataset/
#       lobby/
#           img001.jpg
#       corridor/
#           img002.jpg
#       ...
#
# Usage:
#   python clip_scene_classifier.py --data dataset/ --output scene_classifier.pkl
#
# Then in globals.py:
#   from clip_scene_classifier import SceneClassifier
#   scene_classifier = SceneClassifier.load("scene_classifier.pkl")
#   label, confidence = scene_classifier.predict(pil_image)

import os
import argparse
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import clip

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ── Config ────────────────────────────────────────────────────────────────────

CLIP_MODEL  = "ViT-B/32"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42
LR_C        = 0.1
LR_MAX_ITER = 2000


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_embeddings(data_dir, device):
    data_dir = Path(data_dir)
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    embeddings, labels, paths = [], [], []
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        raise ValueError(f"No subfolders found in {data_dir}.")

    print(f"\nFound {len(class_dirs)} classes:")
    for d in class_dirs:
        imgs = [f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        print(f"  {d.name:<35} {len(imgs)} images")

    print(f"\nExtracting CLIP embeddings ({CLIP_MODEL}) ...")

    for class_dir in class_dirs:
        class_name = class_dir.name
        img_files  = [f for f in class_dir.iterdir()
                      if f.suffix.lower() in IMAGE_EXTS]
        for img_path in img_files:
            try:
                img = Image.open(img_path).convert("RGB")
                inp = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(inp)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy().flatten())
                labels.append(class_name)
                paths.append(str(img_path))
            except Exception as e:
                print(f"  [WARN] Skipping {img_path.name}: {e}")

    print(f"Extracted {len(embeddings)} embeddings across {len(set(labels))} classes.\n")
    return np.array(embeddings), labels, paths


# ── Training + Evaluation ─────────────────────────────────────────────────────

def train(embeddings, labels):
    """
    Evaluate with 5-fold stratified cross-validation, then train a final
    classifier on ALL data for deployment.
    Returns (classifier, label_encoder).
    """
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    class_names = le.classes_

    clf_proto = LogisticRegression(
        C            = LR_C,
        max_iter     = LR_MAX_ITER,
        class_weight = 'balanced',
        random_state = RANDOM_SEED,
        solver       = 'lbfgs',
    )

    # ── 5-fold cross-validation ───────────────────────────────────────────────
    # With small per-class counts a single 80/20 split gives noisy results
    # (only 2 test samples per class). 5-fold means every image gets to be
    # in the test set once, giving a much more reliable accuracy estimate.
    print("Running 5-fold stratified cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    y_pred_cv = cross_val_predict(clf_proto, embeddings, y, cv=cv)

    print("\n── Per-class results (5-fold CV) ────────────────────────────────")
    print(classification_report(y, y_pred_cv,
                                target_names=class_names,
                                zero_division=0))

    acc = (y_pred_cv == y).mean()
    print(f"Overall CV accuracy: {acc*100:.1f}%")

    print("\n── Per-class accuracy ───────────────────────────────────────────")
    for i, cls in enumerate(class_names):
        mask    = y == i
        cls_acc = (y_pred_cv[mask] == y[mask]).mean()
        n       = mask.sum()
        bar     = "█" * int(cls_acc * 20)
        print(f"  {cls:<35} {cls_acc*100:5.1f}%  {bar}  (n={n})")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred_cv)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = 'd',
        cmap        = 'Blues',
        xticklabels = class_names,
        yticklabels = class_names,
        ax          = ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Scene Classifier — 5-Fold CV Confusion Matrix  "
                 f"(overall acc: {acc*100:.1f}%)", fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nConfusion matrix saved -> confusion_matrix.png")

    # ── Final classifier trained on ALL data ──────────────────────────────────
    print("\nTraining final classifier on all data ...")
    clf_final = LogisticRegression(
        C            = LR_C,
        max_iter     = LR_MAX_ITER,
        class_weight = 'balanced',
        random_state = RANDOM_SEED,
        solver       = 'lbfgs',
    )
    clf_final.fit(embeddings, y)
    print("Final classifier ready.")

    return clf_final, le


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_classifier(clf, le, output_path):
    payload = {"classifier": clf, "label_encoder": le, "clip_model": CLIP_MODEL}
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Classifier saved -> {output_path}")


# ── Inference wrapper ─────────────────────────────────────────────────────────

class SceneClassifier:
    """
    Drop-in classifier for the agent pipeline.

    Usage in globals.py:
        from clip_scene_classifier import SceneClassifier
        scene_classifier = SceneClassifier.load("scene_classifier.pkl")

    Usage in process_message():
        label, confidence = scene_classifier.predict(pil_img)
        top3 = scene_classifier.predict_topk(pil_img, k=3)
    """

    def __init__(self, clf, le, clip_model_name=CLIP_MODEL):
        self.clf         = clf
        self.le          = le
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self._model      = None
        self._preprocess = None
        self._model_name = clip_model_name

    def _ensure_model(self):
        if self._model is None:
            self._model, self._preprocess = clip.load(
                self._model_name, device=self.device
            )
            self._model.eval()

    def _embed(self, pil_image):
        self._ensure_model()
        inp = self._preprocess(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._model.encode_image(inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def predict(self, pil_image):
        """Returns (label: str, confidence: float)."""
        emb   = self._embed(pil_image)
        probs = self.clf.predict_proba(emb)[0]
        idx   = int(np.argmax(probs))
        return self.le.classes_[idx], float(probs[idx])

    def predict_topk(self, pil_image, k=3):
        """Returns list of (label, confidence) sorted by confidence desc."""
        emb   = self._embed(pil_image)
        probs = self.clf.predict_proba(emb)[0]
        top_k = np.argsort(probs)[::-1][:k]
        return [(self.le.classes_[i], float(probs[i])) for i in top_k]

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return SceneClassifier(
            clf             = payload["classifier"],
            le              = payload["label_encoder"],
            clip_model_name = payload.get("clip_model", CLIP_MODEL),
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train CLIP scene classifier")
    parser.add_argument("--data",   required=True,
                        help="Path to dataset folder (one subfolder per class)")
    parser.add_argument("--output", default="scene_classifier.pkl",
                        help="Output path for saved classifier")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embeddings, labels, _ = extract_embeddings(args.data, device)
    clf, le = train(embeddings, labels)
    save_classifier(clf, le, args.output)

    print("\nDone. To use in your agent:")
    print("  from clip_scene_classifier import SceneClassifier")
    print(f'  scene_classifier = SceneClassifier.load("{args.output}")')
    print('  label, confidence = scene_classifier.predict(pil_image)')


if __name__ == "__main__":
    main()