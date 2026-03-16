"""
BASNet Wrapper for Salient Object Detection
Integrates with your PerceptionPipeline.
Handles both the official repo and your local file structure.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# --- CONFIGURABLE PATHS ---
BASNET_REPO_PATH = "C:/Users/ALIENWARE/Desktop/cloned repos/BASNet"  # Path to cloned repo
BASNET_MODEL_PATH = os.path.join(BASNET_REPO_PATH, "saved_models", "basnet_bsi", "basnet.pth")
# ---------------------------

sys.path.insert(0, BASNET_REPO_PATH)

# Flexible import: try both possible module names
BASNetModel = None
import sys
sys.path.insert(0, r"C:/Users/ALIENWARE/Desktop/cloned repos/BASNet")
from model import BASNet
BASNetModel = BASNet

class BASNetSaliency:

    def __init__(self, device=None, model_path=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_path = model_path or BASNET_MODEL_PATH

        print(f"Loading BASNet model from {self.model_path} on {self.device}...")
        self.net = BASNetModel(3, 1)  # 3 input channels, 1 output
        self._load_pretrained()
        self.net.to(self.device)
        self.net.eval()

        # Standard preprocessing used in BASNet training
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_pretrained(self):
        print(f"Loading weights from {self.model_path}")
        state_dict = torch.load(self.model_path, map_location=self.device)

        # Remove 'module.' prefix if model was saved with DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict)

    @torch.no_grad()
    def get_saliency_map(self, image_pil):
        # Preprocess
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        # Forward pass – BASNet may return multiple outputs
        outputs = self.net(input_tensor)

        # Take the first output as the final saliency map
        if isinstance(outputs, (tuple, list)):
            pred = outputs[0]   # d0 – the refined saliency map
        else:
            pred = outputs

        pred = torch.sigmoid(pred)  # shape (1, 1, 256, 256) or similar

        # Resize to original image size
        orig_w, orig_h = image_pil.size
        pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=True)

        sal_map = pred.squeeze().cpu().numpy()  # (H, W) float32
        return sal_map

    def get_saliency_score(self, image):

        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        sal_map = self.get_saliency_map(image_pil)
        return float(np.mean(sal_map))


# Quick standalone test
if __name__ == "__main__":
    import cv2
    import sys

    sal = BASNetSaliency()
    img_path = "C:/Users/ALIENWARE/Unity/Spatial_Navigation_proj/core/perception/Images/img_000034.png"
    img = cv2.imread(img_path) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    score = sal.get_saliency_score(img_rgb)
    print(f"Saliency score: {score:.4f}")

    # Show saliency map
    sal_map = sal.get_saliency_map(Image.fromarray(img_rgb))
    sal_map_img = (sal_map * 255).astype(np.uint8)
    cv2.imshow("Original", img)               # OpenCV expects BGR, so show original BGR image
    cv2.imshow("Saliency Map", sal_map_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Ask if user wants to save the saliency map
    save = input("Save saliency map? (y/n): ").strip().lower()
    if save == 'y':
        # Generate default filename based on input image
        base = os.path.splitext(os.path.basename(img_path))[0]
        default_name = f"{base}_saliency.png"
        save_path = input(f"Enter filename (default: {default_name}): ").strip()
        if not save_path:
            save_path = default_name
        # Ensure the directory exists (if path includes folders)
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_path, sal_map_img)
        print(f"Saliency map saved to {save_path}")