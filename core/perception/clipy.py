import torch
import clip
import numpy as np
from PIL import Image

class CLIPModel:
    def __init__(self, model_name="ViT-B/32", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print("loaded model correctly")
        self.model.eval()
        print("in evaluation")

    def encode_text(self, text_list):
        tokens = clip.tokenize(text_list).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, pil_image):
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def score(self, pil_image, text_list):
        image_features = self.encode_image(pil_image)
        text_features = self.encode_text(text_list)

        # Cosine similarity (already normalised)
        similarity = (image_features @ text_features.T).squeeze(0)  # shape: (n_texts,)
        # Convert to probabilities via softmax (temperature from CLIP is implicit)
        probs = similarity.softmax(dim=-1).cpu().numpy()
        return probs

    def top_match(self, pil_image, text_list, k=1):
        probs = self.score(pil_image, text_list)
        indices = np.argsort(probs)[::-1][:k]
        return [(text_list[i], probs[i]) for i in indices]

    def top_score(self, pil_image, text_list):
        probs = self.score(pil_image, text_list)
        if len(probs) == 0:
            return None, 0.0
        best_idx = int(np.argmax(probs))
        return text_list[best_idx], float(probs[best_idx])
