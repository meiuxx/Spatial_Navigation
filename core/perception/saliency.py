import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

BASNET_REPO_PATH = "C:/Users/ALIENWARE/Desktop/cloned repos/BASNet"  # Path to cloned repo
BASNET_MODEL_PATH = "C:/Users/ALIENWARE/Desktop/cloned repos/BASNet/saved_models/basnet_bsi/basnet.pth"

sys.path.insert(0, r"C:/Users/ALIENWARE/Desktop/cloned repos/BASNet")
from model import BASNet
BASNetModel = BASNet

class BASNetSaliency:

    def __init__(self):
        self.device="cuda"
        self.model_path = BASNET_MODEL_PATH

        print(f"loading BASNet model from {self.model_path} on {self.device}...")
        self.net = BASNetModel(3, 1)  # 3 input channels, 1 output
        self._load_pretrained()
        self.net.to(self.device)
        self.net.eval()

        # standard preprocessing used in BASNet training
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) #those are ImageNet mean and std values (check online)
        ])

    def _load_pretrained(self):
        print(f"Loading weights from {self.model_path}")
        state_dict = torch.load(self.model_path, map_location=self.device)

        # remove 'module.' prefix if model was saved with torch.nn.DataParallel (multple GPUs)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict)

    @torch.no_grad()
    def get_saliency_map(self, image_pil):
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        outputs = self.net(input_tensor)
        
        # we dont know the format of the output
        # if it's of type list or tuple, take first instance, esle keeep as is
        raw_logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        # min-max normalization
        map_min = torch.min(raw_logits)
        map_max = torch.max(raw_logits)
        norm_map = (raw_logits - map_min) / (map_max - map_min + 1e-8)

        # resize and convert
        orig_w, orig_h = image_pil.size
        norm_map = F.interpolate(norm_map, size=(orig_h, orig_w), mode='bilinear')
        
        #convert it to 2d, move tensor to cpu, convert to numpy array
        return norm_map.squeeze().cpu().numpy()
    
    def get_saliency_score(self, image):

        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        sal_map = self.get_saliency_map(image_pil)
        return float(np.mean(sal_map))