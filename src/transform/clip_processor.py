import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ClipProcessor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_image_vector(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=["dummy"], images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.squeeze().cpu().numpy().tolist()
        except Exception as e:
            return None

    def get_text_vector(self, text_query):
        try:
            dummy_image = Image.new('RGB', (224, 224), color='black')
            inputs = self.processor(text=[text_query], images=dummy_image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                text_features = outputs.text_embeds
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.squeeze().cpu().numpy().tolist()
        except Exception as e:
            return None