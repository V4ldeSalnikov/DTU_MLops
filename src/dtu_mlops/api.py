import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from medmnist import INFO
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_model() -> torch.nn.Module:#For now pretrained model as I don't have a trained model available
    """Load pretrained ResNet18 model."""
    #loading  model
    model = models.resnet18(pretrained=True)
    #setting model to evaluation mode
    model.eval()
    return model.to(DEVICE)

# Image preprocessing pipeline (basic so far, can be improved)
def get_preprocessing_pipeline() -> transforms.Compose:
    """Get preprocessing pipeline for images."""
    #getting information on number of image channels (RGB or Grayscale) for trained model
    info = INFO["organamnist"]  # Using organamnist as reference
    output_channels = info["n_channels"] # RGB or Grayscale
    #chosing 'standard' mean and std values for normalization if dataset statistics are not available
    mean = (0.5,) * output_channels
    std = (0.5,) * output_channels
    #preparing transformation pipeline
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    #returning the transformation pipeline
    return trans
def get_class_labels(data_flag: str = "organamnist") -> list[str]:
    """Get class labels for MedMNIST dataset."""
    #retrieving dataset info
    info = INFO[data_flag]
    labels = info["label"]
    #returning class labels
    return labels

def classify_images(images) -> list[str]:
    """Classify images using pretrained ResNet18 model."""
    #
    if images is None:
        return []
    # Handle single image passed as string
    if isinstance(images, str):
        images = [images]
    #
    predis = []
    #
    for image in images:
        #getting image to classify
        image = Image.open(image).convert("RGB")#as pretrained resnet expects 3 channel input for now it needs to be converted to RGB (remove when loading of trained model changed)
        input_tensor = preprocess(image).unsqueeze(0)
        #classyfing image
        with torch.no_grad():
            output = model(input_tensor)#raw logits from model
            probs = torch.nn.functional.softmax(output[0], dim=0)#converting them to probabilities
            top_class = probs.argmax().item()#getting highest probability class index
        #
        if(top_class > 10):
            top_class = 5
        #
        filename = Path(image).name
        #getting label based on predicted index
        label = class_labels[str(top_class)]
        #
        predis.append(label)
    #
    return predis

model = load_model()
preprocess = get_preprocessing_pipeline()
class_labels = get_class_labels()

with gr.Blocks() as demo:
    gr.Markdown("# ResNet18 Image Classifier")
    
    with gr.Row():
        images_input = gr.File(file_count="multiple", file_types=["image"], label="Upload Images")
        submit_btn = gr.Button("Classify", scale=0)
    
    gallery = gr.Gallery(label="Results", columns=3, rows=2, height="auto")
    
    def process(files):
        if not files:
            return []
        
        results = []
        for f in files:
            img = Image.open(f.name).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top_class = probs.argmax().item()
            
            if top_class > 10:
                top_class = 5
            
            label = class_labels[str(top_class)]
            filename = Path(f.name).name
            results.append((img, f"{filename}\n{label}"))
        
        return results
    
    submit_btn.click(process, inputs=images_input, outputs=gallery)



if __name__ == "__main__":
    demo.launch()