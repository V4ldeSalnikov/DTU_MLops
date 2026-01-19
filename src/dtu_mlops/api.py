import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from medmnist import INFO
from pathlib import Path
import base64
from io import BytesIO

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

def classify_images(images) -> str:
    """Classify images and return formatted HTML with embedded images."""
    if images is None:
        return "<p>No images uploaded</p>"
    
    if isinstance(images, str):
        images = [images]
    
    html = "<div style='display: flex; flex-wrap: wrap; gap: 30px; padding: 20px; justify-content: center;'>"
    
    for image_path in images:
        img = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top_class = probs.argmax().item()
        
        if top_class > 10:
            top_class = 5
        
        label = class_labels[str(top_class)]
        filename = Path(image_path).name
        
        # Convert image to base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        html += f"""
        <div style='border: 2px solid #ddd; padding: 15px; border-radius: 8px; background: #f9f9f9; width: 280px;'>
            <p style='font-size: 14px; color: #666; margin: 0 0 10px 0; text-align: center; font-weight: bold;'>{filename}</p>
            <img src='data:image/jpeg;base64,{img_str}' style='width: 250px; height: 250px; object-fit: contain; display: block; margin: 0 auto 10px;'>
            <p style='font-size: 18px; color: #0066cc; margin: 10px 0 0 0; text-align: center; font-weight: bold;'>{label}</p>
        </div>
        """
    
    html += "</div>"
    return html

model = load_model()
preprocess = get_preprocessing_pipeline()
class_labels = get_class_labels()

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'> MLOps project - MedMNIST dataset Image Classifier</h1>")
    
    with gr.Column():
        gr.Markdown("<h2 style='text-align: center;'> Upload Images</h2>")
        images_input = gr.File(file_count="multiple", file_types=["image"], label="Upload Images")
        
        with gr.Row():
            submit_btn = gr.Button("Classify")
            reset_btn = gr.Button("Reset")
        
        gr.Markdown("<h2 style='text-align: center;'> Results</h2>")
        output = gr.HTML(label="Results")
    
    def reset():
        return None, ""
    
    submit_btn.click(classify_images, inputs=images_input, outputs=output)
    reset_btn.click(reset, outputs=[images_input, output])

if __name__ == "__main__":
    demo.launch()