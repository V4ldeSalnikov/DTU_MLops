import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from medmnist import INFO
from pathlib import Path
import base64
from io import BytesIO
import os

from dtu_mlops.model import resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_model(model_path: str | Path = "models/resnet18_best.pth") -> torch.nn.Module:
    """Load trained ResNet18 model."""
    #getting model path
    model_path = Path(model_path)
    #checking if model file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    #loading  model
    model = resnet18(num_classes=11, in_channels=1)
    #Loading model checkpoint (assumption that it is checkpoint with additional metadata over weights)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    #Extracting model weights from checkpoint(or not )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:#if it is not only weights but full checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])
    else:#if it is just weights
        model.load_state_dict(checkpoint)
    #setting model to evaluation mode
    model.eval()
    #returning the model on appropriate device
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
    # Handle case with no images
    if images is None:
        return "<p>No images uploaded</p>"
    # Ensure images is a list if(case when only one image is uploaded is problematic without it)
    if isinstance(images, str):
        images = [images]
    #creating HTML structure for results
    html = "<div style='display: flex; flex-wrap: wrap; gap: 30px; padding: 20px; justify-content: center;'>"
    #loop over images and classify them
    for image_path in images:
        #preparing image for classification
        img = Image.open(image_path).convert("L")  # Convert to grayscale (as project uses grayscale images)
        input_tensor = preprocess(img).unsqueeze(0)
        #forward pass + softmax to get probabilities
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top_class = probs.argmax().item()
        #getting class label
        label = class_labels[str(top_class)]
        #getting image filename
        filename = Path(image_path).name
        #Preparing image for embedding in HTML (base64 encoding)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        #adding current image block to HTML
        html += f"""
        <div style='border: 2px solid #ddd; padding: 15px; border-radius: 8px; background: #f9f9f9; width: 280px;'>
            <p style='font-size: 14px; color: #666; margin: 0 0 10px 0; text-align: center; font-weight: bold;'>{filename}</p>
            <img src='data:image/jpeg;base64,{img_str}' style='width: 250px; height: 250px; object-fit: contain; display: block; margin: 0 auto 10px;'>
            <p style='font-size: 18px; color: #0066cc; margin: 10px 0 0 0; text-align: center; font-weight: bold;'>{label}</p>
        </div>
        """
    #closing HTML container
    html += "</div>"
    #returning results
    return html

###main code to launch Gradio app###

#prepare model and preprocessing pipeline (kind of backend)
model = load_model()
preprocess = get_preprocessing_pipeline()
class_labels = get_class_labels()
#preparing Gradio interface (frontend)
with gr.Blocks() as demo:
    #app "title"
    gr.Markdown("<h1 style='text-align: center;'> MLOps project - MedMNIST dataset Image Classifier</h1>")
    #app spine layout
    with gr.Column():
        #title of load segment
        gr.Markdown("<h2 style='text-align: center;'> Upload Images</h2>")
        #images loading component
        images_input = gr.File(file_count="multiple", file_types=["image"], label="Upload Images")
        #buttons row for app functionality
        with gr.Row():
            submit_btn = gr.Button("Classify")
            reset_btn = gr.Button("Reset")
        #title of results segment
        gr.Markdown("<h2 style='text-align: center;'> Results</h2>")
        #classification results output component
        output = gr.HTML(label="Results")
    #getting callable reset function
    def reset():
        return None, ""
    #linking buttons to functions
    submit_btn.click(classify_images, inputs=images_input, outputs=output)
    reset_btn.click(reset, outputs=[images_input, output])
#when file is run directly, launch the app
if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.launch(server_name=server_name)
