import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from medmnist import INFO
import gradio as gr
import os
import base64
from io import BytesIO

from dtu_mlops.model import resnet18, resnet50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#taken from Mikolaj code with closed PR
def load_model_from_hf(
    repo_id: str,
    filename: str,
    model_type: str,
    num_classes: int,
    in_channels: int,
    device: str,
) -> torch.nn.Module:
    """Load trained model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Model checkpoint filename
        model_type: Type of model ('resnet18' or 'resnet50')
        num_classes: Number of output classes
        in_channels: Number of input channels
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Downloading model from Hugging Face: {repo_id}/{filename}")
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Create model
    if model_type == "resnet18":
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    else:
        model = resnet50(num_classes=num_classes, in_channels=in_channels)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model
#taken from Mikolaj code with closed PR


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
model =  load_model_from_hf(#taken from Mikolaj code with closed PR
            repo_id="mikidzw/organamnist-resnet18",
            filename="resnet18_best.pth",
            model_type="resnet18",
            num_classes=11,
            in_channels=1,
            device=DEVICE,
        )
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