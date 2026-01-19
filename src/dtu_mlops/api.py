import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from medmnist import INFO

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

def classify_image(image) -> str:
    """Classify image using pretrained ResNet18 model."""
   



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
    #getting label based on predicted index
    label = class_labels[str(top_class)]
    

    return label


model = load_model()
preprocess = get_preprocessing_pipeline()
class_labels = get_class_labels()

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath", label="Upload Images"),
    outputs=gr.Label(num_top_classes=1),
    title="ResNet18 Image Classifier",
    description="Upload one or more images and get predictions from a pretrained ResNet18 model.",
)

if __name__ == "__main__":
    interface.launch()