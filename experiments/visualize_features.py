import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms
from transformers import ViTForImageClassification, ViTImageProcessor

RACE_MAPPING = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others",
}

GENDER_MAPPING = {
    0: "Male",
    1: "Female",
}


class ViTLogitsWrapper(nn.Module):
    """Wrapper so that Grad-CAM receives tensor logits instead of Hugging Face output objects for ViT"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor):
        return self.model(pixel_values=input_tensor).logits


def load_resnet_model(checkpoint_path, device):
    model = models.resnet50()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_vgg_model(checkpoint_path, device):
    model = models.vgg16()
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_vit_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model_name = checkpoint_args.get("model_name", "google/vit-base-patch16-224")

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_processor = ViTImageProcessor.from_pretrained(model_name)
    return model, image_processor


def resolve_checkpoint_path(model_type, checkpoint_path):
    if checkpoint_path:
        return checkpoint_path

    defaults = {
        "resnet50": "results/resnet50_best.pt",
        "vgg16": "results/vgg16_best.pt",
        "vit": "results/vit/vit_best.pt",
    }
    if model_type.lower() not in defaults:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported: resnet50, vgg16, vit"
        )
    return defaults[model_type.lower()]


def load_model(checkpoint_path, device, model_type="resnet50"):
    """Load a model checkpoint. Supports resnet50, vgg16, and vit."""
    model_type = model_type.lower()
    if model_type == "resnet50":
        return load_resnet_model(checkpoint_path, device), None
    if model_type == "vgg16":
        return load_vgg_model(checkpoint_path, device), None
    if model_type == "vit":
        return load_vit_model(checkpoint_path, device)
    raise ValueError(f"Unknown model type: {model_type}. Supported: resnet50, vgg16, vit")


def get_eval_transform(image_mean=None, image_std=None):
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )


def preprocess_image(image, transform):
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    processed = transform(image)
    return processed


def denormalize_image(tensor, image_mean=None, image_std=None):
    """Convert normalized tensor back to [0, 1] for visualization."""
    tensor = tensor.clone().detach()
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, image_mean, image_std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.cpu().numpy().transpose(1, 2, 0)


def resize_cam(grayscale_cam, width, height):
    cam_image = Image.fromarray(np.uint8(np.clip(grayscale_cam * 255.0, 0, 255)))
    cam_image = cam_image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(cam_image, dtype=np.float32) / 255.0


def get_vit_cam_components(model):
    target_layers = [model.vit.encoder.layer[-1].layernorm_before]

    patch_size = model.config.patch_size
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]

    image_size = model.config.image_size
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]

    grid_size = image_size // patch_size

    def reshape_transform(tensor):
        tensor = tensor[:, 1:, :]
        batch_size, _, hidden_size = tensor.size()
        tensor = tensor.reshape(batch_size, grid_size, grid_size, hidden_size)
        return tensor.permute(0, 3, 1, 2)

    return target_layers, reshape_transform


def generate_gradcam(model, input_tensor, device, model_type="resnet50"):
    """Generate grad-CAM heatmap for the input tensor."""
    reshape_transform = None
    cam_model = model
    if model_type.lower() == "resnet50":
        target_layers = [model.layer4[-1]]
    elif model_type.lower() == "vgg16":
        target_layers = [model.features[-1]]
    elif model_type.lower() == "vit":
        target_layers, reshape_transform = get_vit_cam_components(model)
        cam_model = ViTLogitsWrapper(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: resnet50, vgg16, vit")
    
    with GradCAM(model=cam_model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        # For binary classification, we look at positive class (female, class 1)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    
    return grayscale_cam[0, :]


def forward_model(model, input_tensor, model_type="resnet50"):
    if model_type.lower() == "vit":
        return model(pixel_values=input_tensor).logits
    return model(input_tensor)


def visualize_race_group(
    model,
    dataset,
    race_id,
    transform,
    device,
    output_dir,
    num_samples=5,
    model_type="resnet50",
    image_mean=None,
    image_std=None,
):
    """Visualize grad-CAM for samples of a specific race."""
    race_name = RACE_MAPPING.get(race_id, f"Race_{race_id}")
    race_dir = os.path.join(output_dir, f"race_{race_id}_{race_name}")
    os.makedirs(race_dir, exist_ok=True)
    
    race_samples = [sample for sample in dataset if sample["race"] == race_id]
    
    if len(race_samples) == 0:
        print(f"No samples found for race {race_id}")
        return
    
    selected_samples = race_samples[:num_samples]
    
    for idx, sample in enumerate(selected_samples):
        image = sample["image"].convert("RGB")
        gender = sample["gender"]
        gender_name = GENDER_MAPPING.get(gender, f"Gender_{gender}")
        
        # Preprocess and generate grad-CAM
        processed_image = preprocess_image(image, transform)
        input_batch = processed_image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = forward_model(model, input_batch, model_type=model_type)
            prediction = torch.sigmoid(logits).item()
        
        grayscale_cam = generate_gradcam(model, input_batch, device, model_type)
        
        # Denormalize for visualization
        denorm_image = denormalize_image(processed_image, image_mean=image_mean, image_std=image_std)
        
        # Resize CAM to match image size
        grayscale_cam = resize_cam(grayscale_cam, denorm_image.shape[1], denorm_image.shape[0])
        
        # Create visualization
        visualization = show_cam_on_image(denorm_image, grayscale_cam, use_rgb=True)
        
        # Save figure
        output_filename = f"sample_{idx}_gender_{gender_name}.png"
        output_path = os.path.join(race_dir, output_filename)
        
        vis_image = Image.fromarray((visualization * 255).astype(np.uint8))
        vis_image.save(output_path)
        
        print(f"True: {gender_name}, Pred (0 for male, 1 for female): {prediction:.4f}")
        print(f"Saved: {output_path}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize grad-CAM for gender classification across races")
    parser.add_argument("--model-type", type=str, default="resnet50", choices=["resnet50", "vgg16", "vit"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceM4/FairFace")
    parser.add_argument("--dataset-config", type=str, default="1.25")
    parser.add_argument("--output-dir", type=str, default="results/gradcam_visualizations")
    parser.add_argument("--num-samples-per-race", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = resolve_checkpoint_path(args.model_type, args.checkpoint)
    
    print("Loading model...")
    model_result = load_model(checkpoint_path, device, model_type=args.model_type)
    if args.model_type.lower() == "vit":
        model, image_processor = model_result
        image_mean = image_processor.image_mean
        image_std = image_processor.image_std
    else:
        model = model_result[0]
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
    
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split="validation")
    
    transform = get_eval_transform(image_mean=image_mean, image_std=image_std)
    
    os.makedirs(args.output_dir, exist_ok=True)

    race_ids = RACE_MAPPING.keys()
    
    print(f"Generating grad-CAM visualizations for races: {race_ids}")
    for race_id in race_ids:
        print(f"\nProcessing race {race_id} ({RACE_MAPPING.get(race_id, 'Unknown')})...")
        visualize_race_group(
            model,
            dataset,
            race_id,
            transform,
            device,
            args.output_dir,
            args.num_samples_per_race,
            model_type=args.model_type,
            image_mean=image_mean,
            image_std=image_std,
        )
    
    print(f"\nVisualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
