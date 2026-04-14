import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms

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


def load_model(checkpoint_path, device):
    model = models.resnet50()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def preprocess_image(image, transform):
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    processed = transform(image)
    return processed


def denormalize_image(tensor):
    """Convert normalized tensor back to [0, 1] for visualization."""
    tensor = tensor.clone().detach()
    for t, m, s in zip(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.cpu().numpy().transpose(1, 2, 0)


def generate_gradcam(model, input_tensor, device):
    """Generate grad-CAM heatmap for the input tensor."""
    target_layers = [model.layer4[-1]]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # For binary classification, we look at positive class (female, class 1)
        targets = None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    return grayscale_cam[0, :]


def visualize_race_group(model, dataset, race_id, transform, device, output_dir, num_samples=5):
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
            logits = model(input_batch)
            prediction = torch.sigmoid(logits).item()
        
        grayscale_cam = generate_gradcam(model, input_batch, device)
        
        # Denormalize for visualization
        denorm_image = denormalize_image(processed_image)
        
        # Resize CAM to match image size
        grayscale_cam = cv2.resize(grayscale_cam, (denorm_image.shape[1], denorm_image.shape[0]))
        
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
    parser.add_argument("--checkpoint", type=str, default="results/resnet50_best.pt")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceM4/FairFace")
    parser.add_argument("--dataset-config", type=str, default="1.25")
    parser.add_argument("--output-dir", type=str, default="results/gradcam_visualizations")
    parser.add_argument("--num-samples-per-race", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split="validation")
    
    transform = get_eval_transform()
    
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
        )
    
    print(f"\nVisualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
