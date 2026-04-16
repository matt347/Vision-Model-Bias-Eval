import argparse
import os
import random
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def get_transforms(image_size, image_mean, image_std):
	train_transform = transforms.Compose(
		[
			transforms.RandomResizedCrop(image_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=image_mean, std=image_std),
		]
	)
	eval_transform = transforms.Compose(
		[
			transforms.Resize(int(image_size * 1.14)),
			transforms.CenterCrop(image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=image_mean, std=image_std),
		]
	)
	return train_transform, eval_transform


def preprocess_example(example, transform):
	images = [img.convert("RGB") for img in example["image"]]
	labels = [int(g) for g in example["gender"]]
	return {
		"pixel_values": [transform(image) for image in images],
		"labels": labels,
	}


def build_datasets(dataset_name, dataset_config, image_size, image_mean, image_std):
	ds = load_dataset(dataset_name, dataset_config)

	print(f"Dataset loaded with splits: {ds.keys()}")

	train_ds = ds["train"]
	val_ds = ds["validation"]

	train_transform, eval_transform = get_transforms(image_size, image_mean, image_std)
	train_ds = train_ds.with_transform(partial(preprocess_example, transform=train_transform))
	val_ds = val_ds.with_transform(partial(preprocess_example, transform=eval_transform))
	return train_ds, val_ds


def make_model(model_name, pretrained=True):
	if pretrained:
		model = ViTForImageClassification.from_pretrained(
			model_name,
			num_labels=1,
			ignore_mismatched_sizes=True,
		)
		return model

	config = ViTConfig.from_pretrained(model_name)
	config.num_labels = 1
	return ViTForImageClassification(config)


@dataclass
class Metrics:
	loss: float
	accuracy: float


def run_epoch(model, dataloader, criterion, device, is_train, optimizer=None):
	model.train(is_train)

	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	for batch in tqdm(dataloader, leave=False):
		images = batch["pixel_values"]
		labels = batch["labels"]
		if labels.ndim == 1:
			labels = labels.unsqueeze(1)
		labels = labels.float()

		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		if is_train:
			optimizer.zero_grad(set_to_none=True)

		with torch.set_grad_enabled(is_train):
			outputs = model(pixel_values=images)
			logits = outputs.logits
			loss = criterion(logits, labels)

			if is_train:
				loss.backward()
				optimizer.step()

		probs = torch.sigmoid(logits)
		preds = (probs >= 0.5).float()

		batch_size = labels.size(0)
		total_loss += loss.item() * batch_size
		total_correct += (preds == labels).sum().item()
		total_samples += batch_size

	return Metrics(
		loss=total_loss / total_samples,
		accuracy=total_correct / total_samples,
	)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
	return run_epoch(
		model=model,
		dataloader=dataloader,
		criterion=criterion,
		device=device,
		is_train=True,
		optimizer=optimizer,
	)


def eval_one_epoch(model, dataloader, criterion, device):
	return run_epoch(
		model=model,
		dataloader=dataloader,
		criterion=criterion,
		device=device,
		is_train=False,
		optimizer=None,
	)


def parse_args():
	parser = argparse.ArgumentParser(description="Fine-tune ViT on FairFace gender")
	parser.add_argument("--dataset-name", type=str, default="HuggingFaceM4/FairFace")
	parser.add_argument("--dataset-config", type=str, default="1.25")
	parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--image-size", type=int, default=224)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--freeze-backbone", action="store_true")
	parser.add_argument("--pretrained", action="store_true")
	parser.add_argument("--output-dir", type=str, default="results")
	return parser.parse_args()


def main():
	args = parse_args()
	set_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	os.makedirs(args.output_dir, exist_ok=True)

	image_processor = ViTImageProcessor.from_pretrained(args.model_name)
	image_mean = image_processor.image_mean
	image_std = image_processor.image_std

	train_ds, val_ds = build_datasets(
		dataset_name=args.dataset_name,
		dataset_config=args.dataset_config,
		image_size=args.image_size,
		image_mean=image_mean,
		image_std=image_std,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	model = make_model(model_name=args.model_name, pretrained=args.pretrained)
	if args.freeze_backbone:
		print("freezing backbone parameters...")
		for name, param in model.named_parameters():
			if not name.startswith("classifier"):
				param.requires_grad = False
	else:
		print("training all parameters...")
	model.to(device)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=args.lr,
		weight_decay=args.weight_decay,
	)

	best_val_acc = 0.0
	for epoch in range(1, args.epochs + 1):
		train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_metrics = eval_one_epoch(model, val_loader, criterion, device)

		print(
			f"Epoch {epoch}/{args.epochs} | "
			f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
			f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
		)

		if val_metrics.accuracy > best_val_acc:
			best_val_acc = val_metrics.accuracy
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"args": vars(args),
					"best_val_acc": best_val_acc,
				},
				os.path.join(args.output_dir, "vit_best.pt"),
			)

	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"args": vars(args),
			"best_val_acc": best_val_acc,
		},
		os.path.join(args.output_dir, "vit_last.pt"),
	)


if __name__ == "__main__":
	main()