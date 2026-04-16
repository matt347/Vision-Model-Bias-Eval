import argparse
import json
import os
import random
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from datasets import concatenate_datasets, load_dataset
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
	"""Convert images to RGB, apply transform."""
	images = [img.convert("RGB") for img in example["image"]]
	labels = [int(g) for g in example["gender"]]
	return {
		"pixel_values": [transform(image) for image in images],
		"labels": labels,
	}


def get_race_names(ds):
	race_feature = ds.features["race"]
	if hasattr(race_feature, "names") and race_feature.names is not None:
		return list(race_feature.names)
	unique_ids = sorted(set(ds["race"]))
	return [f"race_{idx}" for idx in unique_ids]


def get_race_id(race_value, race_names):
	alias_map = {
		"asian": ["east asian", "southeast asian"],
		"latino": ["latino hispanic", "latino_hispanic"],
		"hispanic": ["latino hispanic", "latino_hispanic"],
		"others": ["middle eastern"],
		"other": ["middle eastern"],
	}

	if isinstance(race_value, int):
		if 0 <= race_value < len(race_names):
			return race_value
		raise ValueError(f"Race id {race_value} out of range [0, {len(race_names) - 1}]")

	race_value = str(race_value)
	if race_value.isdigit():
		race_id = int(race_value)
		if 0 <= race_id < len(race_names):
			return race_id

	normalized_to_id = {name.lower(): idx for idx, name in enumerate(race_names)}
	key = race_value.lower().replace("-", " ").replace("_", " ").strip()
	if key in alias_map:
		for alias_name in alias_map[key]:
			if alias_name in normalized_to_id:
				return normalized_to_id[alias_name]

	for name_lower, idx in normalized_to_id.items():
		if key == name_lower.replace("_", " "):
			return idx


def filter_race(ds, race_id):
	return ds.filter(lambda ex: ex["race"] == race_id)


def build_eval_groups(val_ds, train_race_id, race_names):
	eval_groups = {}
	in_dist = filter_race(val_ds, train_race_id)
	eval_groups[f"in_dist::{race_names[train_race_id]}"] = in_dist

	ood_parts = []
	for race_id, race_name in enumerate(race_names):
		if race_id == train_race_id:
			continue
		race_subset = filter_race(val_ds, race_id)
		eval_groups[f"ood::{race_name}"] = race_subset
		if len(race_subset) > 0:
			ood_parts.append(race_subset)

	if ood_parts:
		eval_groups["ood::all"] = concatenate_datasets(ood_parts)
	return eval_groups


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
	n_samples: int


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

	if total_samples == 0:
		return Metrics(loss=0.0, accuracy=0.0, n_samples=0)

	return Metrics(
		loss=total_loss / total_samples,
		accuracy=total_correct / total_samples,
		n_samples=total_samples,
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
	parser = argparse.ArgumentParser(
		description="Train ViT gender classifier on one race and evaluate cross-race bias"
	)
	parser.add_argument("--dataset-name", type=str, default="HuggingFaceM4/FairFace")
	parser.add_argument("--dataset-config", type=str, default="1.25")
	parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
	parser.add_argument("--train-race", type=str, default="White")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--image-size", type=int, default=224)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--freeze-backbone", action="store_true")
	parser.add_argument("--pretrained", action="store_true")
	parser.add_argument("--output-dir", type=str, default="results")
	return parser.parse_args()


def save_report(output_dir, report):
	os.makedirs(output_dir, exist_ok=True)

	json_path = os.path.join(output_dir, "gender_bias_report.json")
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)

	return json_path


def main():
	args = parse_args()
	set_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	ds = load_dataset(args.dataset_name, args.dataset_config)
	train_raw = ds["train"]
	val_raw = ds["validation"]

	race_names = get_race_names(train_raw)
	train_race_id = get_race_id(args.train_race, race_names)
	train_race_name = race_names[train_race_id]

	print("Race map:")
	for idx, race_name in enumerate(race_names):
		print(f"  {idx}: {race_name}")
	print(f"\nTraining only on race: {train_race_id} ({train_race_name})")

	train_race_ds = filter_race(train_raw, train_race_id)
	eval_groups_raw = build_eval_groups(val_raw, train_race_id, race_names)

	print(f"Train samples ({train_race_name}): {len(train_race_ds)}")
	for group_name, group_ds in eval_groups_raw.items():
		print(f"Eval samples ({group_name}): {len(group_ds)}")

	image_processor = ViTImageProcessor.from_pretrained(args.model_name)
	image_mean = image_processor.image_mean
	image_std = image_processor.image_std

	train_transform, eval_transform = get_transforms(args.image_size, image_mean, image_std)
	train_ds = train_race_ds.with_transform(partial(preprocess_example, transform=train_transform))
	eval_groups = {}
	for group_name, group_ds in eval_groups_raw.items():
		eval_groups[group_name] = group_ds.with_transform(
			partial(preprocess_example, transform=eval_transform)
		)

	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	eval_loaders = {
		name: DataLoader(
			split,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.num_workers,
			pin_memory=torch.cuda.is_available(),
		)
		for name, split in eval_groups.items()
	}

	model = make_model(model_name=args.model_name, pretrained=args.pretrained)
	if args.freeze_backbone:
		for name, param in model.named_parameters():
			if not name.startswith("classifier"):
				param.requires_grad = False
	model.to(device)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=args.lr,
		weight_decay=args.weight_decay,
	)

	in_dist_key = f"in_dist::{train_race_name}"
	for epoch in range(1, args.epochs + 1):
		train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
		in_dist_metrics = eval_one_epoch(model, eval_loaders[in_dist_key], criterion, device)

		print(
			f"Epoch {epoch}/{args.epochs} | "
			f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
			f"in_dist_loss={in_dist_metrics.loss:.4f} in_dist_acc={in_dist_metrics.accuracy:.4f}"
		)

	rows = []
	metrics_by_group = {}
	for group_name, loader in eval_loaders.items():
		metrics = eval_one_epoch(model, loader, criterion, device)
		metrics_by_group[group_name] = metrics

	in_dist_acc = metrics_by_group[in_dist_key].accuracy
	ood_accs = [
		metrics.accuracy
		for group_name, metrics in metrics_by_group.items()
		if group_name.startswith("ood::") and group_name != "ood::all"
	]
	mean_ood_acc = float(np.mean(ood_accs)) if ood_accs else 0.0
	worst_ood_acc = float(np.min(ood_accs)) if ood_accs else 0.0

	for group_name, metrics in metrics_by_group.items():
		_, race_name = group_name.split("::", 1)
		rows.append(
			{
				"group": group_name.split("::", 1)[0],
				"race": race_name,
				"n_samples": metrics.n_samples,
				"loss": round(metrics.loss, 6),
				"accuracy": round(metrics.accuracy, 6),
				"gap_vs_in_dist": round(in_dist_acc - metrics.accuracy, 6),
			}
		)

	report = {
		"train_race": {"id": train_race_id, "name": train_race_name},
		"summary": {
			"in_distribution_accuracy": in_dist_acc,
			"mean_out_of_distribution_accuracy": mean_ood_acc,
			"worst_out_of_distribution_accuracy": worst_ood_acc,
			"gap_mean_ood_vs_in_dist": in_dist_acc - mean_ood_acc,
			"gap_worst_ood_vs_in_dist": in_dist_acc - worst_ood_acc,
		},
		"per_group": rows,
		"config": vars(args),
	}

	run_name = f"gender_from_{train_race_name.replace(' ', '_')}"
	out_dir = os.path.join(args.output_dir, run_name)
	os.makedirs(out_dir, exist_ok=True)

	json_path = save_report(out_dir, report)

	print("\n=== Bias Summary ===")
	print(f"In-dist accuracy ({train_race_name}): {in_dist_acc:.4f}")
	print(f"Mean OOD accuracy: {mean_ood_acc:.4f}")
	print(f"Worst OOD accuracy: {worst_ood_acc:.4f}")
	print(f"Accuracy gap (in-dist - mean OOD): {in_dist_acc - mean_ood_acc:.4f}")
	print(f"Accuracy gap (in-dist - worst OOD): {in_dist_acc - worst_ood_acc:.4f}")
	print(f"Report JSON: {json_path}")


if __name__ == "__main__":
	main()