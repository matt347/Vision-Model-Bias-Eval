import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_RACES = [
	"White",
	"Black",
	"Indian",
	"Middle Eastern",
	"Latino_Hispanic",
	"East Asian",
	"Southeast Asian",
]


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run the race-specific ResNet gender bias experiment once per race and aggregate results"
	)
	parser.add_argument("--script", type=str, default="experiments/compare_race.py")
	parser.add_argument("--races", nargs="*", default=DEFAULT_RACES)
	parser.add_argument("--dataset-name", type=str, default="HuggingFaceM4/FairFace")
	parser.add_argument("--dataset-config", type=str, default="1.25")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--image-size", type=int, default=224)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--freeze-backbone", action="store_true")
	parser.add_argument("--pretrained", action="store_true")
	parser.add_argument("--output-dir", type=str, default="results/race_comparison")
	return parser.parse_args()


def run_experiment(script_path, race_name, args, output_root):
	before_reports = {
		p.resolve(): p.stat().st_mtime
		for p in output_root.glob("gender_from_*/gender_bias_report.json")
		if p.is_file()
	}
	cmd = [
		sys.executable,
		script_path,
		"--train-race",
		race_name,
		"--dataset-name",
		args.dataset_name,
		"--dataset-config",
		args.dataset_config,
		"--epochs",
		str(args.epochs),
		"--batch-size",
		str(args.batch_size),
		"--lr",
		str(args.lr),
		"--weight-decay",
		str(args.weight_decay),
		"--num-workers",
		str(args.num_workers),
		"--image-size",
		str(args.image_size),
		"--seed",
		str(args.seed),
		"--output-dir",
		str(output_root),
	]
	if args.freeze_backbone:
		cmd.append("--freeze-backbone")
	if args.pretrained:
		cmd.append("--pretrained")

	print(f"\n=== Training model on {race_name} ===")
	subprocess.run(cmd, check=True)

	after_reports = [
		p
		for p in output_root.glob("gender_from_*/gender_bias_report.json")
		if p.is_file()
	]
	if not after_reports:
		raise FileNotFoundError("No gender_bias_report.json found under output directory")

	new_reports = [p for p in after_reports if p.resolve() not in before_reports]
	if new_reports:
		json_path = max(new_reports, key=lambda p: p.stat().st_mtime)
	else:
		json_path = max(after_reports, key=lambda p: p.stat().st_mtime)

	with open(json_path, "r", encoding="utf-8") as f:
		return json.load(f)


def format_table(rows):
	headers = ["Train Race", "In-Dist", "Mean OOD", "Worst OOD", "Gap Mean", "Gap Worst"]
	data_rows = []
	for row in rows:
		data_rows.append(
			[
				row["train_race"],
				f'{row["in_distribution_accuracy"]:.4f}',
				f'{row["mean_out_of_distribution_accuracy"]:.4f}',
				f'{row["worst_out_of_distribution_accuracy"]:.4f}',
				f'{row["gap_mean_ood_vs_in_dist"]:.4f}',
				f'{row["gap_worst_ood_vs_in_dist"]:.4f}',
			]
		)

	widths = [len(header) for header in headers]
	for row in data_rows:
		for idx, cell in enumerate(row):
			widths[idx] = max(widths[idx], len(cell))

	def render_row(values):
		return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

	lines = [render_row(headers), "-|-".join("-" * width for width in widths)]
	for row in data_rows:
		lines.append(render_row(row))
	return "\n".join(lines)


def main():
	args = parse_args()
	script_path = Path(args.script)
	output_root = Path(args.output_dir)
	output_root.mkdir(parents=True, exist_ok=True)

	aggregated_rows = []
	all_reports = []

	for race_name in args.races:
		report = run_experiment(str(script_path), race_name, args, output_root)
		all_reports.append(report)
		summary = report["summary"]
		aggregated_rows.append(
			{
				"train_race": report["train_race"]["name"],
				"in_distribution_accuracy": summary["in_distribution_accuracy"],
				"mean_out_of_distribution_accuracy": summary["mean_out_of_distribution_accuracy"],
				"worst_out_of_distribution_accuracy": summary["worst_out_of_distribution_accuracy"],
				"gap_mean_ood_vs_in_dist": summary["gap_mean_ood_vs_in_dist"],
				"gap_worst_ood_vs_in_dist": summary["gap_worst_ood_vs_in_dist"],
			}
		)

	aggregated_report = {
		"config": {
			"dataset_name": args.dataset_name,
			"dataset_config": args.dataset_config,
			"epochs": args.epochs,
			"batch_size": args.batch_size,
			"lr": args.lr,
			"weight_decay": args.weight_decay,
			"num_workers": args.num_workers,
			"image_size": args.image_size,
			"seed": args.seed,
			"freeze_backbone": args.freeze_backbone,
			"pretrained": args.pretrained,
			"races": args.races,
		},
		"runs": all_reports,
		"comparison": aggregated_rows,
	}

	json_path = output_root / "race_comparison_summary.json"
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(aggregated_report, f, indent=2)

	print("\n=== Final Comparison Table ===")
	print(format_table(aggregated_rows))
	print(f"\nAggregate JSON: {json_path}")


if __name__ == "__main__":
	main()
