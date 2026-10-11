import argparse
from .workflow import train, predict
from .data import obtain_data
from .common import ROOT

def main():
    parser = argparse.ArgumentParser(description="Pre-call marketing response prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download the official UCI bank-full CSV")
    commands.add_parser("demo", help="Run the labelled offline simulated demonstration")
    training = commands.add_parser("train")
    training.add_argument("--mode", choices=["quick", "full"], default="full")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--data", help="Optional bank-full CSV, separated by semicolons")
    training.add_argument("--output", default="outputs/full")
    inference = commands.add_parser("predict")
    inference.add_argument("--model", default="outputs/full/models/model.joblib")
    inference.add_argument("--input", default="examples/customer.csv")
    args = parser.parse_args()
    if args.command == "download":
        print(obtain_data())
    elif args.command == "demo":
        train("quick", 42, ROOT / "data/demo/bank_demo.csv", ROOT / "outputs/demo")
    elif args.command == "train":
        train(args.mode, args.seed, args.data, args.output)
    else:
        print(predict(args.model, args.input).to_json(orient="records", indent=2))
