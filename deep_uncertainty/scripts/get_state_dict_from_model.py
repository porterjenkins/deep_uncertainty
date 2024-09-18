import argparse
import os

from deep_uncertainty.utils.model_utils import extract_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to the model checkpoint")
    parser.add_argument(
        "--output-dir", type=str, help="Directory to save the extracted state dictionary"
    )
    args = parser.parse_args()

    fname = os.path.split(args.model_path)[-1]
    output_path = os.path.join(args.output_dir, f"state_dict_{fname}")
    extract_state_dict(args.model_path, output_path)
