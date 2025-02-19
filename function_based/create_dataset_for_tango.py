import os
import sys
import json
import random
import pandas as pd
import shutil

from config import DATASET_PATH, DATASET_OUTPUT_FOLDER

def round_floats(obj, precision=3):
    if isinstance(obj, dict):
        return {
            k: round_floats(v, precision)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [round_floats(elem, precision) for elem in obj]
    if isinstance(obj, float):
        return round(obj, precision)
    return obj


def dict_to_readable_string(d, indent=" "):
    def recurse(obj, level=0):
        current_indent = indent * level
        next_indent = indent * (level + 1)
        if isinstance(obj, dict):
            items = []
            for key, value in obj.items():
                if isinstance(value, dict):
                    items.append(
                        f'{current_indent}"{key}": {{\n'
                        f'{recurse(value, level + 1)}\n'
                        f'{current_indent}}}'
                    )
                elif isinstance(value, list):
                    list_items = []
                    for item in value:
                        if isinstance(item, dict):
                            list_items.append(
                                f'{{\n{recurse(item, level + 1)}\n'
                                f'{next_indent}}}'
                            )
                        else:
                            list_items.append(f'"{item}"')
                    items.append(
                        f'{current_indent}"{key}": '
                        f'[{", ".join(list_items)}]'
                    )
                else:
                    if isinstance(value, str):
                        items.append(
                            f'{current_indent}"{key}": "{value}"'
                        )
                    else:
                        items.append(
                            f'{current_indent}"{key}": {value}'
                        )
            return ",\n".join(items)
        else:
            return f"{current_indent}{obj}"
    formatted_string = recurse(d)
    return f"{{\n{formatted_string}\n}}"


def process_json_files(input_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    json_files = [
        f for f in os.listdir(input_dir) if f.endswith(".json")
    ]
    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        sys.exit(1)
    for filename in json_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(processed_dir, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = round_floats(data, precision=3)
        function_based_explanation = {
            "shapes": data.pop("shapes"),
            "patterns": data.pop("patterns"),
        }
        readable_explanation = dict_to_readable_string(
            function_based_explanation
        )
        spectrogram_base = data.get("spectrogram_base", "N/A")
        duration = data.get("duration", "N/A")
        hz = data.get("hz", "N/A")
        spectrogram_base_explain = (
            f"spectrogram based: {spectrogram_base}.\n"
            f"spectrogram duration: {duration} and hz: {hz}.\n"
        )
        explain = spectrogram_base_explain + readable_explanation
        data[
            "function_based_explanation_spectrogram"
        ] = explain
        location = data.get("file_path") or data.get("location")
        if not location:
            print(
                f"파일 경로 정보가 누락되었습니다: {filename}"
            )
            sys.exit(1)
        new_data = {
            "dataset": "function_based",
            "location": location,
            "captions": data[
                "function_based_explanation_spectrogram"
            ],
            "labels": ""
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)


def copy_processed_files(processed_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(processed_dir):
        if filename.endswith(".json"):
            src = os.path.join(processed_dir, filename)
            dst = os.path.join(output_dir, filename)
            shutil.copy(src, dst)


def merge_json_files(input_dir, merged_json_path):
    all_data = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_data.append(data)
    if not all_data:
        print(f"No data found in '{input_dir}'.")
        sys.exit(1)
    with open(merged_json_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_merged_dataset(merged_json_path,
                         output_dir,
                         train_ratio=0.8,
                         valid_ratio=0.1,
                         test_ratio=0.1):
    with open(merged_json_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = [
        json.loads(line.strip())
        for line in lines if line.strip()
    ]
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    train_path = os.path.join(
        output_dir, "train_function_based.json"
    )
    valid_path = os.path.join(
        output_dir, "valid_function_based.json"
    )
    test_path = os.path.join(
        output_dir, "test_function_based_subset.json"
    )
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(valid_path, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        f"Split dataset into {len(train_data)} train, "
        f"{len(valid_data)} valid, {len(test_data)} test entries."
    )


def confirm_and_remove_dir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return True


def main():
    input_json_dir = "augmentation/output/json"
    processed_dir = "augmentation/processed_json"
    output_dir = DATASET_OUTPUT_FOLDER
    if not os.path.exists(input_json_dir):
        print(
            f"The source directory '{input_json_dir}' does not exist."
        )
        sys.exit(1)
    if not any(
        f.endswith(".json")
        for f in os.listdir(input_json_dir)
    ):
        print(
            f"No valid JSON files found in '{input_json_dir}'."
        )
        sys.exit(1)
    for dir_path in [processed_dir, output_dir]:
        confirm_and_remove_dir(dir_path)
    process_json_files(input_json_dir, processed_dir)
    copy_processed_files(processed_dir, output_dir)
    merged_json_path = os.path.join(
        output_dir, "merged_dataset.json"
    )
    merge_json_files(output_dir, merged_json_path)
    split_merged_dataset(
        merged_json_path,
        output_dir,
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1
    )


if __name__ == "__main__":
    main()
