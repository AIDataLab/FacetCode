from src.root_dir_path import ROOT_DIR

DATA_ROOT_DIR = ROOT_DIR


def load_data(data_name, data_type, model_name, data_dir=None):
    import os
    import json
    solve_dataset = []
    if data_dir is None:
        data_dir = DATA_ROOT_DIR
    print(f"Load data_dir: {data_dir}")
    input_dir = os.path.join(data_dir, data_name, model_name)
    print(f"Loading data from {input_dir}")

    if not os.path.exists(input_dir):
        input_dir = data_dir

    if os.path.exists(input_dir) and os.path.isdir(input_dir):
        files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        if len(files) > 0:
            for filename in files:
                with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as fin:
                    data = json.load(fin)
                    if isinstance(data, list):
                        solve_dataset.append((filename, data))
                    elif isinstance(data, dict) and "total" in data:
                        solve_dataset.append((filename, data["total"]))
        else:
            print(f"No JSON files found in {input_dir}")
    else:
        print(f"Directory does not exist: {input_dir}")

    if len(solve_dataset) == 0:
        icd_file = os.path.join(data_dir, "icd_codes_sample.json")
        if os.path.exists(icd_file):
            with open(icd_file, "r", encoding="utf-8") as fin:
                solve_dataset.append(("icd_codes_sample.json", json.load(fin)))

    return solve_dataset