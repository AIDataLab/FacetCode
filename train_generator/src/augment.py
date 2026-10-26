import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from src.utils import get_model, model_generate
from src.prompt_template import REWRITING_PROMPT, QA_GENERATION_PROMPT, fix_json_qa, fix_qa

random.seed(42)


def load_icd_data(data_path):
    with open(data_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    return data


def main(args):
    output_dir = os.path.join(args.output_dir, args.model_name.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)

    print("### Loading ICD dataset ###")
    dataset = load_icd_data(args.data_path)
    total_size = len(dataset)
    print(f"Total ICD codes: {total_size}")

    model, tokenizer, generation_config = get_model(args.model_name, max_new_tokens=512)
    args.model_name = args.model_name.split("/")[-1]
    generation_config["temperature"] = 0.1

    all_results = []
    pbar = tqdm(total=total_size * args.topk)

    for did, data in enumerate(dataset):
        code = data.get("code", f"code_{did}")
        desc = data.get("description", "")
        parent = data.get("parent_description", "")
        siblings = data.get("sibling_descriptions", [])

        icd_result = {
            "code": code,
            "description": desc,
            "parent_description": parent,
            "sibling_descriptions": siblings,
            "passages": [],
            "augment": []
        }

        passages = data.get("passages", [])
        if len(passages) == 0:
            passages = [desc]

        for pid, psg in enumerate(passages[:args.topk]):
            aug_entry = {
                "pid": pid,
                "passage": psg,
            }

            rewrite_prompt = REWRITING_PROMPT.format(passage=psg)
            rewrite = model_generate(rewrite_prompt, model, tokenizer, generation_config)
            aug_entry[f"{args.model_name}_rewrite"] = rewrite

            qa_prompt = QA_GENERATION_PROMPT.format(passage=psg)
            qa_output = model_generate(qa_prompt, model, tokenizer, generation_config)
            qa_output = fix_json_qa(qa_output, args.model_name)

            try:
                qa = json.loads(qa_output)
                ret, qa = fix_qa(qa)
                if ret:
                    aug_entry[f"{args.model_name}_qa"] = qa
                    icd_result["augment"].append(aug_entry)
                    icd_result["passages"].append(psg)
            except:
                pass

            pbar.update(1)
            if len(icd_result["augment"]) >= args.topk:
                break

        all_results.append(icd_result)
        pbar.set_description(f"Processed {did+1}/{total_size}")

        if (did + 1) % args.save_interval == 0:
            temp_file = os.path.join(output_dir, f"icd_aug_temp_{did+1}.json")
            with open(temp_file, "w", encoding="utf-8") as fout:
                json.dump(all_results, fout, ensure_ascii=False, indent=2)

    pbar.close()

    final_output_file = os.path.join(output_dir, "icd_augmented.json")
    with open(final_output_file, "w", encoding="utf-8") as fout:
        json.dump(all_results, fout, ensure_ascii=False, indent=2)

    print(f"### Saved to {final_output_file} ###")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3.2-1b-instruct")
    parser.add_argument("--data_path", type=str, default="./data/icd_codes.json")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./data_aug")
    parser.add_argument("--save_interval", type=int, default=100)
    args = parser.parse_args()
    print(args)
    main(args)