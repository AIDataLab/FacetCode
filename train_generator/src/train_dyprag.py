import torch
import os

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())

import os
import gc
import json
import argparse
import torch
import random
from tqdm import tqdm
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from projector import ParameterTranslator
from utils import get_model, load_data, delta_inject, delta_remove
from safetensors.torch import load_file
from transformers import DefaultDataCollator
from typing import List, Dict
import torch.nn.functional as F
from collections import defaultdict
import time


def create_lora_passage_dataset(lora_passage_pairs):
    from torch.utils.data import Dataset

    class LoRAPassageDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            return self.pairs[idx]

    return LoRAPassageDataset(lora_passage_pairs)


def prepare_training_data_multi_datasets(args, tokenizer, datasets):
    all_training_samples = []
    ignored_id = -100
    max_length = 2000
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        args.dataset = dataset_name

        data_list = load_data(dataset_name, args.data_type, args.augment_model, args.data_dir)
        for filename, fulldata in data_list:
            filename = filename.split(".")[0]
            print(f"Collecting data from {filename}")
            data_size = len(fulldata)
            print(f"Total data size: {data_size}")
            sample_data = random.sample(range(data_size), int(data_size * args.sample_rate))

            for test_id in tqdm(sample_data):
                data = fulldata[test_id]
                augment = data["augment"]
                for pid, aug in enumerate(augment):
                    adapter_path = os.path.join(
                        args.output_dir,
                        args.model_name,
                        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
                        dataset_name,
                        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_direct",
                        f"aug_model={args.augment_model}",
                        filename,
                        f"data_{test_id}",
                        f"passage_{pid}"
                    )
                    if os.path.exists(adapter_path):
                        aug_text = aug["passage"]
                        passage_tokens = tokenizer(
                            aug_text,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=max_length
                        )
                        all_training_samples.append({
                            'adapter_path': adapter_path,
                            'passage': aug_text,
                            'passage_tokens': passage_tokens,
                            'file_name': filename,
                            'data_id': test_id,
                            'passage_id': pid,
                            'dataset': dataset_name
                        })

    print(f"\nTotal prepared {len(all_training_samples)} training samples from {len(datasets)} datasets")
    return all_training_samples


class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.model = model

    def __call__(self, examples: List[Dict[str, dict]]) -> Dict[str, torch.Tensor]:
        input_embeds = []
        adapter_paths = []
        model_inputs = []

        for example in examples:
            tokenized = example['passage_tokens']
            input_ids = tokenized.input_ids.to(self.device)
            attention_mask = tokenized.attention_mask.to(self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                input_embeds.append(output.hidden_states[-1][:, -1, :])
                adapter_paths.append(example['adapter_path'])

            model_inputs.append({
                'input_ids': torch.tensor(example.get('input_ids', input_ids)).unsqueeze(0).to(self.device),
                'labels': torch.tensor(example.get('labels', input_ids)).unsqueeze(0).to(self.device),
                'attention_mask': torch.tensor(example.get('attention_mask', attention_mask)).unsqueeze(0).to(self.device)
            })

        return {
            "input_embeds": input_embeds,
            "adapter_paths": adapter_paths,
            "model_inputs": model_inputs,
        }


def main(args):
    datasets = args.datasets
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_projector = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Model device: {device}")
    print(f"Projector device: {device_projector}")

    base_model, tokenizer, generation_config = get_model(args.model_name, max_new_tokens=args.max_new_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
    )
    model = get_peft_model(base_model, peft_config)

    training_samples = prepare_training_data_multi_datasets(args, tokenizer, datasets)
    print(f"Total training samples: {len(training_samples)}")

    if len(training_samples) == 0:
        print("No training samples found. Exiting.")
        return

    dataset = create_lora_passage_dataset(training_samples)
    print(f"Dataset size: {len(dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=TrainingDataCollator(tokenizer, device, model)
    )

    print(f"Initializing projector with {args.projector_p} hidden layers")
    projector = ParameterTranslator(
        ["down_proj", "up_proj", "gate_proj"],
        list(range(base_model.config.num_hidden_layers)),
        base_model.config.hidden_size,
        base_model.config.intermediate_size,
        args.lora_rank,
        args.projector_p,
    ).to(device_projector)
    projector.train()

    model.eval()
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.dyprag_learning_rate)

    losses = defaultdict(list)
    DYPRAG_TRAIN_EPOCH = args.dyprag_train_epochs
    total_time = 0

    for epoch in range(DYPRAG_TRAIN_EPOCH):
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            optimizer.zero_grad()

            input_embeds = batch['input_embeds'][0]
            adapter_path = batch['adapter_paths'][0]
            model_inputs = batch['model_inputs'][0]

            lora_state_dict = load_file(os.path.join(adapter_path, "adapter_model.safetensors"), device=device_projector.type)

            with torch.no_grad():
                model.load_adapter(adapter_path, adapter_name="default")
                model.set_adapter("default")
                lm_origin_outputs = model(**model_inputs)
                lm_origin_loss = lm_origin_outputs.loss
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.data.zero_()

            outputs = projector(input_embeds.to(device_projector).to(torch.float32))
            outputs = {k: v.to(device) for k, v in outputs.items()}
            delta_inject(model, outputs)

            with torch.set_grad_enabled(True):
                lm_outputs = model(**model_inputs)
                lm_loss = lm_outputs.loss
                kl_loss = F.kl_div(
                    F.log_softmax(lm_outputs.logits, dim=-1),
                    F.softmax(lm_origin_outputs.logits, dim=-1),
                    reduction='batchmean'
                ) * 0.01

            mse_loss = torch.tensor(0.0, device=device_projector, dtype=torch.float32)
            for key in outputs:
                if key in lora_state_dict:
                    mse_loss += F.mse_loss(
                        outputs[key].to(device_projector).to(torch.float32),
                        lora_state_dict[key].to(device_projector).to(torch.float32)
                    )
            mse_loss *= 100

            total_loss = lm_loss.to(device_projector) + kl_loss.to(device_projector) + mse_loss

            total_loss.backward()
            optimizer.step()
            delta_remove(model, outputs)

            del outputs
            torch.cuda.empty_cache()

            step_time = time.time() - start_time
            total_time += step_time

            print(f"Epoch {epoch}, Step {step}, MSE: {mse_loss.item():.4f}, LM: {lm_loss.item():.4f}, "
                  f"KL: {kl_loss.item():.4f}, Target LM: {lm_origin_loss.item():.4f}, Time: {step_time:.2f}s")

        save_dir = os.path.join(
            args.output_dir,
            args.model_name,
            f"projector_hidden{args.projector_p}_sample{args.sample_rate}_lr{args.dyprag_learning_rate}"
        )
        os.makedirs(save_dir, exist_ok=True)
        print(f"Save projector to {save_dir}")
        torch.save({'model_state_dict': projector.state_dict()}, os.path.join(save_dir, f'epoch_{epoch}.pt'))

    print(f"Finished DyPRAG Training! Total time: {total_time:.2f}s")


def str_to_list(s):
    return s.split(',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--datasets", type=str_to_list, default=["icd_augmented"])
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dyprag_train_epochs", type=int, default=1)
    parser.add_argument("--sample_rate", type=float, default=0.2)
    parser.add_argument("--dyprag_learning_rate", type=float, default=1e-5)
    parser.add_argument("--projector_p", type=int, default=32)
    parser.add_argument("--augment_model", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="total")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./offline")
    parser.add_argument("--data_dir", type=str, default="./data_aug")
    args = parser.parse_args()

    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)