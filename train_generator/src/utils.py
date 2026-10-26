import os
import json
import torch
import string
import numpy as np
from collections import Counter
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, Qwen2Config
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(cls, prediction: str, ground_truth: Union[str, List[str]], ground_truth_id: Union[str, List[str]] = None):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: Union[str, List[str]], ground_truth_id: Union[str, List[str]] = None):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


def get_model_path(model_name):
    path = ""
    if "llama3.2-1b" in model_name.lower():
        return path + "Meta-Llama-3.2-1B-Instruct"
    elif "llama3-8b" in model_name.lower():
        return path + "Meta-Llama-3-8B-Instruct"
    elif "qwen2.5-1.5b" in model_name.lower():
        return path + "Qwen2.5-1.5B-Instruct"
    elif "qwen2.5-7b" in model_name.lower():
        return path + "Qwen2.5-7B-Instruct"
    return model_name


def get_model_class(model_name):
    model_classes = {
        'llama': 'LlamaForCausalLM',
        'qwen': 'Qwen2ForCausalLM',
    }
    for prefix, model_class in model_classes.items():
        if prefix in model_name.lower():
            return model_class
    return 'LlamaForCausalLM'


def get_model(model_name, max_new_tokens=512, device_map="auto"):
    model_path = get_model_path(model_name)
    model_class_name = get_model_class(model_name)

    if 'qwen' in model_name.lower():
        from transformers import Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    generation_config = dict(
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        temperature=0.1,
        top_k=50,
    )
    return model, tokenizer, generation_config


def model_generate(prompt, model, tokenizer, generation_config):
    messages = [{'role': 'user', 'content': prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    output = model.generate(
        input_ids,
        attention_mask=torch.ones(input_ids.shape).to(model.device),
        **generation_config
    )
    output_text = output.sequences[0][input_len:]
    text = tokenizer.decode(output_text, skip_special_tokens=True)
    return text


def get_attributes(x: nn.Module, attributes: str):
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


def delta_inject(model, adapter_weights):
    modules = set(".".join(k.split(".")[:-2]) for k in adapter_weights.keys())
    for module in modules:
        m = get_attributes(model, module)
        lora_A = adapter_weights[module + ".lora_A.weight"]
        lora_B = adapter_weights[module + ".lora_B.weight"]
        delta = lora_B @ lora_A
        setattr(m, "delta", delta.to(torch.float32))


def delta_remove(model, adapter_weights):
    modules = set(".".join(k.split(".")[:-2]) for k in adapter_weights.keys())
    for module in modules:
        m = get_attributes(model, module)
        delattr(m, "delta")


def evaluate(pred, ground_truth, with_cot=False):
    if not with_cot:
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()
    else:
        if "the answer is" in pred:
            pred = pred[pred.find("the answer is") + len("the answer is"):]
        pred = pred.strip()

    em = BaseDataset.exact_match_score(prediction=pred, ground_truth=ground_truth)["correct"]
    f1_score = BaseDataset.f1_score(prediction=pred, ground_truth=ground_truth)
    return {"em": str(em), "f1": str(f1_score["f1"]), "prec": str(f1_score["precision"]), "recall": str(f1_score["recall"])}