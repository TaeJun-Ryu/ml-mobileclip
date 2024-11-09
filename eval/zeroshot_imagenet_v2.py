#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""
Evaluation code is borrowed from https://github.com/mlfoundations/datacomp/blob/main/eval_utils/wds_eval.py
Licensed under MIT License, see ACKNOWLEDGEMENTS for details.
"""

import os
import argparse

import mobileclip
import torch
from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics import zeroshot_classification as zsc

import time
import traceback
import random
import numpy as np
from lib import slack

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
SEED = 2024
seed_everything(SEED) # Seed 고정

def parse_args(parser):
    parser.add_argument(
        "--model-arch",
        type=str,
        required=True,
        help="Specify model arch from the available choices.",
        choices=['mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b']
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Specify location of model checkpoint.",
    )
    return parser

def create_model(model_arch, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)
    model, _, transform = mobileclip.create_model_and_transforms(
        model_arch, pretrained=model_path
    )
    model.eval()
    model = model.to(device)

    return model, transform, device


def create_webdataset(
    task, transform, data_root=None, dataset_len=None, batch_size=64, num_workers=4
):
    data_folder = f"wds_{task.replace('/','-')}_test"
    if data_root is None:
        data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    else:
        data_root = os.path.join(data_root, data_folder)
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        transform=transform,
        split="test",
        download=False,
    )
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, dataloader


def evaluate_webdataset(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=1,
):
    """Evaluate CLIP model on classification task."""

    # Create model
    model, transform, device = create_model(model_arch, model_path)

    # Load data
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
        zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Evaluate
    classifier = zsc.zero_shot_classifier(
        model,
        mobileclip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
    )
    logits, target = zsc.run_classification(
        model, classifier, dataloader, device, amp=False
    )

    # Compute metrics
    acc1, acc5 = zsc.accuracy(logits, target, topk=(1, 5))
    metrics = {
        "acc1": acc1,
        "acc5": acc5,
    }
    return metrics


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Webdataset evaluation script.")
    # parser = parse_args(parser)
    # args = parser.parse_args()

    task_list = [
        'vtab/caltech101',
        'cifar10',
        'vtab/cifar100',
        'vtab/clevr_count_all',
        'vtab/clevr_closest_object_distance',
        'country211',
        'vtab/dtd',
        'vtab/eurosat',
        'fgvc_aircraft',
        'food101',
        'gtsrb',
        'imagenet1k',
        'imagenet_sketch',
        'imagenetv2',
        'imagenet-a',
        'imagenet-o',
        'imagenet-r',
        'vtab/kitti_closest_vehicle_distance',
        'mnist',
        'objectnet',
        'vtab/flowers',
        'vtab/pets',
        'voc2007',
        'vtab/pcam',
        'renderedsst2',
        'vtab/resisc45',
        'cars',
        'stl10',
        'sun397',
        'vtab/svhn',
        'retrieval/flickr_1k_test_image_text_retrieval',
        'retrieval/mscoco_2014_5k_test_image_text_retrieval',
        'misc/winogavil',
        'wilds/iwildcam',
        'wilds/camelyon17',
        'wilds/fmow',
        'fairness/dollar_street',
        'fairness/geode',
        'fairness/fairface',
        'fairness/utkface'
    ]

    task_list = [
        'vtab/svhn'
    ]

    slack.send_message(f"[MobileClip-v2] Process Start")

    model_arch = 'mobileclip_s0'
    model_path = 'checkpoints/mobileclip_s0.pt'

    for task in task_list:
        try:
            slack.send_message(f"[MobileClip-v2] {task} Start")
            metric = evaluate_webdataset(
                # task=task, model_arch=args.model_arch, model_path=args.model_path
                task=task, model_arch=model_arch, model_path=model_path
            )
            result = {"key":task, "metrics":metric}

            slack.send_message(f"[MobileClip-v2] {task} End | Eval Metrics: {metric}")
            print(f"{task} Eval Metrics: {metric}")

            with open(f"results_v2/{model_arch}_v1.jsonl", "a") as outfile:
                outfile.write(f"{result}\n")
        except:
            traceback_message = traceback.format_exc()
            print(traceback_message)

        time.sleep(5)

    slack.send_message(f"[MobileClip-v2] Process End")