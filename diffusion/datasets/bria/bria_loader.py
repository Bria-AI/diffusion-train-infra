# %%

#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import glob
import logging
import random
from typing import Optional, List
from datetime import datetime
import numpy as np
import torch
import torch.utils.checkpoint
from typing import Union
import transformers
from accelerate.logging import get_logger
from datasets import (
    load_dataset,
    Features,
    Value,
    Image,
    DatasetDict,
    Dataset,
)
from datasets.distributed import split_dataset_by_node
from huggingface_hub import HfFolder, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
import wandb
# import boto3
import os
import sys


sys.path.append("/home/ubuntu/Infra/project_x/")
from datetime import date


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def load_dataset_from_parquets(
    training_dirs: List[str], rank: int, world_size: int
):
    parquet_files = []
    print(training_dirs)
    for train_data_dir in training_dirs:
        files = glob.glob(f"{train_data_dir}/*.parquet")
        print(f"We have {len(files)} data files on {train_data_dir}")
        parquet_files += files

    total = len(parquet_files)
    slice_per_process = int(total / world_size)

    print(f"We have {total} data files in total")

    start = slice_per_process * rank

    # Let process rank world-1 to have slice + the remaining data files
    if rank == (world_size - 1):
        end = total
    else:
        end = start + slice_per_process

    print(
        f"Process {rank} will use data files slice {start}:{end} with {end-start} data files in total"
    )

    parquet_files = parquet_files[start:end]
    ds = load_dataset(
        "parquet",
        data_files=parquet_files,
        features=Features({"text": Value("string"), "image": Image()}),
        split="train",
        columns=["text", "image"],
        streaming=True,
    ).with_format("torch")

    return ds

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dataset(
    remote: Union[str, List] = None,
    local: Union[str, List] = None,
    seed: int = 42,
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    tokenizer_revision: str = "main",
    image_column: str = "image",
    caption_column: Union[str, List] = "text",
    resolution: int = 256,
    center_crop: bool = True,
    random_flip: bool = True,
    drop_last: bool = True,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 1,
    pin_memory: bool = True,
    ):

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    RANK = int(os.environ.get("RANK", 0))

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)
    else:
        # Seed for the data shuffling
        # Since we resume runs we want different data on each resumt to make sure we go over all the data
        seed = int(datetime.now().timestamp()) + RANK
        set_seed(seed)

    tokenizer = CLIPTokenizer.from_pretrained(
        tokenizer_name_or_path,
        subfolder="tokenizer",
        revision=tokenizer_revision,
    )
    
    training_dirs = [local]
    print(training_dirs)

    ds = load_dataset_from_parquets(
        training_dirs=training_dirs, rank=RANK, world_size=WORLD_SIZE
    )

    # Split parquets files accoding to rank
    ds = split_dataset_by_node(ds, rank=RANK, world_size=WORLD_SIZE)

    # --- Shuffeling ----
    # Prepare a buffer of 10K (~ 2GB ram for each process. 10GB for A100 machine with 1000Gb of ram)
    # Smaple from the buffer
    # Each new element is replacing the sampled element in the buffer to avoid dups
    if shuffle:
        ds = ds.shuffle(seed=RANK, buffer_size=10_000)
    dataset = DatasetDict({"train": ds})
    print("Dataset loaded...")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # column_names = dataset["train"].column_names
    column_names = ["image", "text"]

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(remote, None)
    if image_column is None:
        image_column = (
            dataset_columns[0]
            if dataset_columns is not None
            else column_names[0]
        )
    else:
        image_column = image_column
        if image_column not in column_names:
            raise ValueError(
                f"image_column' value '{image_column}' needs to be one of: {', '.join(column_names)}"
            )

    if caption_column is None:
        caption_column = (
            dataset_columns[1]
            if dataset_columns is not None
            else column_names[1]
        )
    else:
        caption_column = caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"caption_column' value '{caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            # elif remote == "imagenet-1k":  # TODO:
            #      with open("imagenet-simple-labels.json") as f:
            #         imagenet_labels = json.load(f)
            #         caption_column = "label"
            #         caption_column.append("A photo of a " + imagenet_labels[caption])
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(
                    random.choice(caption) if is_train else caption[0]
                )
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(resolution)
            if center_crop
            else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip()
            if random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [
            train_transforms(image) for image in images
        ]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    train_dataset = dataset["train"].map(
        preprocess_train,
        batched=True,
        batch_size=batch_size,
        remove_columns=["image", "text"],
    )

    def collate_fn(examples):
        pixel_values = torch.stack(
            [example["pixel_values"] for example in examples]
        )
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])

        return pixel_values, input_ids

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,  # This is needed until we move null text to data loader
        pin_memory=pin_memory,
        # **dataloader_kwargs,
    )

    return train_dataloader