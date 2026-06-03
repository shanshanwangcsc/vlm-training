import time
from megatron.energon import get_train_dataset, get_loader, WorkerConfig
import transformers
from data.energon_dataloader import PackedBatchEncoder
from transformers import AutoProcessor
import torch
model_dir='cache/qwen3_2b'
data_path='/scratch/project_462001202/shanshan/synth-data-bench-training/data/cap_pretrain'

seq_len = 8192

tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir,
            model_max_length=seq_len,
            padding_side="right",
            use_fast=False,
        )
pad_token_id = tokenizer.pad_token_id
processor = AutoProcessor.from_pretrained(
            model_dir,
            max_pixels=1048576,
        )

#breakpoint()
worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            data_parallel_group=1,
            num_workers=0,
        )


task_encoder = PackedBatchEncoder(processor, seq_len)
ds = get_train_dataset(
            data_path,
            batch_size=1,
            shuffle_buffer_size=800,
            max_samples_per_sequence=200,
            task_encoder=task_encoder,
            worker_config=worker_config,
            packing_buffer_size=400
        )
data_loader = get_loader(ds)
data_iter = iter(data_loader)

tokens_seen = 0
tokens_seen_assistant = 0
ntokens_since_last_log = 0
total_ntokens_since_last_log = 0
samples_since_last_log = 0
time_last_log = time.perf_counter()
while True:
    data_start_time = time.perf_counter()
    batch = next(data_iter)
    #breakpoint()

    if batch['cu_seqlens'].ndim > 1:
        batch['cu_seqlens'].squeeze_()

    if batch['image_grid_thw'].ndim > 1:
        # do not use squeeze because we need to have two dims
        batch['image_grid_thw'] = batch['image_grid_thw'][0]

    batch['attention_mask'], batch['original_mask'] = batch['cu_seqlens'], batch['attention_mask']
    #breakpoint()
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            #batch[k] = v.to(device=torch.cuda.current_device(), non_blocking=True)
            batch[k] = v
            #breakpoint()

    # the first and last numbers in cu_seqlens do not count towards the sample count
    # (pun intented)
    #breakpoint()
    batch_samples = batch['attention_mask'].shape[0] - 2
    
    ntokens_batch = (batch['input_ids'] != pad_token_id).sum().item()
    ntokens_batch_assistant = (batch['labels'] != -100).sum().item()
    
    


    batch_efficiency = (ntokens_batch / seq_len ) * 100
    print('packing efficency', batch_efficiency)

    tokens_seen_assistant += ntokens_batch_assistant
    tokens_seen += ntokens_batch
    ntokens_since_last_log += ntokens_batch
    total_ntokens_since_last_log += seq_len
    samples_since_last_log += batch_samples

    data_time_delta = time.perf_counter() - data_start_time
    breakpoint()
#shuffle_buffer_size
#    ↓
#controls ORDER of samples

#packing_buffer_size
#    ↓
#controls HOW MANY samples are available for merging


#max_samples_per_sequence
#    ↓
#controls HOW MANY samples can be merged

#seq_len
#    ↓
#controls MAX TOKENS after merging

# shuffle_buffer_size = 4096
#packing_buffer_size = 1024
#max_samples_per_sequence = 100 (depends on how long each sample is, in this dataset, each sample consists of around 100 -300 tokens)
#seq_len = 8192 (depending on GPU memory)

