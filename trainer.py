import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math
import argparse
from datetime import datetime, timezone, timedelta

import torch
import bitsandbytes as bnb
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from transformers import AutoConfig, BitsAndBytesConfig, get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DummyScheduler, set_seed
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from accelerate import FullyShardedDataParallelPlugin

# Project modules 
from logger import getLogger
from model import LLMLoader           # LoRA 적용 래퍼
from utils import get_datasetM        # metaIntent 데이터 로딩/전처리

from dataset import Dataset, DataCollator


def parse_args():
    p = argparse.ArgumentParser(description="Refactored trainer4 in train.py style (shell flags aligned)")

    # ===== Data & Model =====
    p.add_argument("--data-path", "--training_data_path",
                   dest="training_data_path", type=str, required=True,
                   help="학습 데이터 CSV 경로")
    p.add_argument("--model", "--model_path_or_name",
                   dest="model_path_or_name", type=str, required=True,
                   help="사전학습 모델 이름 또는 경로")
    p.add_argument("--tokenizer",
                   dest="tokenizer_path_or_name", type=str, required=False, default="",
                   help="토크나이저 이름 또는 경로(미지정 시 model과 동일하게 처리)")

    # (옵션) teacher
    p.add_argument("--teacher-model", "--teacher_model_path_or_name",
                   dest="teacher_model_path_or_name", type=str, default="",
                   help="선택: teacher 모델 경로 (미사용 가능)")

    # ===== Task / Intent =====
    p.add_argument("--intent-level", "--intent_level",
                   dest="intent_level", type=str, default="metaIntent",
                   choices=["metaIntent", "intent"], help="의도 레벨")

    # ===== Lengths =====
    p.add_argument("--max-source-len", "--max_source_len", "--max_query_length",
                   dest="max_source_len", type=int, default=1024,
                   help="입력 최대 길이")
    p.add_argument("--max-target-len", "--max_target_len",
                   dest="max_target_len", type=int, default=400,
                   help="출력 최대 길이")

    # ===== Training =====
    p.add_argument("--seed", dest="seed", type=int, default=42, help="시드")
    p.add_argument("--epochs", "--epoch_size",
                   dest="epoch_size", type=int, default=3, help="에폭 수")

    p.add_argument("--train-batch-size", "--train_batch_size",
                   dest="train_batch_size", type=int, default=2, help="학습 배치")
    p.add_argument("--dev-batch-size", "--dev_batch_size",
                   dest="dev_batch_size", type=int, default=1, help="검증 배치")

    # grad accumulation: 쉘 스크립트의 --train-acc-step을 그대로 받되,
    # 내부에서는 gradient_accumulation_steps 로 사용하도록 동일 dest로 연결
    p.add_argument("--train-acc-step", "--gradient_accumulation_steps",
                   dest="gradient_accumulation_steps", type=int, default=8,
                   help="Gradient Accumulation Steps")
    p.add_argument("--eval-acc-step", "--eval_acc_step",
                   dest="eval_acc_step", type=int, default=1,
                   help="평가 시 accumulation steps")

    # ===== Optim =====
    p.add_argument("--lr", "--learning-rate", "--learning_rate",
                   dest="learning_rate", type=float, default=2e-4, help="러닝레이트")
    p.add_argument("--warmup-ratio", "--warmup_ratio",
                   dest="warmup_ratio", type=float, default=0.1, help="웜업 비율")
    p.add_argument("--use-scheduler", "--use_scheduler",
                   dest="use_scheduler", type=str, default="linear_warmup",
                   help="스케줄러 종류(예: linear_warmup)")

    # ===== Logging / I/O =====
    p.add_argument("--logging-step", "--logging_step",
                   dest="logging_step", type=int, default=50, help="로그 간격(steps)")
    p.add_argument("--output-path", "--output_path",
                   dest="output_path", type=str, default="./saved_model/trainer4-refactor",
                   help="출력 디렉토리")
    p.add_argument("--num-worker", "--num_worker",
                   dest="num_worker", type=int, default=4, help="DataLoader num_workers")

    # ===== Eval/Save strategy =====
    p.add_argument("--eval-strategy", "--eval_strategy",
                   dest="eval_strategy", type=str, default="steps",
                   choices=["no", "steps", "epoch"], help="평가 전략")
    p.add_argument("--eval-steps", "--eval_steps",
                   dest="eval_steps", type=int, default=1000, help="평가 간격(steps)")
    p.add_argument("--save-strategy", "--save_strategy",
                   dest="save_strategy", type=str, default="steps",
                   choices=["no", "steps", "epoch"], help="저장 전략")
    p.add_argument("--save-steps", "--save_steps",
                   dest="save_steps", type=int, default=1000, help="저장 간격(steps)")

    # ===== LoRA (train.py 스타일 기본값 유지) =====
    p.add_argument("--lora-r", "--lora_r",
                   dest="lora_r", type=int, default=8)
    p.add_argument("--lora-alpha", "--lora_alpha",
                   dest="lora_alpha", type=int, default=32)
    p.add_argument("--lora-dropout", "--lora_dropout",
                   dest="lora_dropout", type=float, default=0.05)

    # ===== Quantization =====
    # 쉘에서 ${USE_4BIT} == "--use-4bit" 일 때 true
    p.add_argument("--use-4bit", "--use_4bit",
                   dest="use_4bit", action="store_true", help="4bit 양자화 사용")

    # Quantization
    # p.add_argument("--use_4bit", action="store_true")
    return p.parse_args()


class SimpleTrainer:
    def __init__(self, model, train_loader, accelerator, device, optimizer, args, logger, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.accelerator = accelerator
        self.device = device
        self.optimizer = optimizer
        self.args = args
        self.logger = logger
        self.scheduler = scheduler

    def train(self, epochs):
        for epoch in epochs:
            self.model.train()
            self.optimizer.zero_grad()

            progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}",
                disable=not self.accelerator.is_main_process,
            )

            for step, batch in enumerate(progress):
                loss = self._train_step(batch)

                if step % self.args.logging_step == 0 and step != 0 and self.accelerator.is_main_process:
                    self._log(epoch, step, loss, len(self.train_loader))

            # Save checkpoint per epoch 
            if self.accelerator.is_main_process:
                self._save(epoch)
            self.accelerator.wait_for_everyone()

    def _train_step(self, batch):
        # input_batch = {k: v.to(self.device) for k, v in batch["input_batch"].items()}
        
        # labels = input_batch["input_ids"].clone()
        input_batch = {k: v.to(self.device) for k, v in batch["input_batch"].items()}
        labels = batch["labels"].to(self.device)
        
        
        with self.accelerator.accumulate(self.model):
            outputs = self.model(**input_batch, labels=labels, output_hidden_states=True)
            loss = outputs.loss

            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

        return loss

    def _log(self, epoch, step, loss, total):
        if self.scheduler:
            try:
                lr = self.scheduler.get_last_lr()[0]
            except Exception:
                # 일부 스케줄러는 get_last_lr 미지원
                lr = self.optimizer.param_groups[0].get("lr", 0.0)
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.args.epoch_size}] | "
                f"Iter [{step}/{total}] | loss: {loss.item():.5f} | lr: {lr:.6f}"
            )
        else:
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.args.epoch_size}] | "
                f"Iter [{step}/{total}] | loss: {loss.item():.5f}"
            )

    def _save(self, epoch):
        # adapter/peft 가중치 저장
        out_dir = os.path.join(self.args.output_path, f"{epoch + 1}")
        os.makedirs(out_dir, exist_ok=True)
        model_to_save = self.accelerator.unwrap_model(self.model)

        model_to_save.save_pretrained(
            out_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
        )
        # 구성도 함께
        try:
            model_to_save.config.to_json_file(os.path.join(out_dir, "config.json"))
        except Exception:
            pass

        self.logger.info(f"[trainer4_refactored] Saved to: {out_dir}")


def main():
    args = parse_args()

    # ==== Pretty header ====
    KST = timezone(timedelta(hours=9))
    MODEL_NAME = os.path.basename(os.path.normpath(args.model_path_or_name))
    print("\n======================= MODE & MODEL =======================")
    print(f" TIME:   {datetime.now(KST)}")
    print(" MODE:   TRAIN START")
    print(f" MODEL:  {MODEL_NAME}")
    print(f" LORA:   r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f" LEVEL:  {args.intent_level}")
    print("===========================================================\n")

    os.makedirs(args.output_path, exist_ok=True)
    logger = getLogger(logging_path=os.path.join(args.output_path, "log.log"))
    logger.info("Logger initialized.")

    # ==== Accelerator (FSDP configs align with train.py) ====
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # fsdp_plugin=fsdp_plugin,
    )
    logger.info(f"{AcceleratorState()}")

    # Seed per-rank
    seed_val = args.seed + accelerator.process_index
    set_seed(seed_val)
    logger.info(f"Seed set to {seed_val}")

    # ==== Quantization ====
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # ==== LoRA PEFT via LLMLoader ====
    from peft import LoraConfig
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        # bias="none",
        task_type="CAUSAL_LM",
    )

    # Base model + tokenizer
    base_model_id = args.model_path_or_name
    if "Phi" in base_model_id:
        model_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    else:
        model_config = AutoConfig.from_pretrained(base_model_id)

    loader = LLMLoader(base_model_id, model_config, bnb_config, peft_config, args)
    loader.load()
    tokenizer = loader.get_tokenizer()
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = True
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==== Data (메인 파이프라인과 동일: get_datasetM) ====
    texts, labels = get_datasetM(args.training_data_path, args)
    logger.info(f"Loaded {len(texts)} training samples.")

    train_loader = DataLoader(
        Dataset(texts, labels, mode="train"),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=DataCollator(tokenizer, args.max_source_len, mode="train"),
        pin_memory=True,
    )
    logger.info(f"|Train| steps per epoch: {len(train_loader)}")

    model = loader.get_model()

    # ==== Optimizer & Scheduler ====
    optimizer = bnb.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        optim_bits=8,
        is_paged=True,
    )

    num_updates_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    total_updates = args.epoch_size * num_updates_per_epoch

    if args.use_scheduler == "linear_warmup":
        if accelerator.state.deepspeed_plugin is None or \
           "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config:
            logger.info("Using HF linear warmup scheduler")
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_updates * args.warmup_ratio),
                num_training_steps=total_updates,
            )
        else:
            logger.info("Using DeepSpeed DummyScheduler")
            scheduler = DummyScheduler(
                optimizer,
                total_num_steps=total_updates,
                warmup_num_steps=int(total_updates * args.warmup_ratio),
            )
    else:
        scheduler = None

    # Prepare for distributed
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    device = accelerator.device

    # ==== Logging train config ====
    logger.info(f"  Num examples = {len(train_loader.dataset)}")
    logger.info(f"  Num Epochs = {args.epoch_size}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  World size = {accelerator.state.num_processes}")
    logger.info(f"  Grad Accum = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Total update steps = {total_updates}")

    # ==== Train ====
    epochs = trange(0, int(args.epoch_size), desc="Epoch", disable=not accelerator.is_main_process)
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        accelerator=accelerator,
        device=device,
        optimizer=optimizer,
        args=args,
        logger=logger,
        scheduler=scheduler,
    )
    trainer.train(epochs)


if __name__ == "__main__":
    main()
