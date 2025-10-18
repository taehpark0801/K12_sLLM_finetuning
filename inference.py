#!/usr/bin/env python3
import os, json, argparse, csv
from datetime import datetime, timezone, timedelta

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROMPT_TEMPLATE = (
    "너는 국어 교과 평가를 담당하는 채점자이다.\n"
    "아래 제시문을 읽고 학생의 답변이 (가)와 (나)의 목적을 얼마나 정확히 파악했는지 0~2점으로 채점하라.\n"
    "출력은 오직 \"0점\", \"1점\", \"2점\" 중 하나만 써라. 다른 설명이나 이유는 절대 쓰지 마라.\n\n"
    "[제시문 요약]\n"
    "(가): 과장·허위 광고의 문제점을 알려 주고 주의시키는 내용\n"
    "(나): '쑥쑥 주스'를 홍보하고 판매하려는 광고\n\n"
    "[채점 기준]\n"
    "2점: (가)와 (나)의 목적을 모두 정확히 파악함\n"
    "1점: 둘 중 하나만 정확히 파악함\n"
    "0점: 둘 다 틀리거나 목적을 제대로 파악하지 못함\n"
    "의미가 같으면 표현이 달라도 정답으로 인정하라.\n\n"
    "[채점결과 예시]\n"
    "0점\n\n"
    "[학생 답변]\n"
    "{response}\n\n"
    "[채점 결과]\n"
)

def build_prompt(resp: str) -> str:
    return PROMPT_TEMPLATE.format(response=resp if resp is not None else "")

def parse_args():
    p = argparse.ArgumentParser(description="Batch inference: JSON -> CSV")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--input-json", required=True, help="배열 JSON 파일 (fields: response, point)")
    p.add_argument("--output-csv", required=True, help="저장할 CSV 경로")
    p.add_argument("--tokenizer", default="")
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    return p.parse_args()

def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Tokenizer
    tok_name = args.tokenizer or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Base model (optional 4bit)
    quant = None
    if args.use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    # Load JSON
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "input JSON은 리스트여야 합니다."

    # Run & write CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["response", "gold_point", "gold_label", "pred_label", "raw_output"])

        for item in data:
            resp = item.get("response", "")
            gold_point = item.get("point", None)
            gold_label = f"{gold_point}점" if gold_point is not None else ""

            prompt = build_prompt(resp)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            gen = model.generate(
                **inputs,
                do_sample=(args.temperature > 0.0),
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            out = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # print(f"Response: {out}")
            s = out.strip().splitlines()
            pred = s[0].strip() if s else out.strip()
            # print(f"Predicted Label: {pred}")

            # normalize to 0점/1점/2점
            if "2" in pred:
                pred_norm = "2점"
            elif "1" in pred:
                pred_norm = "1점"
            elif "0" in pred:
                pred_norm = "0점"
            else:
                pred_norm = pred  # 혹시 정확히 출력했으면 그대로

            writer.writerow([resp, gold_point, gold_label, pred_norm, out])

if __name__ == "__main__":
    main()
