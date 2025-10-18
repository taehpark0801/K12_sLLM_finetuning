# dataset.py

import torch

class Dataset(torch.utils.data.Dataset):
    """
    trainer.py에서 get_datasetM으로 만든 texts(labels)
    반환: {"text": prompt, "label": "0점"/"1점"/"2점"}
    """
    def __init__(self, texts, labels, mode="train"):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.mode = mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}


class DataCollator:
    """
    - 프롬프트 구간은 label을 -100으로 마스킹
    - 정답("0점/1점/2점") 토큰만 loss에 기여
    - 입력 시퀀스 끝에 정답 토큰도 실제로 붙여
    """
    def __init__(self, tokenizer, max_source_len, mode="train"):
        self.tok = tokenizer
        self.max_source_len = max_source_len
        self.mode = mode

    def __call__(self, batch):
        prompts = [ex["text"] for ex in batch]
        targets = [ex["label"] for ex in batch]   # "0점"/"1점"/"2점"

        enc = self.tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_source_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # 타깃 토큰화 (아주 짧음)
        with self.tok.as_target_tokenizer():
            tgt = self.tok(targets, padding=False, add_special_tokens=False)

        # 라벨 초기화: 프롬프트는 -100
        labels = torch.full_like(input_ids, fill_value=-100)

        # 각 샘플마다 프롬프트 뒤에 정답 토큰을 실제 입력에도 붙이고,
        # 그 위치에만 라벨을 채워 loss를 주게 함
        eos_id = self.tok.eos_token_id
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos_id

        new_input_ids = []
        new_attention = []
        new_labels = []

        for i in range(len(prompts)):
            src_ids = input_ids[i]
            src_attn = attention_mask[i]

            tgt_ids = tgt["input_ids"][i] + [eos_id]
            tgt_tensor = torch.tensor(tgt_ids, dtype=src_ids.dtype)

            # 프롬프트+정답 이어붙이기 (길이 자르기)
            concat_ids = torch.cat([src_ids, tgt_tensor])
            concat_attn = torch.cat([src_attn, torch.ones_like(tgt_tensor)])

            concat_ids = concat_ids[:self.max_source_len]
            concat_attn = concat_attn[:self.max_source_len]

            # 새 labels: 기본 -100, 끝쪽 '정답 구간'만 실제 토큰
            lab = torch.full_like(concat_ids, fill_value=-100)
            # 유효 길이
            valid_len = int(concat_attn.sum().item())
            # 붙은 정답 길이 (자르며 일부 잘릴 수 있음)
            tgt_len = min(len(tgt_ids), valid_len)  # 안전
            # 끝에서 tgt_len 범위만 라벨 지정
            lab[valid_len - tgt_len:valid_len] = concat_ids[valid_len - tgt_len:valid_len]

            new_input_ids.append(concat_ids)
            new_attention.append(concat_attn)
            new_labels.append(lab)

        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention, dim=0)
        labels = torch.stack(new_labels, dim=0)

        return {
            "input_batch": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            "labels": labels,
        }
