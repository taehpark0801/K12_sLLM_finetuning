#!/usr/bin/env python3
import argparse, re, json
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

def to_num(x):
    """ '0점', '1점', '2점', '0', 0 등 -> 0/1/2 로 정규화 """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        try:
            xi = int(x)
            return xi if xi in (0,1,2) else None
        except:
            return None
    s = str(x).strip()
    m = re.search(r"[012]", s)
    return int(m.group()) if m else None

def main():
    ap = argparse.ArgumentParser(description="Compute Cohen's kappa from out.csv")
    ap.add_argument("--csv", required=True, help="out.csv 경로")
    ap.add_argument("--save-report", default="", help="리포트를 JSON으로 저장할 경로(옵션)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # 예상 컬럼: response, gold_point, gold_label, pred_label, raw_output
    if "gold_point" not in df.columns or "pred_label" not in df.columns:
        raise SystemExit("CSV에 gold_point / pred_label 컬럼이 필요합니다.")

    df["gold"] = df["gold_point"].apply(to_num)
    df["pred"] = df["pred_label"].apply(to_num)

    before = len(df)
    df = df.dropna(subset=["gold","pred"])
    df["gold"] = df["gold"].astype(int)
    df["pred"] = df["pred"].astype(int)

    if df.empty:
        raise SystemExit("유효한 (gold, pred) 샘플이 없습니다.")

    # Metrics
    kappa = cohen_kappa_score(df["gold"], df["pred"])  # 일반(비가중)
    kappa_quadratic = cohen_kappa_score(df["gold"], df["pred"], weights="quadratic")
    acc = accuracy_score(df["gold"], df["pred"])
    cm = confusion_matrix(df["gold"], df["pred"], labels=[0,1,2])

    # Print
    print(f"Samples (total/used): {before} / {len(df)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's kappa (unweighted): {kappa:.4f}")
    print(f"Cohen's kappa (quadratic): {kappa_quadratic:.4f}")
    print("\nConfusion Matrix (rows=gold 0/1/2, cols=pred 0/1/2):")
    print(pd.DataFrame(cm, index=[0,1,2], columns=[0,1,2]).to_string())

    # Optional save
    if args.save_report:
        report = {
            "num_total": int(before),
            "num_used": int(len(df)),
            "accuracy": acc,
            "kappa_unweighted": kappa,
            "kappa_quadratic": kappa_quadratic,
            "confusion_matrix": cm.tolist(),
        }
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report -> {args.save_report}")

if __name__ == "__main__":
    main()
