#!/usr/bin/env python
"""
Per-entity ("macro") evaluation of prediction file(s).

Each prediction CSV must contain:
  - a 'label'       column  (ground truth)
  - a 'prediction'  column  (predicted value; probabilities for classification)
  - a grouping column identifying the entity to evaluate per
                            (cell_iname, target_sequence, or gene_name)

For each file the metric is computed PER ENTITY and then averaged across entities;
the per-entity table is saved as well. The task type and grouping column are
auto-inferred (from the file name for the 7 InsilicoCell tasks, otherwise from the
data). The 14 files in ./prediction are reference examples of the expected format.

  - regression tasks     -> Pearson r, RMSE
  - classification tasks -> AUROC, macro-F1 (threshold 0.5 on the probability)

Requirements: pandas, numpy, scikit-learn

Usage
-----
python evaluate_predictions.py --files my_predictions.csv --out_dir ./evaluation
python evaluate_predictions.py --files pred1.csv pred2.csv --out_dir ./evaluation
"""
import argparse, os, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error

# Known InsilicoCell tasks: file-name token -> (grouping column, metric family).
KNOWN = {
    "drug_sensitivity":                    ("cell_iname",      "reg"),
    "drug-induced_gene_expression_change": ("cell_iname",      "reg"),
    "gene_effect_score":                   ("cell_iname",      "reg"),
    "CNV":                                 ("cell_iname",      "reg"),
    "drug-protein_binding":                ("target_sequence", "reg"),
    "gene_mutation":                       ("cell_iname",      "clf"),
    "TF-gene_association":                 ("target_sequence", "clf"),
}
GROUP_CANDIDATES = ["cell_iname", "target_sequence", "gene_name"]
THRESHOLD = 0.5

# ---- metrics (pandas .corr + sklearn, same as the reference pipeline) ----
def reg_group(y, p):
    return {"pearson_r": pd.Series(y).corr(pd.Series(p), method="pearson"),   # NaN if <2 pts / zero variance
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}

def clf_group(y, p):
    if len(np.unique(y)) < 2:             # AUROC/macro-F1 need both classes present
        return {"AUROC": np.nan, "macro_F1": np.nan}
    return {"AUROC": roc_auc_score(y, p),
            "macro_F1": f1_score(y, (p > THRESHOLD).astype(int), average="macro")}

def detect_task(path):
    t = re.sub(r"_(sample|entity)-level.*$", "", re.sub(r"^pred_", "", os.path.basename(path)[:-4]))
    return next((k for k in KNOWN if k in t), None)

def is_binary(s):
    return set(pd.unique(s.dropna())).issubset({0, 1, 0.0, 1.0})

def evaluate_file(path, out_dir):
    name = os.path.basename(path)
    df = pd.read_csv(path)
    if "label" not in df.columns or "prediction" not in df.columns:
        print(f"[SKIP] {name}: needs 'label' and 'prediction' columns. Found: {list(df.columns)}")
        return None

    task = detect_task(path)
    gcol = KNOWN[task][0] if task in KNOWN else next((c for c in GROUP_CANDIDATES if c in df.columns), None)
    if not gcol:
        print(f"[SKIP] {name}: no grouping column (cell_iname/target_sequence/gene_name). Found: {list(df.columns)}")
        return None
    kind = KNOWN[task][1] if task in KNOWN else ("clf" if is_binary(df["label"]) else "reg")

    d = df[[gcol, "label", "prediction"]].replace([np.inf, -np.inf], np.nan).dropna()
    if task == "drug_sensitivity":                                  # evaluate drug sensitivity on ln(x+1) scale
        d["label"] = np.log1p(d["label"]); d["prediction"] = np.log1p(d["prediction"])
    rows = []
    for gid, g in d.groupby(gcol, sort=False):
        y = g["label"].to_numpy(float); p = g["prediction"].to_numpy(float)
        rows.append({gcol: gid, **(reg_group(y, p) if kind == "reg" else clf_group(y, p))})
    per = pd.DataFrame(rows)
    metric_cols = [c for c in per.columns if c != gcol]
    per = per.dropna(subset=metric_cols).reset_index(drop=True)     # drop entities with undefined metrics

    os.makedirs(os.path.join(out_dir, "per_entity"), exist_ok=True)
    per.to_csv(os.path.join(out_dir, "per_entity", f"{name[:-4]}_per_{gcol}.csv"), index=False)

    summary = {"file": name, "task": task or "custom", "type": kind, "group_by": gcol, "n_groups": len(per)}
    for c in metric_cols:
        summary[f"mean_{c}"] = per[c].mean()                        # mean across entities (no std)
    note = " | ln(x+1) transform" if task == "drug_sensitivity" else ""
    print(f"[OK]   {name}: {kind} | grouped by {gcol} | {len(per)} entities{note}")
    return summary

def main():
    ap = argparse.ArgumentParser(description="Per-entity evaluation of your prediction file(s).")
    ap.add_argument("--files", nargs="+", required=True, help="Your prediction CSV file(s).")
    ap.add_argument("--out_dir", default="./evaluation", help="Where to write results.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    res = pd.DataFrame([r for r in (evaluate_file(f, args.out_dir) for f in args.files) if r is not None])
    if res.empty:
        raise SystemExit("Nothing evaluated.")
    res.to_csv(os.path.join(args.out_dir, "summary_mean_across_entities.csv"), index=False)

    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_colwidth", 60,
                  "display.float_format", lambda v: f"{v:.4f}")
    reg, clf = res[res.type == "reg"], res[res.type == "clf"]
    if len(reg):
        print("\n== REGRESSION (per-entity Pearson & RMSE, averaged over entities) ==")
        print(reg[["file","group_by","n_groups","mean_pearson_r","mean_RMSE"]].to_string(index=False))
    if len(clf):
        print("\n== CLASSIFICATION (per-entity AUROC & macro-F1, averaged over entities) ==")
        print(clf[["file","group_by","n_groups","mean_AUROC","mean_macro_F1"]].to_string(index=False))
    print(f"\nSaved: {os.path.join(args.out_dir, 'summary_mean_across_entities.csv')}")
    for _, row in res.iterrows():
        print("       " + os.path.join(args.out_dir, "per_entity", f"{row['file'][:-4]}_per_{row['group_by']}.csv"))

if __name__ == "__main__":
    main()
