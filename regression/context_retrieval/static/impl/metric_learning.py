from __future__ import annotations

import os
import glob
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

from regression.utils import read_data_csv
from regression.context_retrieval import StaticContextRetrieval, Context, ContextSimilarity


class MetricLearningRetrieval(StaticContextRetrieval):
    """
    Triplet metric-learning retriever.

    - Learns a Siamese encoder+head that outputs a similarity score s(i,t) in [0,1].
    - Trains with triplet loss (anchor=target context, pos=similar source, neg=dissimilar source).
    - Retrieves contexts by sorting distance = 1 - score (smaller is better).

    Expected training labels file:
      train_dir/concordant_similarity.csv
        columns: target_id, source_id, similarity   (similarity in [0,1])

    Expected context_id format in that file:
      "{workload_label}__{hardware_label}"  where hardware_label is like "16c64g".

    LOOCV:
      loocv() retrains for each target context id, holding out all rows with target_id==that id
      AND all rows with source_id==that id (so the held-out context is never seen).
      Then it evaluates ranking quality on the held-out target:
        - Spearman correlation between predicted scores and true similarities
        - Hit@K for K in {1,3,5,10}
        - Mean rank of the best true source (oracle) under predicted ranking
    """

    _model = None
    _scaler = MinMaxScaler()

    # ---------- NN components ----------
    class Encoder(nn.Module):
        def __init__(self, input_dim: int, emb_dim: int = 64, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, emb_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class Head(nn.Module):
        """Learned similarity head -> sigmoid -> score in [0,1]."""
        def __init__(self, emb_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4 * emb_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
            u = torch.cat([z_a, z_b, torch.abs(z_a - z_b), z_a * z_b], dim=1)
            return torch.sigmoid(self.net(u)).squeeze(1)

    class Siamese(nn.Module):
        def __init__(self, input_dim: int, emb_dim: int = 64, enc_hidden: int = 128, head_hidden: int = 128):
            super().__init__()
            self.encoder = MetricLearningRetrieval.Encoder(input_dim, emb_dim, enc_hidden)
            self.head = MetricLearningRetrieval.Head(emb_dim, head_hidden)

        def score(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
            z_a = self.encoder(x_a)
            z_b = self.encoder(x_b)
            return self.head(z_a, z_b)

    class TripletLoss(nn.Module):
        def __init__(self, margin: float = 0.2):
            super().__init__()
            self.margin = margin

        def forward(self, s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
            return torch.mean(F.relu(self.margin - s_pos + s_neg))

    class TripletDataset(Dataset):
        def __init__(self, triplets: pd.DataFrame, id2x: Dict[str, np.ndarray], device: torch.device):
            self.triplets = triplets.reset_index(drop=True)
            self.id2x = id2x
            self.device = device

        def __len__(self) -> int:
            return len(self.triplets)

        def __getitem__(self, idx: int):
            r = self.triplets.iloc[idx]
            xa = torch.tensor(self.id2x[r["anchor_id"]], dtype=torch.float32, device=self.device)
            xp = torch.tensor(self.id2x[r["pos_id"]], dtype=torch.float32, device=self.device)
            xn = torch.tensor(self.id2x[r["neg_id"]], dtype=torch.float32, device=self.device)
            return xa, xp, xn

    # ---------- API ----------
    def __init__(
        self,
        system,
        data_dir: str,
        train_dir: str,
        seed: int = 42,
        excluding_factors: List[str] = ["hardware", "workload"],
    ):
        self.system = system
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.excluding_factors = excluding_factors
        self.config: Optional[dict] = None
        self.set_seed(seed)

    @property
    def model(self):
        return MetricLearningRetrieval._model

    @model.setter
    def model(self, val):
        MetricLearningRetrieval._model = val

    @property
    def scaler(self):
        return MetricLearningRetrieval._scaler

    def set_hyper_parameters(self, config: dict):
        self.config = config

    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---------- feature handling ----------
    def _feature_cols(self) -> List[str]:
        return [
            "tps",
            "Average Memory Usage Percentage",
            "InnoDB Buffer Pool Cache Hit Rate",
            "InnoDB Dirty Buffer Pages",
            "Current QPS (Queries Per Second)",
            "Max CPU Usage (100 - Idle)",
            "InnoDB Rows Deleted (60s Rate)", "InnoDB Rows Inserted (60s Rate)",
            "InnoDB Rows Read (60s Rate)", "InnoDB Rows Updated (60s Rate)",
            "Average Disk IOPS (Read)", "Average Disk IOPS (Write)",
        ]

    def _context_id(self, wl: str, hw: str) -> str:
        return f"{wl}__{hw}"

    def get_sample(self, csv_path: str, workload_label: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df = df[df["workload_label"] == workload_label]
        if len(df) < 2:
            raise ValueError("Not enough rows after filtering workload_label.")
        return df.iloc[[1]]

    def _build_id2x(self, fit_scaler: bool) -> Dict[str, np.ndarray]:
        feat_cols = self._feature_cols()
        rows: List[np.ndarray] = []
        ids: List[str] = []

        for fp in glob.glob(os.path.join(self.data_dir, "*.csv")):
            if os.path.basename(fp) == "train.csv":
                continue
            df = pd.read_csv(fp)
            if len(df) == 0 or "workload_label" not in df.columns:
                continue

            df = df.drop(columns=df.columns[df.isnull().any()], errors="ignore")
            hw = f"{df['num_cpu'].iloc[0]}c{df['mem_size'].iloc[0]}g"

            for wl in df["workload_label"].unique():
                try:
                    sample = self.get_sample(fp, wl)
                except Exception:
                    continue
                cols = [c for c in feat_cols if c in sample.columns]
                if not cols:
                    continue
                x = sample[cols].astype(float).iloc[0].values
                rows.append(x)
                ids.append(self._context_id(wl, hw))

        if not rows:
            raise RuntimeError("No contexts found in data_dir to build features.")

        X = np.vstack(rows)
        Xs = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        return {ids[i]: Xs[i] for i in range(len(ids))}

    def _build_triplets(self, scores_df: pd.DataFrame, pos_k: int, neg_k: int, max_triplets_per_target: int) -> pd.DataFrame:
        triplets = []
        for tid, g in scores_df.groupby("target_id"):
            g = g.sort_values("similarity", ascending=False)
            pos = g.head(pos_k)["source_id"].tolist()
            neg = g.tail(neg_k)["source_id"].tolist()
            if not pos or not neg:
                continue
            for _ in range(max_triplets_per_target):
                triplets.append({"anchor_id": tid, "pos_id": random.choice(pos), "neg_id": random.choice(neg)})
        if not triplets:
            raise RuntimeError("No triplets built. Check concordant_similarity.csv content.")
        return pd.DataFrame(triplets)

    # ---------- internal: fit model on provided label df ----------
    def _fit_from_scores_df(self, scores_df: pd.DataFrame, id2x: Dict[str, np.ndarray]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = self.config or {
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 50,
            "margin": 0.2,
            "emb_dim": 64,
            "enc_hidden": 128,
            "head_hidden": 128,
            "pos_k": 5,
            "neg_k": 5,
            "max_triplets_per_target": 50,
        }

        # Keep only ids we can featurize
        scores_df = scores_df[
            scores_df["target_id"].isin(id2x.keys()) & scores_df["source_id"].isin(id2x.keys())
        ].copy()

        triplets = self._build_triplets(
            scores_df,
            pos_k=int(cfg["pos_k"]),
            neg_k=int(cfg["neg_k"]),
            max_triplets_per_target=int(cfg["max_triplets_per_target"]),
        )

        input_dim = len(next(iter(id2x.values())))
        model = self.Siamese(
            input_dim=input_dim,
            emb_dim=int(cfg["emb_dim"]),
            enc_hidden=int(cfg["enc_hidden"]),
            head_hidden=int(cfg["head_hidden"]),
        ).to(device)

        loss_fn = self.TripletLoss(margin=float(cfg["margin"]))
        opt = optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

        ds = self.TripletDataset(triplets, id2x, device=device)
        dl = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True)

        for epoch in range(int(cfg["num_epochs"])):
            model.train()
            tot, n = 0.0, 0
            for xa, xp, xn in dl:
                opt.zero_grad()
                s_pos = model.score(xa, xp)
                s_neg = model.score(xa, xn)
                loss = loss_fn(s_pos, s_neg)
                loss.backward()
                opt.step()
                tot += float(loss.item()) * xa.size(0)
                n += xa.size(0)
            # Keep prints minimal during sweeps; comment out if needed
            # print(f"Epoch [{epoch+1}/{cfg['num_epochs']}], Loss: {tot/max(n,1):.6f}")

        return model

    # ---------- training ----------
    def train_model(self):
        sim_path = os.path.join(self.train_dir, "concordant_similarity.csv")
        if not os.path.exists(sim_path):
            raise FileNotFoundError(
                f"Missing {sim_path}. Expected columns: target_id, source_id, similarity."
            )
        scores_df = pd.read_csv(sim_path)

        # Build context vectors and fit scaler ONCE for this training
        id2x = self._build_id2x(fit_scaler=True)

        model = self._fit_from_scores_df(scores_df, id2x)
        self.model = model
        return model

    # ---------- retrieval ----------
    def _prepare_target(self, target_data_path: str, workload_label: str) -> Tuple[str, str, np.ndarray]:
        file = os.path.basename(target_data_path)
        metric_path = os.path.join(self.data_dir, file)

        sample = self.get_sample(metric_path, workload_label)
        hw = f"{sample['num_cpu'].iloc[0]}c{sample['mem_size'].iloc[0]}g"
        tid = self._context_id(workload_label, hw)

        cols = [c for c in self._feature_cols() if c in sample.columns]
        x = sample[cols].astype(float).iloc[0].values.reshape(1, -1)

        # scaler should already be fit if train_model ran; if not, fit quickly.
        try:
            xs = self.scaler.transform(x).flatten()
        except Exception:
            _ = self._build_id2x(fit_scaler=True)
            xs = self.scaler.transform(x).flatten()

        return tid, hw, xs

    def retrieve_contexts(self, target_context: Context) -> List[ContextSimilarity]:
        if self.model is None:
            self.train_model()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        target_data_path = target_context.hardware
        wl_t = target_context.workload

        _, hw_t, x_t = self._prepare_target(target_data_path, wl_t)
        x_t = torch.tensor(x_t, dtype=torch.float32, device=device).unsqueeze(0)

        id2x = self._build_id2x(fit_scaler=False)

        results: List[ContextSimilarity] = []
        self.model.eval()
        with torch.no_grad():
            for cid, x_i_np in id2x.items():
                wl_i, hw_i = cid.split("__", 1)

                if "workload" in self.excluding_factors and wl_i == wl_t:
                    continue
                if "hardware" in self.excluding_factors and hw_i == hw_t:
                    continue

                x_i = torch.tensor(x_i_np, dtype=torch.float32, device=device).unsqueeze(0)
                score = self.model.score(x_t, x_i)  # (1,) in [0,1]
                dist = 1.0 - score                  # concordant-pair distance

                param_df_path = f"dataset/transfer_learning/mysql/chimera_tech/{hw_i}-result.csv"
                param_df = read_data_csv(param_df_path, self.system, wl_i)

                d = float(dist.item())
                s = float(score.item())
                results.append(ContextSimilarity(param_df, wl_i, hw_i, d, s))

        results.sort(key=lambda r: r.distance)
        return results

    # ============================================================
    # LOOCV: retrain leaving out each target context and evaluate
    # ============================================================
    @staticmethod
    def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
        """Spearman correlation without scipy (handles ties reasonably via rankdata-ish approach)."""
        if len(a) < 2:
            return float("nan")

        def rank(x: np.ndarray) -> np.ndarray:
            # average ranks for ties
            order = np.argsort(x)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(x), dtype=float)

            # tie handling
            sorted_x = x[order]
            i = 0
            while i < len(x):
                j = i
                while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
                    j += 1
                if j > i:
                    r = (i + j) / 2.0
                    ranks[order[i:j + 1]] = r
                i = j + 1
            return ranks

        ra = rank(a)
        rb = rank(b)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = (np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum()))
        if denom == 0:
            return float("nan")
        return float((ra * rb).sum() / denom)

    def loocv(
        self,
        topk_list: List[int] = [1, 3, 5, 10],
        verbose: bool = True,
        save_csv: Optional[str] = "loocv_metric_learning.csv",
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Leave-One-Context-Out CV.

        For each held-out context id T:
          - train on all pairs where target_id!=T AND source_id!=T
          - evaluate on all rows where target_id==T:
              predicted score for each source_id
              compare with true similarity:
                - Spearman correlation (higher is better)
                - Hit@K: whether the *best-true* source appears in top-K predicted
                - Oracle-best rank under predicted (1 is best)

        Returns:
          df (per-target metrics), summary dict (mean metrics)
        """
        sim_path = os.path.join(self.train_dir, "concordant_similarity.csv")
        if not os.path.exists(sim_path):
            raise FileNotFoundError(
                f"Missing {sim_path}. Expected columns: target_id, source_id, similarity."
            )
        all_scores = pd.read_csv(sim_path)

        # Fit scaler once on all contexts (stable features across folds)
        id2x = self._build_id2x(fit_scaler=True)

        # Keep only ids we can featurize
        all_scores = all_scores[
            all_scores["target_id"].isin(id2x.keys()) & all_scores["source_id"].isin(id2x.keys())
        ].copy()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        per_target_rows = []
        targets = sorted(all_scores["target_id"].unique().tolist())

        for ti, target_id in enumerate(targets):
            test_df = all_scores[all_scores["target_id"] == target_id].copy()
            if len(test_df) < 3:
                # Too few candidates -> unreliable ranks
                continue

            train_df = all_scores[
                (all_scores["target_id"] != target_id) & (all_scores["source_id"] != target_id)
            ].copy()

            # Train a fresh model for this fold
            self.set_seed(42)  # keep deterministic across folds; change if you want stochastic
            fold_model = self._fit_from_scores_df(train_df, id2x)
            fold_model.eval()

            # Predict scores for all sources in this target's test set
            x_t = torch.tensor(id2x[target_id], dtype=torch.float32, device=device).unsqueeze(0)

            preds = []
            trues = []
            src_ids = test_df["source_id"].tolist()
            true_sims = test_df["similarity"].astype(float).values

            with torch.no_grad():
                for sid in src_ids:
                    x_s = torch.tensor(id2x[sid], dtype=torch.float32, device=device).unsqueeze(0)
                    s_hat = float(fold_model.score(x_t, x_s).item())
                    preds.append(s_hat)

            preds = np.array(preds, dtype=float)
            trues = np.array(true_sims, dtype=float)

            # Spearman between predicted and true similarity over sources for this target
            rho = self._spearman_corr(preds, trues)

            # Find oracle best source by TRUE similarity, then get its rank under predicted
            oracle_best_idx = int(np.argmax(trues))
            oracle_best_sid = src_ids[oracle_best_idx]

            pred_order = np.argsort(-preds)  # descending predicted similarity
            oracle_rank = int(np.where(np.array(src_ids)[pred_order] == oracle_best_sid)[0][0]) + 1  # 1-based

            # Hit@K
            hit_at = {}
            for k in topk_list:
                k = int(k)
                topk_sids = set(np.array(src_ids)[pred_order[:k]].tolist())
                hit_at[f"hit@{k}"] = 1.0 if oracle_best_sid in topk_sids else 0.0

            row = {
                "target_id": target_id,
                "num_candidates": len(src_ids),
                "spearman": rho,
                "oracle_rank": oracle_rank,
                **hit_at,
            }
            per_target_rows.append(row)

            if verbose and (ti % 10 == 0 or ti == len(targets) - 1):
                print(f"[LOOCV] {ti+1}/{len(targets)} done")

        df = pd.DataFrame(per_target_rows)

        summary = {}
        if len(df) > 0:
            summary["n_targets"] = int(len(df))
            summary["spearman_mean"] = float(df["spearman"].mean(skipna=True))
            summary["oracle_rank_mean"] = float(df["oracle_rank"].mean())
            for k in topk_list:
                summary[f"hit@{int(k)}_mean"] = float(df[f"hit@{int(k)}"].mean())
        else:
            summary["n_targets"] = 0

        if save_csv is not None:
            df.to_csv(save_csv, index=False)
            if verbose:
                print(f"[LOOCV] saved: {save_csv}")

        if verbose:
            print("[LOOCV] summary:", summary)

        return df, summary


if __name__ == "__main__":
    import argparse

    def main():
        parser = argparse.ArgumentParser(
            description="MetricLearningRetrieval: retrieve contexts and compare predicted vs true concordant similarity."
        )
        parser.add_argument("--data_dir", type=str, required=True, help="Directory containing per-hardware CSV metric files.")
        parser.add_argument("--train_dir", type=str, required=True, help="Directory containing concordant_similarity.csv.")
        parser.add_argument("--target_csv", type=str, required=True, help="Path to the target CSV file (often one of data_dir/*.csv).")
        parser.add_argument("--workload", type=str, required=True, help="Target workload_label.")
        parser.add_argument("--topk", type=int, default=10, help="Number of retrieved contexts to show.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        args = parser.parse_args()

        # ---- Minimal system stub (replace with your actual system if needed) ----
        class _DummySystem:
            pass

        system = _DummySystem()

        retriever = MetricLearningRetrieval(
            system=system,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            seed=args.seed,
            excluding_factors=["hardware", "workload"],
        )

        # --- Load ground-truth similarity table ---
        sim_path = os.path.join(args.train_dir, "concordant_similarity.csv")
        if not os.path.exists(sim_path):
            raise FileNotFoundError(f"Missing {sim_path}")
        sim_df = pd.read_csv(sim_path)
        needed = {"target_id", "source_id", "similarity"}
        if not needed.issubset(sim_df.columns):
            raise ValueError(f"{sim_path} must contain columns {sorted(needed)}")

        # --- Build target_id exactly the same way as the retriever does ---
        target_input = retriever.get_sample(args.target_csv, args.workload)
        hardware_label = f"{target_input['num_cpu'].iloc[0]}c{target_input['mem_size'].iloc[0]}g"
        target_id = retriever._context_id(args.workload, hardware_label)

        # --- Retrieve contexts (predicted ranking) ---
        target_ctx = Context(hardware=args.target_csv, workload=args.workload)
        results = retriever.retrieve_contexts(target_ctx)

        # Helper: get true similarity from table (NaN if missing)
        def true_sim_for(source_hw: str, source_wl: str) -> float:
            source_id = retriever._context_id(source_wl, source_hw)
            row = sim_df[(sim_df["target_id"] == target_id) & (sim_df["source_id"] == source_id)]
            if len(row) == 0:
                return float("nan")
            return float(row["similarity"].iloc[0])

        print("\n=== Retrieved contexts (top-k) ===")
        for i, r in enumerate(results[: args.topk], start=1):
            ts = true_sim_for(r.hardware, r.workload)
            print(
                f"{i:02d}  workload={r.workload:20s}  hardware={r.hardware:10s}  "
                f"pred={r.score:.6f}  true={ts:.6f}  dist={r.distance:.6f}"
            )

        # --- (1) Compare for the retrieved top-1 context ---
        top1 = results[0]
        top1_true = true_sim_for(top1.hardware, top1.workload)
        top1_pred = float(top1.score)

        # --- (2) Find the actually most similar context (by TRUE similarity) among the same candidate set ---
        # Build a candidate set matching retrieve_contexts() filtering (excluding same workload/hardware if configured)
        # We’ll use the IDs present in concordant_similarity.csv for this target_id, then filter.
        cand = sim_df[sim_df["target_id"] == target_id].copy()
        if len(cand) == 0:
            print("\nNo ground-truth rows found for this target_id in concordant_similarity.csv.")
            return

        # Apply the same excluding_factors rules
        t_wl = args.workload
        t_hw = hardware_label
        # source_id format: "{wl}__{hw}"
        cand["source_wl"] = cand["source_id"].apply(lambda s: s.split("__", 1)[0])
        cand["source_hw"] = cand["source_id"].apply(lambda s: s.split("__", 1)[1])

        if "workload" in retriever.excluding_factors:
            cand = cand[cand["source_wl"] != t_wl]
        if "hardware" in retriever.excluding_factors:
            cand = cand[cand["source_hw"] != t_hw]

        if len(cand) == 0:
            print("\nAfter excluding_factors, no candidates remain in ground-truth table.")
            return

        # Find the oracle-best by true similarity
        oracle_row = cand.iloc[int(np.argmax(cand["similarity"].astype(float).values))]
        oracle_wl = str(oracle_row["source_wl"])
        oracle_hw = str(oracle_row["source_hw"])
        oracle_true = float(oracle_row["similarity"])

        # Predict similarity for the oracle-best using the trained model directly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure model exists (retrieve_contexts already trains if needed)
        retriever.model.eval()

        # Build scaled vectors using the same scaler as the retriever
        id2x = retriever._build_id2x(fit_scaler=False)  # uses existing fitted scaler
        oracle_id = retriever._context_id(oracle_wl, oracle_hw)
        if oracle_id not in id2x:
            oracle_pred = float("nan")
        else:
            x_t = torch.tensor(id2x[target_id], dtype=torch.float32, device=device).unsqueeze(0)
            x_o = torch.tensor(id2x[oracle_id], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                oracle_pred = float(retriever.model.score(x_t, x_o).item())

        print("\n=== Comparison summary ===")
        print(f"Target: workload={args.workload} hardware={hardware_label}  (target_id={target_id})")

        print("\n[Retrieved top-1]")
        print(f"  context: workload={top1.workload} hardware={top1.hardware}")
        print(f"  predicted similarity: {top1_pred:.6f}")
        print(f"  true similarity:      {top1_true:.6f}")

        print("\n[Oracle (true most similar)]")
        print(f"  context: workload={oracle_wl} hardware={oracle_hw}")
        print(f"  predicted similarity: {oracle_pred:.6f}")
        print(f"  true similarity:      {oracle_true:.6f}")

        # Optional: show whether retrieved top-1 matches oracle
        match = (top1.workload == oracle_wl) and (top1.hardware == oracle_hw)
        print(f"\nTop-1 matches oracle? {match}")

    main()
