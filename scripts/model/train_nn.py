"""
Train an MLP for SDS source prediction.

Uses the same data pipeline as the RF scripts (FeatureCache, build_train_set,
build_eval_set), with a PyTorch MLP instead of a Random Forest.

Defaults come from config.py and are overridable via CLI.

Usage:
  python train_nn.py                              # all from config
  python train_nn.py --epochs 100 --lr 5e-4       # overrides
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from sds.config import (
    VAL_CALIB_FRAC, VAL_SPLIT_SEED, MODELS_PATH, RESOLUTION,
    SPATIAL_RES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    NN_SAMPLING, NN_PARAMS, NN_TAG,
)
from sds.paths import resolve_path, project_relative, SDS_PATH
from sds.log_utils import header, section, done
from sds.data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    collect_indices, build_train_set, build_eval_set,
    platt_fit, platt_apply, compute_metrics, find_best_threshold,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SDSNet(nn.Module):
    """
    Simple MLP for binary SDS classification.
    input → hidden layers with BatchNorm + ReLU + Dropout → logit
    """
    def __init__(self, n_features, hidden_sizes=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict_proba_nn(model, X_tensor, device, batch_size=8192):
    model.eval()
    probas = []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device)
        logits = model(batch)
        probas.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train neural network for SDS prediction")

    # Data / paths
    parser.add_argument("--sds_path", type=str, default=None,
                        help="Path to SDS file (default: config.SDS_PATH)")
    parser.add_argument("--resolution", default=None,
                        help=f"Temporal resolution (default: {RESOLUTION})")
    parser.add_argument("--spatial", type=float, default=None,
                        help=f"Spatial resolution (default: {SPATIAL_RES})")
    parser.add_argument("--train_years", nargs=2, type=int, default=None,
                        help=f"Train year range (default: {TRAIN_YEARS})")
    parser.add_argument("--val_years", nargs=2, type=int, default=None,
                        help=f"Val year range (default: {VAL_YEARS})")
    parser.add_argument("--test_years", nargs=2, type=int, default=None,
                        help=f"Test year range (default: {TEST_YEARS})")

    # Sampling overrides
    parser.add_argument("--ratio", type=int, default=None,
                        help=f"Override neg:pos ratio "
                             f"(default: {NN_SAMPLING['neg_ratio']})")
    parser.add_argument("--pos_cap", type=int, default=None,
                        help=f"Override num_train_pos_cap "
                             f"(default: {NN_SAMPLING['num_train_pos_cap']})")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Override seed (default: {NN_SAMPLING['seed']})")

    # Model architecture
    parser.add_argument("--hidden", nargs="+", type=int, default=None,
                        help=f"Hidden layer sizes "
                             f"(default: {NN_PARAMS['hidden_sizes']})")
    parser.add_argument("--dropout", type=float, default=None,
                        help=f"Dropout rate (default: {NN_PARAMS['dropout']})")

    # Training
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Max epochs (default: {NN_PARAMS['epochs']})")
    parser.add_argument("--batch_size", type=int, default=None,
                        help=f"Batch size (default: {NN_PARAMS['batch_size']})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {NN_PARAMS['lr']})")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help=f"Weight decay (default: {NN_PARAMS['weight_decay']})")
    parser.add_argument("--patience", type=int, default=None,
                        help=f"Early stopping patience "
                             f"(default: {NN_PARAMS['patience']})")
    parser.add_argument("--pos_weight", type=float, default=None,
                        help="Positive class weight for BCE loss "
                             "(default: auto from ratio)")

    # Output
    parser.add_argument("--tag", type=str, default=None,
                        help=f"Output tag (default: {NN_TAG})")
    args = parser.parse_args()

    # ── Resolve parameters ────────────────────────────────────────────────
    resolution  = args.resolution or RESOLUTION
    spatial_res = args.spatial or SPATIAL_RES
    train_years = args.train_years or TRAIN_YEARS
    val_years   = args.val_years or VAL_YEARS
    test_years  = args.test_years or TEST_YEARS
    sds_path    = resolve_path(args.sds_path or SDS_PATH)
    tag         = args.tag or NN_TAG

    # Sampling config
    sampling_cfg = NN_SAMPLING
    overrides = {}
    if args.ratio is not None:
        overrides["neg_ratio"] = args.ratio
    if args.pos_cap is not None:
        overrides["num_train_pos_cap"] = args.pos_cap
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        sampling_cfg = {**sampling_cfg, **overrides}

    # NN params
    hidden_sizes = tuple(args.hidden) if args.hidden else NN_PARAMS["hidden_sizes"]
    dropout      = args.dropout if args.dropout is not None else NN_PARAMS["dropout"]
    epochs       = args.epochs if args.epochs is not None else NN_PARAMS["epochs"]
    batch_size   = args.batch_size if args.batch_size is not None else NN_PARAMS["batch_size"]
    lr           = args.lr if args.lr is not None else NN_PARAMS["lr"]
    weight_decay = args.weight_decay if args.weight_decay is not None else NN_PARAMS["weight_decay"]
    patience     = args.patience if args.patience is not None else NN_PARAMS["patience"]
    pos_weight_override = args.pos_weight

    seed = sampling_cfg['seed']

    sds_file = Path(sds_path)
    if not sds_file.exists():
        sys.exit(f"ERROR: labels not found: {sds_file} (run the backtracking first)")

    # ── Output directory ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = MODELS_PATH / f"{tag}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────
    log_file = out_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger("train_nn")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    header("Neural network", {
        "Labels": project_relative(sds_file),
        "Resolution": f"{resolution}, {spatial_res}°",
        "Train / val / test": f"{train_years} / {val_years} / {test_years}",
        "Sampling": f"1:{sampling_cfg['neg_ratio']}, pos cap {sampling_cfg['num_train_pos_cap']}, "
                    f"seed {seed}",
        "Network": f"hidden {list(hidden_sizes)}, dropout {dropout}",
        "Training": f"{epochs} epochs, batch {batch_size}, lr {lr}, weight decay {weight_decay}, "
                    f"patience {patience}",
        "Device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "Output": project_relative(out_dir),
    }, out=logger.info)

    # ── Load features ─────────────────────────────────────────────────────
    cache = FeatureCache(
        resolution=resolution,
        spatial_res=spatial_res,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        logger=logger,
    )

    # ── Load SDS + collect indices ────────────────────────────────────────
    ds_sds, sds_var = load_sds_source(sds_file, logger)
    land_indices = compute_land_indices(
        ds_sds["latitude"].values, ds_sds["longitude"].values)

    section("Samples", out=logger.info)
    train_pos, train_neg = collect_indices(
        cache, cache.train_years, ds_sds, sds_var, land_indices, logger, label="Train")

    val_pos, val_neg = collect_indices(
        cache, cache.val_years, ds_sds, sds_var, land_indices, logger, label="Val")

    test_pos, test_neg = collect_indices(
        cache, cache.test_years, ds_sds, sds_var, land_indices, logger, label="Test")

    ds_sds.close()
    del ds_sds
    gc.collect()

    # ── Build datasets ────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    X_train, y_train, n_pos_train = build_train_set(
        cache, train_pos, train_neg, sampling_cfg, logger)

    X_val, y_val = build_eval_set(
        cache, val_pos, val_neg, sampling_cfg, rng, logger, "Val")

    X_test, y_test = build_eval_set(
        cache, test_pos, test_neg, sampling_cfg, rng, logger, "Test")

    del train_pos, train_neg, val_pos, val_neg, test_pos, test_neg
    gc.collect()

    n_features = X_train.shape[1]
    # ── Standardize features ──────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ── Tensors + DataLoader ──────────────────────────────────────────────
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val.copy()).float()
    X_test_t = torch.from_numpy(X_test.copy()).float()

    del X_train
    gc.collect()

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SDSNet(n_features, hidden_sizes=hidden_sizes,
                   dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # ── Loss with class weighting ─────────────────────────────────────────
    if pos_weight_override is not None:
        pw = pos_weight_override
    else:
        n_neg_train = int((y_train == 0).sum())
        n_pos_t = int((y_train == 1).sum())
        pw = n_neg_train / max(n_pos_t, 1)
    pw_tensor = torch.tensor([pw], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=NN_PARAMS["lr_decay_factor"],
        patience=NN_PARAMS["lr_decay_patience"])

    # ── Training loop ─────────────────────────────────────────────────────
    section(f"Train ({n_params:,} parameters, pos_weight {pw:.2f})", out=logger.info)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    best_model_path = out_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion,
                                 optimizer, device)

        # Validate
        proba_val = predict_proba_nn(model, X_val_t, device)
        thr = find_best_threshold(y_val, proba_val)
        val_m = compute_metrics(y_val, proba_val, thr)
        val_f1 = val_m["f1"]

        scheduler.step(val_f1)
        t_epoch = time.time() - t0

        logger.info(f"  Epoch {epoch:3d}/{epochs}  "
                    f"loss={train_loss:.4f}  "
                    f"val_F1={val_f1:.4f}  P={val_m['precision']:.4f}  "
                    f"R={val_m['recall']:.4f}  AUC={val_m['auc']:.4f}  "
                    f"thr={thr:.4f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.1e}  "
                    f"({t_epoch:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_f1": round(val_f1, 4),
            "val_precision": round(val_m["precision"], 4),
            "val_recall": round(val_m["recall"], 4),
            "val_auc": round(val_m["auc"], 4),
            "threshold": round(thr, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(t_epoch, 1),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "feature_vars": cache.feature_vars,
                "n_features": n_features,
                "hidden_sizes": list(hidden_sizes),
                "dropout": dropout,
                "threshold": thr,
                "epoch": epoch,
                "val_metrics": val_m,
            }, best_model_path)
        else:
            patience_counter += 1

        if patience > 0 and patience_counter >= patience:
            logger.info(f"  Early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ── Load best model and evaluate with Platt calibration ─────────────
    section(f"Calibrate and evaluate (best epoch {best_epoch})", out=logger.info)

    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Platt calibration on validation set (same as RF pipeline)
    from sklearn.model_selection import train_test_split as tts

    proba_val_raw = predict_proba_nn(model, X_val_t, device)

    unique_val = np.unique(y_val)
    if len(unique_val) >= 2 and int(y_val.sum()) >= 4:
        # Split val into calibration + evaluation halves
        val_idx = np.arange(len(y_val))
        cal_idx, eval_idx = tts(val_idx, test_size=VAL_CALIB_FRAC, stratify=y_val,
                                random_state=VAL_SPLIT_SEED)

        proba_cal = proba_val_raw[cal_idx]
        y_cal = y_val[cal_idx]
        proba_eval_raw = proba_val_raw[eval_idx]
        y_eval = y_val[eval_idx]

        # Fit Platt scaling
        platt_lr = platt_fit(y_cal, proba_cal)
        proba_eval_cal = platt_apply(platt_lr, proba_eval_raw)

        # Find optimal threshold on calibrated probabilities
        thr_cal = find_best_threshold(y_eval, proba_eval_cal)
        val_m_cal = compute_metrics(y_eval, proba_eval_cal, thr_cal)

        # uncalibrated, for comparison
        thr_uncal = find_best_threshold(y_eval, proba_eval_raw)
        val_m_uncal = compute_metrics(y_eval, proba_eval_raw, thr_uncal)

        logger.info(f"  Val:  F1={val_m_cal['f1']:.4f}  P={val_m_cal['precision']:.4f}  "
                    f"R={val_m_cal['recall']:.4f}  AUC={val_m_cal['auc']:.4f}  thr={thr_cal:.4f} "
                    f"(uncalibrated F1={val_m_uncal['f1']:.4f})")

        best_thr = thr_cal
        val_m = val_m_cal
    else:
        logger.warning("  Val set has too few positives for Platt calibration")
        platt_lr = None
        best_thr = find_best_threshold(y_val, proba_val_raw)
        val_m = compute_metrics(y_val, proba_val_raw, best_thr)
        logger.info(f"  Val:  F1={val_m['f1']:.4f}  P={val_m['precision']:.4f}  "
                    f"R={val_m['recall']:.4f}  AUC={val_m['auc']:.4f}")

    # Test with Platt calibration
    proba_test_raw = predict_proba_nn(model, X_test_t, device)
    if platt_lr is not None:
        proba_test = platt_apply(platt_lr, proba_test_raw)
    else:
        proba_test = proba_test_raw
    test_m = compute_metrics(y_test, proba_test, best_thr)
    logger.info(f"  Test: F1={test_m['f1']:.4f}  P={test_m['precision']:.4f}  "
                f"R={test_m['recall']:.4f}  AUC={test_m['auc']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    # Training history
    hist_path = out_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)

    # Platt scaler
    if platt_lr is not None:
        platt_path = out_dir / "platt_scaler.joblib"
        import joblib
        joblib.dump(platt_lr, platt_path)

    # Full results JSON
    results = {
        "tag": tag,
        "timestamp": timestamp,
        "device": str(device),
        "n_features": n_features,
        "n_params": n_params,
        "architecture": {
            "hidden_sizes": list(hidden_sizes),
            "dropout": dropout,
        },
        "training": {
            "epochs_trained": epoch,
            "best_epoch": best_epoch,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "pos_weight": pw,
            "patience": patience,
        },
        "threshold": best_thr,
        "platt_calibrated": platt_lr is not None,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "n_train": len(y_train),
        "n_pos_train": int(y_train.sum()),
        "feature_vars": cache.feature_vars,
        "sampling_config": {
            "neg_ratio": sampling_cfg['neg_ratio'],
            "num_train_total_cap": sampling_cfg['num_train_total_cap'],
            "num_train_pos_cap": sampling_cfg['num_train_pos_cap'],
            "val_test_total_cap": sampling_cfg['val_test_total_cap'],
            "seed": sampling_cfg['seed'],
        },
        "data_config": {
            "resolution": resolution,
            "spatial_res": spatial_res,
            "train_years": train_years,
            "val_years": val_years,
            "test_years": test_years,
            "sds_path": project_relative(sds_path),
        },
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    done(out_dir, out=logger.info)


if __name__ == "__main__":
    main()