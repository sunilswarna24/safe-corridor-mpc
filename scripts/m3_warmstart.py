"""Phase 3: NN U-warm-start for unicycle MPC — collect, train, ONNX, cold/warm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import A_MAX, N, W_MAX  # N used by WarmStartModel
from src.scenarios import AWS_AISLE, two_crossers
from src.simulate_dynamic import run_dynamic_tracking
from src.warmstart import WarmStartModel, train_mlp


def _row(name: str, r: dict) -> str:
    return (
        f"{name:10s}  iters_mean={r['solve_iters_mean']:.1f}  "
        f"p95={r['solve_ms_p95']:.2f}ms  med={r['solve_ms_med']:.2f}ms  "
        f"hits={r['mover_hits']}  rms={r['err_rms']:.3f}  ok={r['ok_frac']:.3f}"
    )


def main() -> None:
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    movers = two_crossers()
    common = dict(
        **AWS_AISLE,
        movers=movers,
        plant="unicycle",
        reactive=True,
        use_corridors=True,
        use_supervisory=True,
        predict_movers=True,
    )

    print("Collecting unicycle MPC samples for warm-start...")
    Xs, Ys = [], []
    for seed in (3, 7, 11):
        collect = run_dynamic_tracking(**common, seed=seed)
        Xs.append(np.asarray(collect["feat_rows"], dtype=float))
        Ys.append(np.asarray(collect["u_seqs"], dtype=float))
    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    if len(X) < 20:
        raise RuntimeError(f"too few samples: {len(X)}")
    print(f"  samples: {len(X)}  feat_dim={X.shape[1]}  U_dim={Y.shape[1]}")

    print("Training numpy MLP...")
    mlp, x_mean, x_std, y_mean, y_std = train_mlp(
        X, Y, hidden=128, lr=1e-3, epochs=250, batch=64, seed=0
    )
    model = WarmStartModel(mlp, x_mean, x_std, y_mean, y_std, A_MAX, W_MAX, N)
    yhat = mlp.predict((X - x_mean) / x_std) * y_std + y_mean
    mse = float(np.mean((yhat - Y) ** 2))
    mae = float(np.mean(np.abs(yhat - Y)))
    print(f"  train MSE={mse:.4f}  MAE={mae:.4f}")

    npz_path = out / "warmstart_unicycle.npz"
    onnx_path = out / "warmstart_unicycle.onnx"
    model.save(npz_path)
    try:
        model.export_onnx(onnx_path)
        print(f"  wrote {onnx_path}")
    except ImportError as e:
        print(f"  ONNX export skipped: {e}")

    def warm_fn(z, ref, boxes):
        return model.predict_u(z, ref, boxes)

    print("Closed-loop cold vs warm @ 20 Hz")
    cold = run_dynamic_tracking(**common, seed=5, cold_start=True)
    print(_row("cold", cold))
    warm = run_dynamic_tracking(**common, seed=5, warm_start_fn=warm_fn)
    print(_row("warm", warm))

    if onnx_path.exists():
        try:
            from src.warmstart import OnnxWarmStart

            ort_model = OnnxWarmStart(
                onnx_path, x_mean, x_std, y_mean, y_std, A_MAX, W_MAX, N
            )

            def ort_fn(z, ref, boxes):
                return ort_model.predict_u(z, ref, boxes)

            warm_ort = run_dynamic_tracking(**common, seed=5, warm_start_fn=ort_fn)
            print(_row("onnx", warm_ort))
        except ImportError as e:
            print(f"  onnxruntime skipped: {e}")

    table = out / "m3_warmstart.txt"
    lines = [
        "closed-loop @ 20 Hz",
        "variant   iters_mean  p95_ms  med_ms  hits  rms",
        f"cold      {cold['solve_iters_mean']:.1f}       {cold['solve_ms_p95']:.2f}   {cold['solve_ms_med']:.2f}   {cold['mover_hits']}   {cold['err_rms']:.3f}",
        f"warm      {warm['solve_iters_mean']:.1f}       {warm['solve_ms_p95']:.2f}   {warm['solve_ms_med']:.2f}   {warm['mover_hits']}   {warm['err_rms']:.3f}",
    ]
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {npz_path}")
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
