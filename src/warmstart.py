"""NN warm-start for unicycle MPC input sequence U."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def feature_dim(N: int) -> int:
    # z(4) + ref_xy(N*2) + centers(N*2) + half(N*2)
    return 4 + N * 2 + N * 2 + N * 2


def pack_features(z: np.ndarray, ref: np.ndarray, corridors: np.ndarray) -> np.ndarray:
    boxes = np.asarray(corridors, dtype=float).reshape(-1, 4)
    centers = 0.5 * np.column_stack([boxes[:, 0] + boxes[:, 1], boxes[:, 2] + boxes[:, 3]])
    half = 0.5 * np.column_stack([boxes[:, 1] - boxes[:, 0], boxes[:, 3] - boxes[:, 2]])
    return np.concatenate(
        [
            np.asarray(z, dtype=float).reshape(4),
            np.asarray(ref, dtype=float)[:, :2].ravel(),
            centers.ravel(),
            half.ravel(),
        ]
    )


class NumpyMLP:
    """Tiny 2-layer ReLU MLP (numpy) for ONNX-friendly warm-start."""

    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> None:
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        h = np.maximum(0.0, x @ self.w1.T + self.b1)
        y = h @ self.w2.T + self.b2
        return y[0] if single else y

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path: Path) -> "NumpyMLP":
        data = np.load(path)
        return cls(data["w1"], data["b1"], data["w2"], data["b2"])


def train_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    hidden: int = 64,
    lr: float = 1e-3,
    epochs: int = 80,
    batch: int = 64,
    seed: int = 0,
) -> tuple[NumpyMLP, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plain SGD on MSE; returns (mlp, x_mean, x_std, y_mean, y_std)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, d = X.shape
    out = Y.shape[1]
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-6
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0) + 1e-6
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    w1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(hidden, d))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(out, hidden))
    b2 = np.zeros(out)

    idx = np.arange(n)
    for ep in range(epochs):
        rng.shuffle(idx)
        # mild LR decay
        step = lr * (0.5 ** (ep // max(epochs // 3, 1)))
        for s in range(0, n, batch):
            bi = idx[s : s + batch]
            xb = Xn[bi]
            yb = Yn[bi]
            h_pre = xb @ w1.T + b1
            h = np.maximum(0.0, h_pre)
            pred = h @ w2.T + b2
            err = pred - yb
            gw2 = err.T @ h / len(bi)
            gb2 = err.mean(axis=0)
            dh = err @ w2
            dh[h_pre <= 0.0] = 0.0
            gw1 = dh.T @ xb / len(bi)
            gb1 = dh.mean(axis=0)
            w2 -= step * gw2
            b2 -= step * gb2
            w1 -= step * gw1
            b1 -= step * gb1

    return NumpyMLP(w1, b1, w2, b2), x_mean, x_std, y_mean, y_std


class WarmStartModel:
    """Predict U from features; clips to input bounds."""

    def __init__(
        self,
        mlp: NumpyMLP,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        a_max: float,
        w_max: float,
        N: int,
    ) -> None:
        self.mlp = mlp
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.a_max = a_max
        self.w_max = w_max
        self.N = N

    def predict_u(self, z: np.ndarray, ref: np.ndarray, corridors: np.ndarray) -> np.ndarray:
        feat = pack_features(z, ref, corridors)
        xn = (feat - self.x_mean) / self.x_std
        u = self.mlp.predict(xn) * self.y_std + self.y_mean
        u = u.reshape(self.N, 2)
        u[:, 0] = np.clip(u[:, 0], -self.a_max, self.a_max)
        u[:, 1] = np.clip(u[:, 1], -self.w_max, self.w_max)
        return u.ravel()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.mlp.w1,
            b1=self.mlp.b1,
            w2=self.mlp.w2,
            b2=self.mlp.b2,
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
            a_max=self.a_max,
            w_max=self.w_max,
            N=self.N,
        )

    @classmethod
    def load(cls, path: Path) -> "WarmStartModel":
        data = np.load(path)
        mlp = NumpyMLP(data["w1"], data["b1"], data["w2"], data["b2"])
        return cls(
            mlp,
            data["x_mean"],
            data["x_std"],
            data["y_mean"],
            data["y_std"],
            float(data["a_max"]),
            float(data["w_max"]),
            int(data["N"]),
        )

    def export_onnx(self, path: Path) -> None:
        """Write a minimal ONNX graph (Gemm-Relu-Gemm) without torch."""
        try:
            from onnx import TensorProto, helper, numpy_helper, save
        except ImportError as exc:
            raise ImportError("onnx package required for export (pip install onnx)") from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d_in = self.x_mean.shape[0]
        hidden = self.mlp.w1.shape[0]
        d_out = self.mlp.w2.shape[0]

        # Network expects already-normalized x; export raw weights
        nodes = [
            helper.make_node("Gemm", ["x", "W1", "B1"], ["h_pre"], alpha=1.0, beta=1.0, transB=1),
            helper.make_node("Relu", ["h_pre"], ["h"]),
            helper.make_node("Gemm", ["h", "W2", "B2"], ["y"], alpha=1.0, beta=1.0, transB=1),
        ]
        graph = helper.make_graph(
            nodes,
            "unicycle_warmstart",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, d_in])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, d_out])],
            [
                numpy_helper.from_array(self.mlp.w1.astype(np.float32), "W1"),
                numpy_helper.from_array(self.mlp.b1.astype(np.float32), "B1"),
                numpy_helper.from_array(self.mlp.w2.astype(np.float32), "W2"),
                numpy_helper.from_array(self.mlp.b2.astype(np.float32), "B2"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        save(model, str(path))


class OnnxWarmStart:
    """CPU ONNX Runtime inference for the warm-start net (normalized I/O)."""

    def __init__(
        self,
        onnx_path: Path,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        a_max: float,
        w_max: float,
        N: int,
    ) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)
        self.a_max = a_max
        self.w_max = w_max
        self.N = N
        self.input_name = self.sess.get_inputs()[0].name

    def predict_u(self, z: np.ndarray, ref: np.ndarray, corridors: np.ndarray) -> np.ndarray:
        feat = pack_features(z, ref, corridors).astype(np.float32)
        xn = (feat - self.x_mean) / self.x_std
        yn = self.sess.run(None, {self.input_name: xn[None, :]})[0][0]
        y = yn * self.y_std + self.y_mean
        u = y.reshape(self.N, 2)
        u[:, 0] = np.clip(u[:, 0], -self.a_max, self.a_max)
        u[:, 1] = np.clip(u[:, 1], -self.w_max, self.w_max)
        return u.ravel()
