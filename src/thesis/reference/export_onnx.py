"""torchvision-Modell → ONNX, plus Hilfsbefehl zum Finden der Knotennamen.

    thesis-export-onnx resnet18                 # → artifacts/onnx/resnet18.onnx
    thesis-onnx-nodes artifacts/onnx/resnet18.onnx --grep layer4

Die Knotennamen hängen von torch-/Exporter-Version ab – deshalb nicht raten,
sondern mit ``thesis-onnx-nodes`` nachsehen und in die Config
(``end_node_names``) übernehmen. Gesucht sind: der Ausgang der letzten
Conv-Stufe (bei ResNet18 das letzte ReLU in layer4, vor dem AvgPool) und der
Logits-Knoten (Gemm).
"""

from __future__ import annotations

import argparse

from ..common.config import ARTIFACTS_DIR, IMAGE_SIZE

MODELS = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
}


def main() -> None:
    import torch  # noqa: PLC0415
    import torchvision.models as tvm  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="torchvision → ONNX")
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    fn_name, weights_name = MODELS[args.model]
    weights = getattr(tvm, weights_name).IMAGENET1K_V1
    model = getattr(tvm, fn_name)(weights=weights).eval()

    out = ARTIFACTS_DIR / "onnx" / f"{args.model}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    kwargs = dict(opset_version=args.opset, input_names=["input"], output_names=["logits"])
    try:  # neuere torch-Versionen: klassischen TorchScript-Exporter explizit wählen
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)
    except TypeError:  # ältere torch-Versionen kennen das Argument nicht
        torch.onnx.export(model, dummy, str(out), **kwargs)
    print(f"ONNX → {out}  (Gewichte: {weights})")


def list_nodes_main() -> None:
    import onnx  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="ONNX-Knoten auflisten")
    ap.add_argument("onnx_path")
    ap.add_argument("--grep", default="")
    args = ap.parse_args()

    graph = onnx.load(args.onnx_path).graph
    for node in graph.node:
        if args.grep in node.name:
            print(f"{node.op_type:<16} {node.name:<50} → {', '.join(node.output)}")


if __name__ == "__main__":
    main()
