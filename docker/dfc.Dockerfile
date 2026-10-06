# Hailo Dataflow Compiler 3.33.1 (läuft mit HailoRT 4.23.0 auf dem Pi, geprüft 06.10.2026) – nur x86_64.
# Bis 06.10.2026: 3.33.0. 3.33.1 behebt laut Changelog einen Bug, der die GPU-Nutzung in der
# Optimierung verhinderte; bei Optimierungsstufe 0 sind die Ausgaben auf dem Chip bitgleich zu 3.33.0.
#
# 1) Wheel aus der Hailo Developer Zone laden (Login; Plattform "Hailo-8/8L" wählen!)
#    und hierher legen:
#        docker/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
#    ⚠️ Nur EIN Wheel in docker/ – COPY nimmt hailo_dataflow_compiler-*.whl, zwei Versionen brechen den Build.
#
# 2) Build (aus dem Repo-Root):
#        docker build -f docker/dfc.Dockerfile -t thesis-dfc docker/
#
# 3) Start – Repo wird gemountet, Ergebnisse landen direkt in artifacts/:
#        docker run --rm -it -v "$PWD":/work -w /work thesis-dfc                # nur CPU
#        (mit GPU: siehe unten, Podman nutzt --device nvidia.com/gpu=all)
#    im Container einmalig:  pip install -e ".[compile]"
#
# GPU: ohne GPU setzt der DFC nur den DEFAULT auf Optimierungsstufe 0 (Equalization +
# Kalibrierung). Explizit gesetzte Stufen 1–4 laufen auch auf der CPU, nur langsam
# (Quelle: Quelltext DFC 3.33, mo_config.py / quantize.py; Details im Vault:
# „DFC 3.33.1 Stellschrauben“). Für die Messungen der Arbeit die Stufe bewusst wählen,
# in der Config festhalten und im Methodikteil nennen.
# GPU-Variante (geprüft 06.10.2026, RTX 3070, Windows + Podman/WSL2):
#   Host: aktueller NVIDIA-Treiber (Windows) + nvidia-container-toolkit in der Podman-Machine (CDI-Spec
#   nach jedem Treiberupdate neu erzeugen: `podman machine ssh` → `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`).
#   CUDA/cuDNN kommen NICHT aus einem nvidia/cuda-Image, sondern als pip-Pakete passend zu TF 2.18
#   (`tensorflow[and-cuda]==2.18.0`). Grund: nvidia/cuda:12.5.1-cudnn-runtime hat cuDNN 9.2.1, TF 2.18
#   verlangt ≥ 9.3 („No DNN in stream executor“), und ohne nvcc fehlt libdevice für die XLA-JIT.
#   podman build -f docker/dfc.Dockerfile -t thesis-dfc:3.33.1-gpu --build-arg WITH_CUDA=1 docker/
#   podman run --rm -it --device nvidia.com/gpu=all -e CUDA_VISIBLE_DEVICES=0 -v "${PWD}:/work" -w /work thesis-dfc:3.33.1-gpu
#   ⚠️ CUDA_VISIBLE_DEVICES=0 ist nötig: Ohne die Variable wählt der DFC die GPU selbst und nimmt nur eine,
#      deren VRAM zu ≤ 5 % belegt ist (hailo_model_optimization/acceleras/utils/nvidia_smi_gpu_selector.py).
#      Die Desktop-GPU unter Windows hat schon ~1,6 GB belegt → „no suitable GPU found, falling back to CPU“.
#   Check im Log: „Using default optimization level of 2“ + „Loaded cuDNN version 90300“.

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip python3-tk \
        graphviz libgraphviz-dev build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/dfc
ENV PATH=/opt/dfc/bin:$PATH

ENV USER=root

COPY hailo_dataflow_compiler-*.whl /tmp/
RUN pip install --upgrade pip && pip install /tmp/hailo_dataflow_compiler-*.whl && rm /tmp/*.whl

# Optional GPU: CUDA/cuDNN/nvcc als pip-Pakete, exakt passend zur vom DFC gepinnten TF-Version.
ARG WITH_CUDA=0
RUN if [ "$WITH_CUDA" = "1" ]; then \
        TFV=$(python -c "import importlib.metadata as m; print(m.version('tensorflow'))") && \
        pip install "tensorflow[and-cuda]==${TFV}"; \
    fi

# Schnelltest:  docker run --rm thesis-dfc hailo --version
