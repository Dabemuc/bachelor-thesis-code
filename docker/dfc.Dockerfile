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
#        docker run --rm -it --gpus all -v "$PWD":/work -w /work thesis-dfc     # mit GPU
#    im Container einmalig:  pip install -e ".[compile]"
#
# GPU: ohne GPU setzt der DFC nur den DEFAULT auf Optimierungsstufe 0 (Equalization +
# Kalibrierung). Explizit gesetzte Stufen 1–4 laufen auch auf der CPU, nur langsam
# (Quelle: Quelltext DFC 3.33, mo_config.py / quantize.py; Details im Vault:
# „DFC 3.33.0 Stellschrauben“). Für die Messungen der Arbeit die Stufe bewusst wählen,
# in der Config festhalten und im Methodikteil nennen.
# Die GPU-Variante braucht auf dem Host den NVIDIA-Treiber + nvidia-container-toolkit
# und im Image passende CUDA/cuDNN-Bibliotheken → BASE_IMAGE dann auf ein
# nvidia/cuda-…-cudnn-…-ubuntu22.04-Image setzen (Version gegen die DFC-3.33-Doku prüfen).

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

# Schnelltest:  docker run --rm thesis-dfc hailo --version
