# Hailo Dataflow Compiler – nur x86_64.
#
# Der DFC ist nicht auf PyPI. Wheel aus der Hailo Developer Zone laden
# (Login nötig) und neben dieses Dockerfile legen:
#     docker/hailo_dataflow_compiler-<VERSION>-py3-none-linux_x86_64.whl
#
# ⚠️ Version passend zur HailoRT-Version auf dem Pi wählen
#    (Pi: `hailortcli fw-control identify` bzw. `dpkg -l | grep hailort`).
#    Für Hailo-8 die DFC-3.x-Linie – NICHT die 5.x-Linie (Hailo-10H).
#
# Build:  docker build -f docker/dfc.Dockerfile -t thesis-dfc docker/
# Run:    docker run --rm -it --gpus all -v "$PWD":/work -w /work thesis-dfc
#         (--gpus all nur mit nvidia-container-toolkit; ohne GPU läuft optimize
#          deutlich langsamer bzw. nur mit reduzierten Optimierungsstufen)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        graphviz libgraphviz-dev build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/dfc
ENV PATH=/opt/dfc/bin:$PATH

COPY hailo_dataflow_compiler-*.whl /tmp/
RUN pip install --upgrade pip && pip install /tmp/hailo_dataflow_compiler-*.whl && rm /tmp/*.whl

# Projektpaket wird beim Start aus dem gemounteten Repo installiert:
#   pip install -e ".[compile]"
