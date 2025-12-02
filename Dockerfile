FROM ghcr.io/tahoebio/tahoe-x1:1.0.0

ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN uv tool install arc-state

COPY cell-eval /tmp/cell-eval
RUN uv tool install /tmp/cell-eval && rm -rf /tmp/cell-eval

COPY tahoe-x1 /tmp/tahoe-x1
RUN pip install /tmp/tahoe-x1 && rm -rf /tmp/tahoe-x1

WORKDIR /workspace
CMD ["/bin/bash"]
