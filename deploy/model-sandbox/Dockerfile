ARG MODEL_SANDBOX_BASE_IMAGE
FROM ${MODEL_SANDBOX_BASE_IMAGE}

ENV PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    MODEL_SANDBOX_N_JOBS=2

RUN python -c "import lightgbm, numpy as np, qlib, torch; assert lightgbm and qlib; assert torch.from_numpy(np.arange(3)).numpy().tolist() == [0, 1, 2]" \
    && rm -rf /app /root/.cache \
    && mkdir /work \
    && chmod 1777 /work

USER 65534:65534
WORKDIR /work
CMD ["python", "-I", "runner.py"]
