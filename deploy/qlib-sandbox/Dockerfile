ARG QLIB_SANDBOX_BASE_IMAGE
FROM ${QLIB_SANDBOX_BASE_IMAGE}

ENV PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    QLIB_REPO=/opt/qlib

# Qlib itself is already built from the governed commit in Dockerfile.worker.
# Only host Docker builds this image; the isolated DinD daemon pulls its digest.
RUN test "${QLIB_COMMIT}" = "d5379c520f66a39953bad76234a7019a72796fd0" \
    && test -d /opt/qlib/qlib \
    && python -c "import catboost, numpy as np, qlib, tables, torch, xgboost; assert catboost and qlib and tables and xgboost; assert torch.from_numpy(np.arange(3)).numpy().tolist() == [0, 1, 2]" \
    && mkdir -p /workspace \
    && ln -s /opt/qlib /workspace/qlib \
    && rm -rf /root/.cache

WORKDIR /workspace
