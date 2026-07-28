ARG FACTOR_SANDBOX_BASE_IMAGE=quantlab-worker-runtime:v2
FROM ${FACTOR_SANDBOX_BASE_IMAGE}

ENV PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1

RUN rm -rf /app /opt/qlib /root/.cache \
    && mkdir /work \
    && chmod 1777 /work

USER 65534:65534
WORKDIR /work
CMD ["python", "-I", "factor.py"]
