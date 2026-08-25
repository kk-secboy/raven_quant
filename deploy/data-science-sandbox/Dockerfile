ARG DATA_SCIENCE_SANDBOX_BASE_IMAGE
FROM ${DATA_SCIENCE_SANDBOX_BASE_IMAGE}

ENV PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1

# This governed CPU image replaces RD-Agent's mutable Kaggle GPU base.
RUN python -c "import matplotlib, numpy, pandas, seaborn, sklearn, xgboost; assert matplotlib and numpy and pandas and seaborn and sklearn and xgboost" \
    && rm -rf /app /root/.cache \
    && mkdir /work \
    && chmod 1777 /work

USER 65534:65534
WORKDIR /work
