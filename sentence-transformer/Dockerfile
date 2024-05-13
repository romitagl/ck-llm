# syntax=docker/dockerfile:1

ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION} as base

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to Pipfile to avoid having to copy them into
# into this layer.
# --deploy — Make sure the packages are properly locked in Pipfile.lock, and abort if the lock file is out-of-date.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=Pipfile,target=Pipfile \
    --mount=type=bind,source=Pipfile.lock,target=Pipfile.lock \
    pip install pipenv

ENV PATH="/.venv/bin:$PATH"

RUN git clone https://github.com/romitagl/kgraph.wiki.git

COPY ./ .

# install dependencies
RUN pipenv install

# download model and store in the cache
RUN pipenv run python retrieval_based_qa.py

# Run the application.
CMD ["pipenv", "run", "python", "main.py"]