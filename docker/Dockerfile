FROM ubuntu:22.04

RUN apt-get update && apt-get upgrade -y \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  git \
  python3-pip \
  python3-dev \
  python-is-python3 \
  # open3d system dependencies: https://www.open3d.org/docs/release/docker.html
  libegl1 \
  libgl1 \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
ENV UNAME=dev

# Add normal sudo-user to container, passwordless
RUN addgroup --gid $GID $UNAME \
  && adduser --disabled-password --gecos '' --uid $UID --gid $GID $UNAME \
  && adduser $UNAME sudo \
  && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
  && sed -i 's/required/sufficient/' /etc/pam.d/chsh \
  && touch /home/$UNAME/.sudo_as_admin_successful

WORKDIR /home/${UNAME}/
ENV HOME=/home/${UNAME}
USER ${UNAME}
ENV PATH="${PATH}:${HOME}/.local/bin"

WORKDIR /home/${UNAME}/forest_inventory_pipeline
COPY --chown=${UID}:${GID} . /home/${UNAME}/forest_inventory_pipeline

RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -v ".[test]"

RUN pytest -v tests

CMD ["/bin/bash"]
