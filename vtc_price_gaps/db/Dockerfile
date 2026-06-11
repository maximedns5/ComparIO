FROM postgis/postgis:16-3.4

RUN apt-get update && apt-get install -y \
    cmake build-essential postgresql-server-dev-16 git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v4.1.3 https://github.com/zachasme/h3-pg.git /tmp/h3-pg \
    && cd /tmp/h3-pg \
    && cmake -B build . \
    && cmake --build build \
    && cmake --install build \
    && rm -rf /tmp/h3-pg