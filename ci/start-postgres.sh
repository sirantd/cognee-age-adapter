#!/usr/bin/env bash
# Start the production PostgreSQL image as production runs it, with the AGE
# files mounted read-only, then make the graph database and its role.
#
# Usage: ci/start-postgres.sh <age-dir>
# Environment: AGE_CONTAINER (default age), AGE_PORT (default 5432),
#              AGE_USER / AGE_PASSWORD / AGE_DATABASE (default cognee_graph),
#              DOCKER_PLATFORM (for example linux/amd64 on an arm64 laptop).
set -euo pipefail

age_dir=$(cd "${1:?usage: start-postgres.sh <age-dir>}" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pins.env
source "$here/pins.env"

name=${AGE_CONTAINER:-age}
port=${AGE_PORT:-5432}
role=${AGE_USER:-cognee_graph}
password=${AGE_PASSWORD:-cognee_graph}
database=${AGE_DATABASE:-cognee_graph}
platform=()
if [ -n "${DOCKER_PLATFORM:-}" ]; then
  platform=(--platform "$DOCKER_PLATFORM")
fi

# The user, PGDATA, PGPORT, shm size and command of the TrueNAS app, read from
# the running container (ix-postgres-postgres-1) on 2026-11-08.
docker run -d --name "$name" "${platform[@]}" \
  --user 999:999 --shm-size 512m \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=home \
  -e PGDATA=/var/lib/postgresql/18/docker -e PGPORT=5432 \
  -v "$age_dir:/opt/age:ro" \
  -p "$port:5432" \
  "$POSTGRES_IMAGE" -c max_connections=100

# TCP, not the socket: the entrypoint's temporary init server listens only on
# the socket, and a setup against it would race the entrypoint.
for _ in $(seq 1 60); do
  if docker exec "$name" pg_isready -q -h 127.0.0.1 -p 5432; then
    break
  fi
  sleep 1
done
docker exec "$name" pg_isready -h 127.0.0.1 -p 5432

psql_super=(docker exec -i "$name" psql -h 127.0.0.1 -U postgres -v ON_ERROR_STOP=1)
"${psql_super[@]}" -d home < "$here/init-cluster.sql"
"${psql_super[@]}" -d home -v role="$role" -v password="$password" -v database="$database" \
  < "$here/init-graph-db.sql"
"${psql_super[@]}" -d "$database" -tAc \
  "SELECT 'age ' || extversion FROM pg_extension WHERE extname = 'age'"
