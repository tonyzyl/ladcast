#!/bin/bash
set -e
set -u
export DEVSIM_ARCH=$(uname -m)

if [ "$DEVSIM_ARCH" = "arm64" ]; then
DEVSIM_ARCH=aarch64
fi

docker run -it -d --name manylinux_2_28 quay.io/pypa/manylinux_2_28_${DEVSIM_ARCH} &&
(cd .. && tar czf symdiff.tgz symdiff && docker cp symdiff.tgz manylinux_2_28:/root/) &&
docker exec manylinux_2_28 bash -c "git config --global --add safe.directory /root/symdiff"
docker exec manylinux_2_28 bash -c "cd /root && tar xzf symdiff.tgz";
docker exec manylinux_2_28 bash -c "cd /root/symdiff && bash scripts/build_manylinux_2_28.sh";
(for i in $(docker exec manylinux_2_28 bash -c "ls /root/symdiff/*.whl"); do docker cp manylinux_2_28:${i} .; done)
docker stop manylinux_2_28

