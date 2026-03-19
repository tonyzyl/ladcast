#!/bin/bash
set -e
set -u
export DEVSIM_ARCH=$(uname -m)

yum install -y bison flex rsync

export CMAKE="cmake"
export CMAKE_CXX_FLAGS=""
export CXX="g++"

export PYTHON3_BIN="python3.9"
export PIP_BIN="/opt/python/cp39-cp39/bin/pip"
${PIP_BIN} install wheel auditwheel
export PYTHON3_INCLUDE=$(${PYTHON3_BIN} -c "from sysconfig import get_paths as gp; print(gp()['include'])")
export PYTHON3_ARCHIVE=""

# SYMDIFF build
(bash scripts/symdiff_centos.sh && cd linux_${DEVSIM_ARCH}_release && make -j2 && make test);
(rsync -avP --exclude __pycache__ lib/symdiff bdist_wheel/)
(rsync -avP --exclude __pycache__ LICENSE NOTICE README.md examples doc bdist_wheel)
(cd bdist_wheel && ${PIP_BIN} wheel .)
(cd bdist_wheel && auditwheel repair *.whl)
(mv bdist_wheel/wheelhouse/*.whl .)

