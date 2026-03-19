#!/bin/bash
set -e
set -u

export CMAKE="cmake"
export CMAKE_CXX_FLAGS=""
export CC=clang;
export CXX=clang++;
export F77="";
export PYTHON3_BIN=python3
export PIP_BIN=pip

${PYTHON3_BIN} -mvenv venv
source venv/bin/activate

${PIP_BIN} install wheel
export PYTHON3_INCLUDE=$(${PYTHON3_BIN} -c "from sysconfig import get_paths as gp; print(gp()['include'])")
export PYTHON3_ARCHIVE=""

# SYMDIFF build
export ARCH_ARG="-DCMAKE_OSX_ARCHITECTURES=arm64"
#export PLAT_NAME="arm64"
#export ARCH_ARG="-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64"
#export PLAT_NAME="universal2"

(bash  scripts/symdiff_macos.sh && cd osx_release && make -j4)
(cd osx_release && make test)

(rsync -avP --exclude __pycache__ bdist_wheel/ dist)
(rsync -avP --exclude __pycache__ lib/symdiff dist/)
(rsync -avP --exclude __pycache__ LICENSE NOTICE README.md examples doc dist)

#FULL_PLAT_NAME=arm64
#echo PACKAGING $FULL_PLAT_NAME
#if [[ -n "$FULL_PLAT_NAME" ]]; then
#(cd dist &&  perl -p -i -e "s/^#plat-name.*/plat-name = ${FULL_PLAT_NAME}/" setup.cfg);
#fi
(cd dist && ${PIP_BIN} wheel .)
(mv dist/*.whl .)



