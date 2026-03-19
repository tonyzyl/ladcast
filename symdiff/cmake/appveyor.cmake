# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
#SET(PYTHON_BIN="/usr/bin/python")

ADD_DEFINITIONS(-DTCL_THREADS -DUSE_TCL_STUBS)
ADD_DEFINITIONS(-DSTATIC_BUILD -D_USE_MATH_DEFINES)
SET (CMAKE_CXX_WARNING_LEVEL 4)


# warning C4251 for dll exports
# warning C4005 concerning macro redefines in stdint.h
# warning C4996: 'isatty': The POSIX name for this item is deprecated
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:strict /wd4251 /wd4005 /wd4996")
SET (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} /fp:strict /wd4251 /wd4005 /wd4996")
#64 bit build
#SET(TCL_INCLUDE "c:/Tcl/include")
#SET(TCL_ARCHIVE "c:/Tcl/lib/tclstub85.lib")

SET(TCL_INCLUDE "${ANACONDA_PATH}/Library/include")
SET(TCL_ARCHIVE "${ANACONDA_PATH}/Library/lib/tclstub86.lib")
SET(TCL_BIN     "${ANACONDA_PATH}/Library/bin/tclsh.exe")
SET (PYTHON3_ARCHIVE ${ANACONDA_PATH}/libs/python3.lib)
SET (PYTHON3_INCLUDE ${ANACONDA_PATH}/include)

SET (FLEX  "C:/msys64/usr/bin/flex.exe")
SET (BISON "C:/msys64/usr/bin/bison.exe")

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:strict /EHsc")
SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SAFESEH:NO")
SET (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /SAFESEH:NO /NODEFAULTLIB:python3")
SET (CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /SAFESEH:NO")

