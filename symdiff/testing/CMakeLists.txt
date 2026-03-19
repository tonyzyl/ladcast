INCLUDE_DIRECTORIES (
)

# pass linker so order important

IF (WIN32)
SET (GOLDENDIR   ${PROJECT_SOURCE_DIR}/goldenresults/unix)
SET (RUNDIR      ${PROJECT_SOURCE_DIR}/testing)
SET (TCLCOMP     ${PROJECT_BIN_DIR}/symdiff_tcl.bat)
SET (PYTHON3COMP  ${PROJECT_BIN_DIR}/symdiff_py3.bat)
SET (RUNDIFFTEST ${PYTHON3COMP} ${PROJECT_SOURCE_DIR}/testing/rundifftest.py)
ELSE (WIN32)
SET (GOLDENDIR   ${PROJECT_SOURCE_DIR}/goldenresults/unix)
SET (RUNDIR      ${PROJECT_SOURCE_DIR}/testing)
SET (TCLCOMP     ${PROJECT_BIN_DIR}/symdiff_tcl)
SET (PYTHON3COMP  ${PROJECT_BIN_DIR}/symdiff_py3)
SET (RUNDIFFTEST ${PYTHON3COMP} ${PROJECT_SOURCE_DIR}/testing/rundifftest.py)
ENDIF (WIN32)

ENABLE_TESTING()

IF(TCLMAIN)
SET (TCLTESTS parsetest_tcl symdiff1 symdiff2 symdiff3 symdiff4 arrhenius_tcl units table)
FOREACH(I ${TCLTESTS})
ADD_TEST("${I}" ${RUNDIFFTEST} --testexe ${TCLCOMP} --goldendir ${GOLDENDIR} --output ${I}.out  --working ${RUNDIR} --args ${RUNDIR}/${I}.tcl)
ENDFOREACH(I)
ENDIF(TCLMAIN)

SET (EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)
SET (PYTHONTESTS parsetest deftest arrhenius modellist1 ordered remove1 subexpression1 undefined1 utf8 models1 table)
IF (PYTHON3)
FOREACH(I ${PYTHONTESTS})
ADD_TEST("${I}_py3" ${RUNDIFFTEST} --testexe ${PYTHON3COMP} --goldendir ${GOLDENDIR} --output ${I}_py3.out --working ${EXAMPLES_DIR} --args ${EXAMPLES_DIR}/${I}.py)
ENDFOREACH(I)
ENDIF (PYTHON3)

