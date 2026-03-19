@echo off
@SETLOCAL
SET BASEDIR=%cd%
SET CONDA_BIN=%2

c:\msys64\usr\bin\pacman -Su --noconfirm rsync zip bison flex git

call %CONDA_BIN% create -y -n symdiff_build python=3
if %errorlevel% neq 0 exit /b %errorlevel%

call %CONDA_BIN% install -y -n symdiff_build cmake tk
if %errorlevel% neq 0 exit /b %errorlevel%

call %CONDA_BIN% activate symdiff_build
if %errorlevel% neq 0 exit /b %errorlevel%

SET PATH="c:\msys64\usr\bin;%PATH%"
c:\msys64\usr\bin\bash %~dp0\setup_appveyor.sh %1
if %errorlevel% neq 0 exit /b %errorlevel%

