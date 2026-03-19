/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include <tcl.h>
#include "SymdiffImport.hh"
#include "TclCommands.hh"

#if 0
int main(int argc, char * argv[])
{
    Tcl_Main(argc, argv, Tcl_AppInit);

}
#endif
extern "C" {
int DLL_PUBLIC
Symdifftcl_Init(Tcl_Interp *interp)
{
  if (Tcl_InitStubs(interp, TCL_VERSION, 0) == nullptr) {
    return TCL_ERROR;
  }
  if (Tcl_PkgProvide(interp, "symdifftcl", "1.0") == TCL_ERROR) {
    return TCL_ERROR;
  }
  dsTcl::Commands_Init(interp);
  return TCL_OK;
}

DLL_PUBLIC Tcl_PackageInitProc *Symdifftcl_SafeInit = Symdifftcl_Init;
DLL_PUBLIC Tcl_PackageUnloadProc *Symdifftcl_SafeUnload = nullptr;
DLL_PUBLIC Tcl_PackageUnloadProc *Symdifftcl_Unload = nullptr;


}



