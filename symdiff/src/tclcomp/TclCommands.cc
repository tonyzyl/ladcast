/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "TclCommands.hh"

#include "TclSymdiffCommands.hh"

#include <tcl.h>
#include <iostream>

namespace dsTcl {

int 
CmdDispatch(ClientData clientData, Tcl_Interp *interp,
          int objc, Tcl_Obj *CONST objv[])
{
  int myerror = TCL_OK;


  try
  {
    tclcmd cmd = reinterpret_cast<dsClientData *>(clientData)->GetCommand();
    myerror = cmd(clientData, interp, objc, objv);
  }
#if 0
  catch (const dsException &x)
  {
    myerror = TCL_ERROR;
    Tcl_SetResult(interp, const_cast<char *>(x.what()), TCL_VOLATILE);
  }
#endif
  catch (std::bad_alloc &)
  {
    myerror = TCL_ERROR;
    Tcl_SetResult(interp, const_cast<char *>("OUT OF MEMORY"), TCL_VOLATILE);
  }

  return myerror;
}

void AddCommands(Tcl_Interp *interp, Commands *clist)
{
  for ( ; (clist->name) != nullptr; ++clist)
  {
    void *cdata = dsClientData::CreateData(clist->name, clist->command);

    Tcl_CreateObjCommand(interp, clist->name, CmdDispatch,
            (ClientData) cdata, (Tcl_CmdDeleteProc *) dsClientData::DeleteData);
  }
}

int 
Commands_Init(Tcl_Interp *interp) {
    AddCommands(interp, TclSymdiffCommands);
    return TCL_OK;
}
}

