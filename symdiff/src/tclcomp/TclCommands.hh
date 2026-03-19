/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef TCLCOMMANDS_HH
#define TCLCOMMANDS_HH
#include <tcl.h>
namespace dsTcl {
typedef int (*tclcmd) (ClientData clientData, Tcl_Interp *interp,
      int objc, Tcl_Obj *CONST objv[]);

struct Commands
{
    const char *name;
    tclcmd     command;
};

int Commands_Init(Tcl_Interp *interp);

class dsClientData {
  private:
    dsClientData(const char *name, tclcmd impl) : command_name_(name), command_impl_(impl)
    {
    }

  public:
    static void *CreateData(const char *name, tclcmd impl)
    {

      void *handle;
      handle = new dsClientData(name, impl);
      return handle;
    } 

    static void DeleteData(void *handle)
    {
      delete reinterpret_cast<dsClientData *>(handle);
    }

    static const char *GetCommandName(void *handle)
    {
      return reinterpret_cast<dsClientData *>(handle)->command_name_;
    }

    tclcmd GetCommand() {
      return command_impl_;
    }

  private:
    dsClientData();
    dsClientData(const dsClientData &);
    dsClientData &operator=(const dsClientData &);


    ~dsClientData()
    {
    }

    const char *command_name_;
    tclcmd      command_impl_;
};
}
#endif
