/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "TclSymdiffCommands.hh"

#include "EquationFunc.hh"
#include "Context.hh"
#include "SubExpr.hh"
#include "ProcessModelOrder.hh"
#include "ProcessOrderedTable.hh"
#include "EngineAPI.hh"

#include <sstream>
#include <tcl.h>

namespace dsTcl {

static Tcl_Obj *returnList(Tcl_Interp *interp, const std::vector<size_t> &index)
{
  Tcl_Obj *obj = Tcl_NewObj();
  for (size_t i = 0; i < index.size(); ++i)
  {
    Tcl_ListObjAppendElement(interp, obj, Tcl_NewIntObj(static_cast<int>(index[i])));
  }
  return obj;
}

/**
Takes one argument, which is the command to evaluate
*/
static int 
symdiffCmd(ClientData clientData, Tcl_Interp *interp,
          int objc, Tcl_Obj *CONST objv[])
{
  int error = TCL_OK;

  std::string errorString;
  sdHelp::ret_pair result;
  result.first = false;

  if (objc == 2)
  {
    const std::string &expr = Tcl_GetStringFromObj(objv[1], nullptr);
    result = sdHelp::SymdiffEval(expr);

    if (!result.first)
    {
      errorString += result.second.string_;
    }

  }
  else
  {
    errorString += "Command takes only 1 argument.";
  }


  if (!errorString.empty())
  {
    std::ostringstream os;
    os << "While calling symdiff interpreter\n";
    errorString = os.str() + errorString;
    error = TCL_ERROR;
    Tcl_SetResult(interp, const_cast<char *>(errorString.c_str()), TCL_VOLATILE);
  }
  else
  {
    Tcl_SetResult(interp, const_cast<char *>(result.second.string_.c_str()), TCL_VOLATILE);
  }

  return error;
}

static int 
symdiffTableCmd(ClientData clientData, Tcl_Interp *interp,
          int objc, Tcl_Obj *CONST objv[])
{
  int error = TCL_OK;

  std::string errorString;
  sdHelp::ret_pair result;
  result.first = false;

  if (objc == 2)
  {
    const std::string &expr = Tcl_GetStringFromObj(objv[1], nullptr);
    result = sdHelp::SymdiffEval(expr);

    if (!result.first)
    {
      errorString += result.second.string_;
    }

  }
  else
  {
    errorString += "Command takes only 1 argument.";
  }


  if (!errorString.empty())
  {
    std::ostringstream os;
    os << "While calling symdiff interpreter\n";
    errorString = os.str() + errorString;
    error = TCL_ERROR;
    Tcl_SetResult(interp, const_cast<char *>(errorString.c_str()), TCL_VOLATILE);
  }
  else
  {
      ProcessOrderedTable pot;
      pot.run(result.second.eqptr_);
      OrderedTable_t table = pot.GetOrderedTable();
      Tcl_Obj *obj = Tcl_NewObj();

      if (!table.empty())
      {
        for (size_t i = 0; i < table.size(); ++i)
        {
          const OrderedTableData &data = table[i];
          const std::string &name = EngineAPI::getName(data.ptr_);
          const std::string &type = EngineAPI::getType(data.ptr_);
          const std::string &value = data.value_;
          const std::vector<size_t> &indexes = data.indexes_;
          const std::vector<size_t> &references = data.references_;

          Tcl_Obj *rowobj = Tcl_NewObj();
          Tcl_Obj *subobj = nullptr;
          subobj = Tcl_NewStringObj(name.c_str(), -1);
          Tcl_ListObjAppendElement(interp, rowobj, subobj);
          subobj = Tcl_NewStringObj(type.c_str(), -1);
          Tcl_ListObjAppendElement(interp, rowobj, subobj);
          subobj = returnList(interp, indexes);
          Tcl_ListObjAppendElement(interp, rowobj, subobj);
          subobj = returnList(interp, references);
          Tcl_ListObjAppendElement(interp, rowobj, subobj);
          subobj = Tcl_NewStringObj(value.c_str(), -1);
          Tcl_ListObjAppendElement(interp, rowobj, subobj);
          Tcl_ListObjAppendElement(interp, obj, rowobj);
        }
      }
      Tcl_SetObjResult(interp, obj);
  }

  return error;
}


static int
modelListCmd(ClientData clientData, Tcl_Interp *interp,
          int objc, Tcl_Obj *CONST objv[])
{
  int error = TCL_OK;

  std::string errorString;
  sdHelp::ret_pair result;
  result.first = false;

  if (objc == 1)
  {
    Tcl_Obj *obj = Tcl_NewObj();
    const ModelMap_t &model_list = Context::GetInstance().GetModelMap(); 
    for (ModelMap_t::const_iterator it = model_list.begin(); it != model_list.end(); ++it)
    {
      const std::string &model_name = (it->first);
      Tcl_Obj *subobj = Tcl_NewStringObj(model_name.c_str(), -1);
      Tcl_ListObjAppendElement(interp, obj, subobj);
    }
    Tcl_SetObjResult(interp, obj);
  }
  else
  {
    errorString += "Command takes no arguments.";
  }


  if (!errorString.empty())
  {
    std::ostringstream os;
    os << "While calling symdiff interpreter\n";
    errorString = os.str() + errorString;
    error = TCL_ERROR;
    Tcl_SetResult(interp, const_cast<char *>(errorString.c_str()), TCL_VOLATILE);
  }

  return error;
}

static int
subexpressionCmd(ClientData clientData, Tcl_Interp *interp,
          int objc, Tcl_Obj *CONST objv[])
{
  int error = TCL_OK;

  std::string errorString;
  sdHelp::ret_pair result;
  result.first = false;

  if (objc == 1)
  {
    SubExpr subexpr;
    ModelMap_t &model_list = Context::GetInstance().GetModelMap(); 
    subexpr.CreateSubexpressions(model_list);
    errorString = subexpr.GetErrorString();
#if  0
    if (errorString.empty())
    {
      subexpr.RemoveZeros(model_list);
    }
#endif
  }
  else
  {
    errorString += "Command takes no arguments.";
  }


  if (!errorString.empty())
  {
    std::ostringstream os;
    os << "While calling symdiff interpreter\n";
    errorString = os.str() + errorString;
    error = TCL_ERROR;
    Tcl_SetResult(interp, const_cast<char *>(errorString.c_str()), TCL_VOLATILE);
  }

  return error;
}

static int
removeZerosCmd(ClientData clientData, Tcl_Interp *interp,
          int objc, Tcl_Obj *CONST objv[])
{
  int error = TCL_OK;

  std::string errorString;
  sdHelp::ret_pair result;
  result.first = false;

  if (objc == 1)
  {
    SubExpr subexpr;
    ModelMap_t &model_list = Context::GetInstance().GetModelMap(); 
    subexpr.RemoveZeros(model_list);
  }
  else
  {
    errorString += "Command takes no arguments.";
  }


  if (!errorString.empty())
  {
    std::ostringstream os;
    os << "While calling symdiff interpreter\n";
    errorString = os.str() + errorString;
    error = TCL_ERROR;
    Tcl_SetResult(interp, const_cast<char *>(errorString.c_str()), TCL_VOLATILE);
  }

  return error;
}

static int
orderedListCmd(ClientData clientData, Tcl_Interp *interp,
          int objc, Tcl_Obj *CONST objv[])
{
  int error = TCL_OK;

  std::string errorString;

  if (objc > 1)
  {
    ModelNameVector_t model_vector_in(objc - 1);

    for (size_t i = 1; i < static_cast<size_t>(objc); ++i)
    {
      const std::string &model_name = Tcl_GetStringFromObj(objv[i], nullptr);
      model_vector_in[i-1] = model_name;
    }

    ProcessModelOrder process;
    const ModelMap_t &model_list = Context::GetInstance().GetModelMap(); 

    const ModelNameVector_t &model_vector_out = process.GetModelVector(model_list, model_vector_in);

    errorString = process.GetErrorString();

    if (errorString.empty())
    {
      Tcl_Obj *obj = Tcl_NewObj();
      for (ModelNameVector_t::const_iterator it = model_vector_out.begin(); it != model_vector_out.end(); ++it)
      {
        const std::string &model_name = *it;
        Tcl_Obj *subobj = Tcl_NewStringObj(model_name.c_str(), -1);
        Tcl_ListObjAppendElement(interp, obj, subobj);
      }
      Tcl_SetObjResult(interp, obj);
    }
  }
  else
  {
    errorString += "Command takes at least one argument.";
  }


  if (!errorString.empty())
  {
    std::ostringstream os;
    os << "While calling symdiff interpreter\n";
    errorString = os.str() + errorString;
    error = TCL_ERROR;
    Tcl_SetResult(interp, const_cast<char *>(errorString.c_str()), TCL_VOLATILE);
  }

  return error;
}

Commands TclSymdiffCommands[] = {
    {"symdiff",    symdiffCmd},
    {"symdiff_table",    symdiffTableCmd},
    {"model_list", modelListCmd},
    {"subexpression", subexpressionCmd},
    {"ordered_list", orderedListCmd},
    {"remove_zeros", removeZerosCmd},
    {nullptr, nullptr}
};

}


