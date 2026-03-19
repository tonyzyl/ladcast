/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "mcModelCompiler.hh"
#include "Functions.hh"

#include <map>
#include <list>
#include <string>
#include <utility>

#include "mcModel.hh"
#include "Context.hh"

Eqo::EqObjPtr ClearModel(const std::string &s)
{
  Context &context = Context::GetInstance();
  context.ClearModel(s);
  return Eqo::con(0.0);
}

Eqo::EqObjPtr DeclareModel(const std::string &s)
{
  Context &context = Context::GetInstance();
  context.DeclareModel(s);
  return Eqo::mod(s);
}

Eqo::EqObjPtr DefineModel(const std::string &s, Eqo::EqObjPtr e)
{
  Context &context = Context::GetInstance();
  context.DefineModel(s, e);
  return Eqo::mod(s);
}

/// Does not error out.  For this use top level command
Eqo::EqObjPtr GetModelValue(const std::string &s)
{
  Eqo::EqObjPtr ret = Eqo::var("UNDEFINED");

  Context &context = Context::GetInstance();
  if (context.IsInModelList(s))
  {
    Eqo::EqObjPtr x = context.FindInModelList(s);
    if (x)
    {
      ret = x;
    }
  }

  return ret;
}

