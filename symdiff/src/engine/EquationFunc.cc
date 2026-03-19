/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/
#include "EquationFunc.hh"
#include "EquationObject.hh"
#include "mcModelCompiler.hh"
#include "parseexpr.hh"
#include "Context.hh"

#include <sstream>
namespace sdHelp {


ret_pair SymdiffEval(const std::string &expr)
{
  bool ret = false;
  ret_data data;

  EvalExpr::error_t terrors;

  Eqo::EqObjPtr testeq = EvalExpr::evaluateExpression(expr, terrors);

  if (!terrors.empty())
  {
    std::ostringstream os;
    os << "While evaluating expression \"" << expr << "\"\n";
    for (EvalExpr::error_t::iterator it = terrors.begin(); it != terrors.end(); ++it)
    {
      os << *it << "\n";
    }
    ret = false;
    data.string_ = os.str();
  }
  else
  {
    data.string_ = testeq->stringValue();
    data.eqptr_  = testeq;
    ret = true;
  }

  return std::make_pair(ret, data);
}

}

