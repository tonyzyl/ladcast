/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef PARSEEXPR_HH
#define PARSEEXPR_HH
#include "SymdiffImport.hh"
#include <memory>
#include <list>
#include <string>

namespace Eqo {
    class EquationObject;
    typedef std::shared_ptr<EquationObject> EqObjPtr;
};

namespace EvalExpr {
typedef std::list<std::string> error_t;

/// We are using the parser for a new purpose
void clearVariableMap();

Eqo::EqObjPtr DLL_PROTECTED evaluateExpression(const std::string &, error_t &);
}

#endif

