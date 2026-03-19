/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "parseexpr.hh"
#include "mcTest.hh"
#include "Functions.hh"
#include <sstream>
namespace EvalExpr {
static error_t *lerrors;

void clearVariableMap() {
    Eqo::variableMap.clear();
}

/// Returns Equation Object containing the result of calculation, as well as any errors
/// error list should be empty on entry
Eqo::EqObjPtr evaluateExpression(const std::string &x, error_t &errors)
{
    std::string y;
    y.reserve(x.size() + 1);
    y += x;
    y += ";";
    lerrors = &errors;
    YY_BUFFER_STATE yyb = nullptr;
//    clearVariableMap();
    yyb = mc_scan_string(y.c_str());
    int retval = 0;
    retval = mcparse();
    if (yyb != nullptr)
    {
        mc_delete_buffer(yyb);
    }
    Eqo::EqObjPtr result;
    if (retval == 0)
    {
        result = Eqo::variableMap["$_"];
    }
    else
    {
        result = Eqo::con(0.0);
    }
    lerrors=nullptr;
    return result;
}
}

// Not in any namespace
int mcerror(const char *msg) {
    EvalExpr::lerrors->push_back(std::string(msg));
    return 0;
}

