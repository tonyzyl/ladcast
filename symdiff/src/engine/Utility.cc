/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "Utility.hh"
#include "EquationObject.hh"

#include <algorithm>

namespace Eqo {
namespace {
//// order by CONST_OBJ, the VARIABLE_OBJ, then everything else see enum
struct EqVecCompare
{
  bool operator()(EqObjPtr x, EqObjPtr y)
  {
    const EqObjType xtype = x->getType();
    const EqObjType ytype = y->getType();

    bool ret = false;

    if (xtype < ytype)
    {
      ret = true;
    }
    else if (xtype == ytype)
    {
      const std::string &xstr = x->stringValue();
      const std::string &ystr = y->stringValue();

      ret = (xstr < ystr);
    }

    return ret;
  }
};

}
/// Sorts in order of constants, variables, then other objects
void SortEqVector(std::vector<EqObjPtr> &v1)
{
  std::sort(v1.begin(), v1.end(), EqVecCompare());
}

}

