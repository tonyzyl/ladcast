/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef UTILITY_HH
#define UTILITY_HH
#include <memory>
#include <vector>
namespace Eqo
{
class EquationObject;
typedef std::shared_ptr<EquationObject> EqObjPtr;
}

namespace Eqo
{
void SortEqVector(std::vector<EqObjPtr> &v1);
}
#endif

