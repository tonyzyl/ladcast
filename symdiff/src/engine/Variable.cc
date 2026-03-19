/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "Variable.hh"
#include "Functions.hh"
#include <sstream>
namespace Eqo {
EqObjPtr Variable::getScale()
{
   return con(1);
}

EqObjPtr Variable::getUnscaledValue()
{
   return shared_from_this();
}

double Variable::getSign()
{
    return 1.0;
}

EqObjPtr Variable::getUnsignedValue()
{
   return shared_from_this();
}


Variable::Variable(std::string var) : EquationObject(VARIABLE_OBJ), value(var)
{
}

std::string Variable::createStringValue() const
{
    return value;
}

EqObjPtr Variable::Derivative(EqObjPtr foo)
{
    if (foo->stringValue() == value)
        return con(1.0);
    else
        return con(0.0);
}

EqObjPtr Variable::Simplify()
{
   return shared_from_this();
}

/**
 * This function may never be called if variables
 * are turned into Pow's.
 */
EqObjPtr Variable::CombineProduct(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return shared_from_this();

   y.push_back(shared_from_this());
   return EqObjPtr(new Product(y));
}

EqObjPtr Variable::CombineAdd(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return EqObjPtr(shared_from_this());

   y.push_back(shared_from_this());
   return EqObjPtr(new Add(y));
}

bool Variable::isZero()
{
   return false;
}

bool Variable::isOne()
{
   return false;
}

EqObjPtr Variable::clone()
{
    EquationObject *n = new Variable(value);
    return EqObjPtr(n);
}

EqObjPtr Variable::subst(const std::string &str, EqObjPtr eqo)
{
    if (str == this->stringValue())
        return eqo;
    else
        return shared_from_this();
}

EqObjPtr Variable::expand()
{
    return shared_from_this();
}

EqObjPtr Variable::getReciprocal()
{
  assert(false);
  return con(0);
}
}

