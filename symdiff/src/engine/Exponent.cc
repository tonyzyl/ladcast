/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "Exponent.hh"
#include "Functions.hh"
#include <sstream>

namespace Eqo {
Exponent::Exponent(EqObjPtr var) : EquationObject(EXPONENT_OBJ), value(var)
{
}

std::string Exponent::createStringValue() const
{
    std::ostringstream os;
    os << "exp(" << value << ")";
    return std::string(os.str());
}

EqObjPtr Exponent::Derivative(EqObjPtr foo)
{
  if (value->getType() == CONST_OBJ)
  {
    return con(0.0);
  }
  else
  {
     return value->Derivative(foo)*exp(value);
  }
}

EqObjPtr Exponent::Simplify()
{
   if (value->isZero())
      return con(1);

   if (value->getType() == LOG_OBJ)
   {
      Log *Y = dynamic_cast<Log *>(value.get());
      return (Y->value);
   }

   return exp(value->Simplify());
}

/**
 * exp(y)*exp(x)=exp(x+y)
 */
EqObjPtr Exponent::CombineProduct(std::vector<EqObjPtr> y)
{
   std::vector<EqObjPtr> tmp;
   tmp.push_back(value);
   const size_t len = y.size();
   for (size_t i=0; i < len; ++i)
   {
      Exponent *Y = dynamic_cast<Exponent *>(y[i].get());
      tmp.push_back(Y->value);
   }
   return EqObjPtr(new Exponent(EqObjPtr(new Add(tmp))));
}

EqObjPtr Exponent::CombineAdd(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return shared_from_this();

   y.push_back(EqObjPtr(new Exponent(value)));
   return EqObjPtr(new Add(y));
}

bool Exponent::isZero()
{
   return false;
}

bool Exponent::isOne()
{
   return false;
}

EqObjPtr Exponent::getScale()
{
   return con(1.0);
}

EqObjPtr Exponent::getUnscaledValue()
{
   return shared_from_this();
}

double Exponent::getSign()
{
    return 1.0;
}

EqObjPtr Exponent::getUnsignedValue()
{
   return shared_from_this();
}

EqObjPtr Exponent::clone()
{
    EquationObject *n = new Exponent(value->clone());
    return EqObjPtr(n);
}

EqObjPtr Exponent::subst(const std::string &str, EqObjPtr eqo)
{
    if (str == this->stringValue())
        return eqo;
    else
        return exp(value->subst(str, eqo));
}

EqObjPtr Exponent::expand()
{
    return exp(value->expand());
}

EqObjPtr Exponent::getReciprocal()
{
  assert(false);
  return con(0);
}
}

