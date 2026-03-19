/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/
#include "Constant.hh"
#include "Functions.hh"
#include <sstream>
#include <iomanip>
#include <cmath>
namespace Eqo {

EqObjPtr Constant::getUnscaledValue()
{
   return con(1);
}

double Constant::getSign()
{
    const double sign = (dvalue < 0.0) ? -1.0 : 1.0;
    return sign;
}

EqObjPtr Constant::getUnsignedValue()
{
    const double val = (dvalue < 0.0) ? -dvalue : dvalue;
    return con(val);
}

Constant::Constant(double x) : EquationObject(CONST_OBJ), dvalue(x)
{
}

std::string Constant::createStringValue() const
{
    std::ostringstream os;
    if (dvalue != floor(dvalue))
    {
      os << std::scientific << std::setprecision(15);
    }
    if (dvalue < 0.0)
    {
      os << "(" << dvalue << ")";
    }
    else
    {
      os << dvalue;
    }
    return std::string(os.str());
}

EqObjPtr Constant::Derivative(EqObjPtr foo)
{
    EqObjPtr dog = con(0);
    return dog;
}

EqObjPtr Constant::Simplify()
{
   return this->clone();
}

EqObjPtr Constant::CombineProduct(std::vector<EqObjPtr> y)
{
   double x=dvalue;
   size_t len=y.size();
   for (size_t i=0; i < len; ++i)
   {
      Constant *Y = static_cast<Constant *>(y[i].get());
      x *= Y->dvalue;
   }
   return EqObjPtr(new Constant(x));
}

EqObjPtr Constant::CombineAdd(std::vector<EqObjPtr> y)
{
    EqObjPtr ret;
    double               dsum(0.0);
    dsum += this->getDoubleValue();


    size_t len=y.size();
    for (size_t i=0; i < len; ++i)
    {
        Constant *Y = static_cast<Constant *>(y[i].get());

        dsum += Y->getDoubleValue();
    }
    return con(dsum);
}

bool Constant::isZero()
{
   return (this->getDoubleValue()==0.0);
}

bool Constant::isOne()
{
   return (this->getDoubleValue()==1.0);
}

EqObjPtr Constant::clone()
{
    return con(dvalue);
}

EqObjPtr Constant::subst(const std::string &str, EqObjPtr eqo)
{
    if (str == this->stringValue())
        return eqo;
    else
        return shared_from_this();
}

EqObjPtr Constant::expand()
{
    return shared_from_this();
}

EqObjPtr Constant::getScale()
{
   return shared_from_this();
}

EqObjPtr Constant::getReciprocal()
{
  return con(1.0/dvalue);
}
}

