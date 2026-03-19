/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "IfElse.hh"
#include "Functions.hh"
#include <sstream>

namespace Eqo {
std::set<std::string> IfObj::getReferencedType(Eqo::EqObjType rt)
{
    std::vector<EqObjPtr> x;
    x.push_back(test);
    x.push_back(arg);
    return UniteReferencedType(rt, x);
}

std::set<std::string> IfElseObj::getReferencedType(Eqo::EqObjType rt)
{
    std::vector<EqObjPtr> x;
    x.push_back(test);
    x.push_back(arg1);
    x.push_back(arg2);
    return UniteReferencedType(rt, x);
}


IfObj::IfObj(EqObjPtr t, EqObjPtr a) : EquationObject(IF_OBJ), test(t), arg(a)
{
}

std::string IfObj::createStringValue() const
{
    std::ostringstream os;
    os << "(if ("
         << test->stringValue()
         << ", "
         << arg->stringValue() << "))";
    return os.str();
}

EqObjPtr IfObj::Derivative(EqObjPtr foo)
{
    return EqObjPtr(new IfObj(test->clone(), arg->Derivative(foo)));
}

EqObjPtr IfObj::Simplify()
{
    return EqObjPtr(new IfObj(test->Simplify(), arg->Simplify()));
}

/// let it just be what it was
EqObjPtr IfObj::CombineProduct(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return (shared_from_this());

   y.push_back(shared_from_this());
   return EqObjPtr(new Product(y));
}

EqObjPtr IfObj::CombineAdd(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return (shared_from_this());

   y.push_back(shared_from_this());
   return EqObjPtr(new Add(y));
}

EqObjPtr IfObj::clone()
{
    return EqObjPtr(new IfObj(test->clone(), arg->clone()));
}

bool IfObj::isZero()
{
    return false;
}

bool IfObj::isOne()
{
    return false;
}

EqObjPtr IfObj::getUnscaledValue()
{
   return shared_from_this();
}

double IfObj::getSign()
{
    return 1.0;
}

EqObjPtr IfObj::getUnsignedValue()
{
   return shared_from_this();
}

EqObjPtr IfObj::expand()
{
   return shared_from_this();
}

EqObjPtr IfObj::getScale()
{
   return con(1.0);
}

EqObjPtr IfObj::subst(const std::string &str, EqObjPtr eqo)
{
    if (str == this->stringValue())
    {
      return eqo->clone();
    }
    else
    {
      return EqObjPtr(new IfObj(test->subst(str, eqo), arg->subst(str,eqo)));
    }
}

EqObjPtr IfObj::getReciprocal()
{
  assert(false);
  return con(0);
}

IfElseObj::IfElseObj(EqObjPtr t, EqObjPtr a1, EqObjPtr a2) : EquationObject(IFELSE_OBJ), test(t), arg1(a1), arg2(a2)
{
}

std::string IfElseObj::createStringValue() const
{
    std::ostringstream os;
    os << "(ifelse ("
         << test->stringValue()
         << ", "
         << arg1->stringValue() << ", "
         << arg2->stringValue() << "))";
    return os.str();
}

EqObjPtr IfElseObj::Derivative(EqObjPtr foo)
{
    return EqObjPtr(new IfElseObj(test->clone(), arg1->Derivative(foo), arg2->Derivative(foo)));
}

EqObjPtr IfElseObj::Simplify()
{
    return EqObjPtr(new IfElseObj(test->Simplify(), arg1->Simplify(), arg2->Simplify()));
}

/// let it just be what it was
EqObjPtr IfElseObj::CombineProduct(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return (shared_from_this());

   y.push_back(shared_from_this());
   return EqObjPtr(new Product(y));
}

EqObjPtr IfElseObj::CombineAdd(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return (shared_from_this());

   y.push_back(shared_from_this());
   return EqObjPtr(new Add(y));
}

EqObjPtr IfElseObj::clone()
{
    return EqObjPtr(new IfElseObj(test->clone(), arg1->clone(), arg2->clone()));
}

bool IfElseObj::isZero()
{
    return false;
}

bool IfElseObj::isOne()
{
    return false;
}

EqObjPtr IfElseObj::getUnscaledValue()
{
   return shared_from_this();
}

double IfElseObj::getSign()
{
    return 1.0;
}

EqObjPtr IfElseObj::getUnsignedValue()
{
   return shared_from_this();
}

EqObjPtr IfElseObj::expand()
{
   return shared_from_this();
}

EqObjPtr IfElseObj::getScale()
{
   return con(1.0);
}

EqObjPtr IfElseObj::subst(const std::string &str, EqObjPtr eqo)
{
  if (str == this->stringValue())
  {
    return eqo->clone();
  }
  else
  {
    return EqObjPtr(new IfElseObj(test->subst(str, eqo), arg1->subst(str,eqo), arg2->subst(str,eqo)));
  }
}

EqObjPtr IfElseObj::getReciprocal()
{
  assert(false);
  return con(0);
}

}

