/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef INLINE_FUNCTIONS_HH
#define INLINE_FUNCTIONS_HH
#include "EquationObject.hh"
#include "Product.hh"
#include "Add.hh"
#include "Exponent.hh"
#include "Log.hh"
#include "Pow.hh"
#include "Constant.hh"
#include "Variable.hh"
namespace Eqo {
/// Non member functions for intuitive math operations
inline EqObjPtr con(double x)
{
   return EqObjPtr(new Constant(x));
}

inline EqObjPtr operator* (EqObjPtr x, EqObjPtr y)
{
  EqObjPtr ret;
  if (x->isZero() || y->isZero())
  {
    ret = con(0.0);
  }
  else if (x->isOne())
  {
    ret = y;
  }
  else if (y->isOne())
  {
    ret = x;
  }
  else if (x->getType() == Eqo::PRODUCT_OBJ)
  {
    std::vector<EqObjPtr> x_args = x->getArgs();

    if (y->getType() == Eqo::PRODUCT_OBJ)
    {
      std::vector<EqObjPtr> y_args = y->getArgs();
      for (std::vector<EqObjPtr>::iterator it =  y_args.begin(); it != y_args.end(); ++it)
      {
        x_args.push_back(*it);
      }
    }
    else
    {
      x_args.push_back(y);
    }

    ret = EqObjPtr(new Product(x_args));
  }
  else if (y->getType() == Eqo::PRODUCT_OBJ)
  {
    std::vector<EqObjPtr> y_args = y->getArgs();
    std::vector<EqObjPtr> x_args;
    x_args.reserve(y_args.size() + 1);
    x_args.push_back(x);
    for (std::vector<EqObjPtr>::iterator it =  y_args.begin(); it != y_args.end(); ++it)
    {
      x_args.push_back(*it);
    }
    ret = EqObjPtr(new Product(x_args));
  }
  else
  {
    ret = EqObjPtr(new Product(x, y));
  }

  return ret;
}

EqObjPtr pow (EqObjPtr x, EqObjPtr y);

inline EqObjPtr operator/ (EqObjPtr x, EqObjPtr y)
{
    if (y->hasReciprocal())
    {
        if ((x->getType() == CONST_OBJ) && (y->getType() == CONST_OBJ))
        {
            return EqObjPtr(new Product(x, y->getReciprocal()))->Simplify();
        }
    }
   return EqObjPtr(new Product(x, pow(y, con(-1))));
}

inline EqObjPtr operator+ (EqObjPtr x, EqObjPtr y)
{
  EqObjPtr ret;
  if (x->isZero())
  {
    ret = y;
  }
  else if (y->isZero())
  {
    ret = x;
  }
  else if (x->getType() == Eqo::ADD_OBJ)
  {
    std::vector<EqObjPtr> x_args = x->getArgs();

    if (y->getType() == Eqo::ADD_OBJ)
    {
      std::vector<EqObjPtr> y_args = y->getArgs();
      for (std::vector<EqObjPtr>::iterator it =  y_args.begin(); it != y_args.end(); ++it)
      {
        x_args.push_back(*it);
      }
    }
    else
    {
      x_args.push_back(y);
    }

    ret = EqObjPtr(new Add(x_args));
  }
  else if (y->getType() == Eqo::ADD_OBJ)
  {
    std::vector<EqObjPtr> y_args = y->getArgs();
    std::vector<EqObjPtr> x_args;
    x_args.reserve(y_args.size() + 1);
    x_args.push_back(x);
    for (std::vector<EqObjPtr>::iterator it =  y_args.begin(); it != y_args.end(); ++it)
    {
      x_args.push_back(*it);
    }

    ret = EqObjPtr(new Add(x_args));
  }
  else
  {
    ret = EqObjPtr(new Add(x,y));
  }

  return ret;
}

inline EqObjPtr operator- (EqObjPtr x, EqObjPtr y)
{
   return (x + (con(-1)*y));
}

inline EqObjPtr exp (EqObjPtr x)
{
   return EqObjPtr(new Exponent(x));
}

inline EqObjPtr pow (EqObjPtr x, EqObjPtr y)
{
   return EqObjPtr(new Pow(x, y));
}

inline EqObjPtr log (EqObjPtr x)
{
   return EqObjPtr(new Log(x));
}

inline EqObjPtr diff(EqObjPtr x, EqObjPtr y)
{
   return x->Derivative(y);
}

inline EqObjPtr var(const char *x)
{
   return EqObjPtr(new Variable(std::string(x)));
}

inline EqObjPtr var(const std::string &x)
{
   return EqObjPtr(new Variable(x));
}

inline EqObjPtr sqrt(EqObjPtr x)
{
    return EqObjPtr(pow(x, con(0.5)));
}

inline EqObjPtr reciprocal_sqrt(EqObjPtr x)
{
    return EqObjPtr(pow(x, con(-0.5)));
}

/// keeps simplifying expression until string value doesn't change
inline EqObjPtr Simplify(EqObjPtr x)
{
   std::string y = x->stringValue();
   EqObjPtr z = x->Simplify();
   while (y != z->stringValue())
   {
      y = z -> stringValue();
      z = z -> Simplify();
   }
   return z;
}

inline EqObjPtr getConstantFactor(EqObjPtr x)
{
   return x->getScale();
}

inline EqObjPtr Expand(EqObjPtr x)
{
   std::string str = x->stringValue();
   EqObjPtr    eq = x->expand();
   while (str != eq->stringValue())
   {
       str = eq->stringValue();
       eq = eq->expand();
   }
   return Simplify(eq);
}

inline EqObjPtr getUnscaledValue(EqObjPtr x)
{
   return x->getUnscaledValue();
}

inline EqObjPtr getUnsignedValue(EqObjPtr x)
{
   return x->getUnsignedValue();
}

inline EqObjPtr getSign(EqObjPtr x)
{
   return EqObjPtr(new Constant(x->getSign()));
}

   /// Sorts in order of constants, variables, then other objects
   void SortEqVector(std::vector<EqObjPtr> &v1);

inline EqObjPtr subst(EqObjPtr a, EqObjPtr b, EqObjPtr c)
{
    return a->subst(b->stringValue(), c);
}

inline EqObjPtr getNegation(EqObjPtr x)
{
    EqObjPtr ret = Eqo::con(-1) * x;
    if (x->getType() == Eqo::CONST_OBJ)
    {
        ret = ret->Simplify();
    }
    return ret;
}
}
#endif


