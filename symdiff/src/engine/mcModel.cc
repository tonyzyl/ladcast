/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "mcModel.hh"
#include "mcModelCompiler.hh"
#include "Functions.hh"
#include "Context.hh"

namespace Eqo {
EqObjPtr Model::getScale()
{
   return con(1.0);
}

EqObjPtr Model::getUnscaledValue()
{
   return shared_from_this();
}

double Model::getSign()
{
    return 1.0;
}

EqObjPtr Model::getUnsignedValue()
{
   return shared_from_this();
}

EqObjPtr Model::expand()
{
   return shared_from_this();
}

EqObjPtr Model::clone()
{
   return mod(value);
}

EqObjPtr Model::subst(const std::string &str, EqObjPtr eqo)
{
    if (str == this->stringValue())
    {
//// TODO: how does this work
      return eqo->clone();
    }
    else
    {
      return shared_from_this();
    }
}

Model::Model(std::string var) : EquationObject(MODEL_OBJ), value(var)
{
}

std::string Model::createStringValue() const
{
    return value;
}

extern Eqo::EqObjPtr DLL_PROTECTED default_derivative_rule(Eqo::EqObjPtr self, Eqo::EqObjPtr foo)
{
    Context &context = Context::GetInstance();
    const std::string &value = self->stringValue();

    if (foo->stringValue() == value)
    {
        return con(1.0);
    }

    EqObjPtr dval;

    if (context.IsInModelList(value))
    {
      EqObjPtr eq = context.FindInModelList(value);
      if (eq)
      {
        dval =  eq->Derivative(foo)->Simplify();
      }
    }

    EqObjPtr ret;

    {
      std::string bar(value);
      bar += "__"; 
      bar += foo->stringValue();

      if (!context.IsInModelList(bar))
      {
         context.DefineModel(bar, dval);
      }
      else
      {
/// model already defined elsewhere, should we redefine it
         context.DefineModel(bar, dval);
      }
      ret = mod(bar);
    }

    return ret;
}

EqObjPtr Model::Derivative(EqObjPtr foo)
{
  Context &context = Context::GetInstance();
  return context.EvaluateModelDerivative(shared_from_this(), foo);
}

EqObjPtr Model::Simplify()
{
   return shared_from_this();
}

/**
 * This function may never be called if variables
 * are turned into Pow's.
 */
EqObjPtr Model::CombineProduct(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return shared_from_this();

   y.push_back(shared_from_this());
   return EqObjPtr(new Product(y));
}

EqObjPtr Model::CombineAdd(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return shared_from_this();

   y.push_back(shared_from_this());
   return EqObjPtr(new Add(y));
}

bool Model::isZero()
{
   return false;
}

bool Model::isOne()
{
   return false;
}

EqObjPtr Model::getReciprocal()
{
  assert(false);
  return con(0);
}
}

