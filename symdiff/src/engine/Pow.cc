/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "Pow.hh"
#include "Functions.hh"
#include <sstream>
#include <cmath>
#include <cstdlib>

namespace Eqo {
Pow::Pow(EqObjPtr b, EqObjPtr e) : EquationObject(POW_OBJ),
   base(b), exponent(e)
{
}

std::string Pow::createStringValue() const
{
    std::ostringstream os;
    // need to control the choice between printing behavior
    os << "pow(" << base << "," << exponent << ")";
//    os << "(" << base << ")^(" << exponent << ")";
//    os << "(" << base << "^" << exponent << ")";
    return std::string(os.str());
}

EqObjPtr Pow::Derivative(EqObjPtr foo)
{
  EqObjPtr ret;
  if (exponent->getType() == CONST_OBJ)
  {
    double ex =static_cast<Constant *>(exponent.get())->getDoubleValue();
    //// x^0 is 1 diff(2^3, x) is 0
    //// it doesn't matter if we consider 0^0 == 0 or 0^0 == 1
    //// since the derivative is the same
    if ((ex == 0.0) || (base->getType() == CONST_OBJ))
    {
      ret = con(0.0);
    }
    /// diff(y^1, x) is diff(y,x)
    else if (ex == 1.0)
    {
      ret = diff(base, foo);
    }
    /// diff(y^2,x) is 2 * y * diff(y,x)
    else if (ex == 2.0)
    {
      ret = con(2.0) * base * diff(base, foo);
    }
    //// diff(y^3, x) is 3*y^2
    else
    {
      ret = con(ex) * pow(base, con(ex-1.0))*diff(base, foo);
    }
  }
  else if (base->getType() == CONST_OBJ)
  {
    //diff(0^y, x) == 0 no matter what
    //// 1^x is 1
    if (base->isZero() || base->isOne())
    {
      ret = con(0.0);
    }
    /// We already handled the case of the exponent being constant above
    else
    {
      ret = log(base) * shared_from_this() * diff(exponent, foo);
    }
  }
  else
  {
     ret = exponent*pow(base,exponent - con(1)) * diff(base, foo) + log(base)*(this->clone())*diff(exponent,foo);
//     ret = diff((log(base)*exponent),foo)*pow(base,exponent);
  }
  return ret;
}

/**
 * if exponent is 0, return 1.0
 * if exponent is 1.0, return base
 */
EqObjPtr Pow::Simplify()
{
   EqObjPtr b  = base->Simplify();
   EqObjPtr expo = exponent->Simplify();

   /// pow(x,0)=1
   if (expo->isZero())
      return con(1);

   /// pow(x,1)=x
   if (expo->isOne())
      return base;

   /// pow(1,x)=1
   if (b->isOne())
      return con(1);

   /// pow(0,x)=0
   if (b->isZero())
      return con(0.0);

   /// pow(m,n)
   if ((b->getType() == CONST_OBJ)
         && (expo->getType() == CONST_OBJ))
   {
      Constant *x=static_cast<Constant *>(b.get());
      Constant *y=static_cast<Constant *>(expo.get());
      return con(std::pow(x->getDoubleValue(),y->getDoubleValue()));
   }

   /// may want to do static cast
   /// pow(pow(x,y),z) = pow(x,(y*z))
   if (b->getType() == POW_OBJ)
   {
      Pow *Y = static_cast<Pow *>(b.get());
//      cout << Y-> base << " " <<  (Y->exponent * exponent) << endl;
      return pow(Y->base, (Y->exponent * expo)->Simplify());
   }
   return pow(b, expo);
}

/**
 * If the same base is shared, add the exponents
 *
 * In the future, if the same exponent is shared
 * combine the bases as a product
 */
EqObjPtr Pow::CombineProduct(std::vector<EqObjPtr> y)
{
   y.push_back(pow(base,exponent));
   const size_t len = y.size();
   std::vector<bool> isDone(len,false);
   std::vector <EqObjPtr> out;

   // If the same base is shared, add the exponents
   for (size_t i=0; i < len; ++i)
   {
      if (isDone[i])
         continue;
      isDone[i]=true;
      Pow *Y1 = static_cast<Pow *>(y[i].get());
      std::vector <EqObjPtr> tmp_exp;
      tmp_exp.push_back(Y1->exponent);

      for (size_t j=i+1; j < len; ++j)
      {
         if (isDone[j])
            continue;

         Pow *Y2 = static_cast<Pow *>(y[j].get());
         if ((Y1->base)->stringValue() == (Y2->base)->stringValue())
         {
            tmp_exp.push_back(Y2->exponent);
            isDone[j]=true;
         }
      }
      if (tmp_exp.size() == 1)
         out.push_back(pow(Y1->base, tmp_exp[0]));
      else
         out.push_back(pow(Y1->base, EqObjPtr(new Add(tmp_exp))));
   }

   if (out.size()==1)
      return out[0];

   return EqObjPtr(new Product(out));
}

EqObjPtr Pow::CombineAdd(std::vector<EqObjPtr> y)
{
   y.push_back(pow(base,exponent));
   return EqObjPtr(new Add(y));
}

bool Pow::isZero()
{
   return false;
}

bool Pow::isOne()
{
   return false;
}

EqObjPtr Pow::getScale()
{
   return con(1);
}

EqObjPtr Pow::getUnscaledValue()
{
   return pow(base,exponent);
}

double Pow::getSign()
{
    return 1.0;
}

EqObjPtr Pow::getUnsignedValue()
{
   return shared_from_this();
}

EqObjPtr Pow::clone()
{
    EquationObject *n = new Pow(base->clone(), exponent->clone());
    return EqObjPtr(n);
}

EqObjPtr Pow::subst(const std::string &str, EqObjPtr eqo)
{
    if (str == this->stringValue())
        return eqo;
    else
    {
        EqObjPtr new_base = base->subst(str, eqo);
        EqObjPtr new_exponent = exponent->subst(str, eqo);
        return EqObjPtr(pow(new_base, new_exponent));
    }
}

// TODO: refacter dangerous code using doubles
// this is very flaky code
// need to make sure that integer exponents are handled properly
// make sure to clone the object before expansion
EqObjPtr Pow::expand()
{
    EqObjPtr expanded_base = base->expand();
    EqObjPtr expanded_exponent = exponent->expand();

    EqObjPtr out = pow(expanded_base, expanded_exponent);
    // want to factor out order greater than 1
//    cout << exponent->getType() << endl;
    if (expanded_exponent->getType() == CONST_OBJ)
    {
        Constant *Y = static_cast<Constant *>(expanded_exponent.get());
        double dval = Y->dvalue;
        bool negative = (dval < 0.0);
        if (dval < 0.0)
        {
            negative = true;
            dval = -dval;
        }

        std::ostringstream os;
        os << dval;
        std::string tstr = os.str();
        // look for non floating point numbers
        // arbitrarily restrict exponents less than 100
        // warning magic number
//      cout << "DEBUG tstr" << tstr << endl;
        if ((dval < 100) && (tstr.find('.') == std::string::npos))
        {
            const int ex = atoi(tstr.c_str());
            if (ex > 1)
            {
                // assume already expanded above
                EqObjPtr newbase = expanded_base->clone();
                out = newbase;
                for (int i = 1; i < ex; ++i)
                {
                    out = out*newbase;
                }
                out = out->expand();
                if (negative)
                    out = pow(out,con(-1));
            }
        }

    }

    return out;
}

EqObjPtr Pow::getReciprocal() {
  assert(false);
  return con(0);
}

}

