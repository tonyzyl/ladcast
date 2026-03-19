/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "Add.hh"
#include "Utility.hh"
#include "Functions.hh"
#include <sstream>

namespace Eqo {
Add::Add(EqObjPtr one, EqObjPtr two) : EquationObject(ADD_OBJ)
{
   values.resize(2);
   values[0]=one;
   values[1]=two;
}

Add::Add(std::vector<EqObjPtr> one) : EquationObject(ADD_OBJ)
{
   values = one;
   SortEqVector(values);
}

std::string Add::createStringValue() const
{
   const size_t len = values.size();
   std::ostringstream os;
   os << "(";
   if (len > 0)
   {
        if (values[0]->getSign() > 0.0)
            os << values[0];
        else
            os << "-" << values[0]->getUnsignedValue();
   }
   for (size_t i=1; i < len; ++i)
   {
        if (values[i]->getSign() > 0.0)
            os << " + " << values[i];
        else
            os << " - " << values[i]->getUnsignedValue();
   }
   os << ")";
    return os.str();
}

EqObjPtr Add::Derivative(EqObjPtr foo)
{
//   cout << EqObjPtr(new Add(values)) << "\t" << foo << endl;
   const size_t len=values.size();
   std::vector<EqObjPtr> vec;
   vec.reserve(len);

   for (size_t i = 0; i < len; ++i)
   {
      EqObjPtr dptr = values[i]->Derivative(foo);
      if (!dptr->isZero())
         vec.push_back(dptr);
   }

   EqObjPtr out;
   if (vec.empty())
      out = con(0);
   else if (vec.size()==1)
      out = vec[0];
   else
      out = EqObjPtr(new Add(vec));

//   cout << out << endl;
   return out;
}

EqObjPtr Add::Simplify()
{
   const size_t tlen = values.size();

   if (tlen==1)
      return values[0]->Simplify();

   std::vector<EqObjPtr> tvals;
   tvals.reserve (tlen);
   for (size_t i=0; i < tlen; ++i)
   {
      tvals.push_back(values[i]->Simplify());
   }

   // When each section is done swap it back into tvals

   {
      // Combine elements which match each other's types
      const size_t len = tvals.size();
      std::vector<bool> isDone(len, false);
      std::vector<EqObjPtr> tmp;
      tmp.reserve(2*tvals.size());

      for (size_t i = 0; i < len; ++i)
      {
         if (isDone[i])
            continue;

         isDone[i] = true;

         EqObjPtr tptr = tvals[i];
         std::vector<EqObjPtr> tmpvec;

         for (size_t j = i + 1; j < len; ++j)
         {
            if (isDone[j])
               continue;

            EqObjPtr uptr = tvals[j];

            if ((tptr.get()->getType())==(uptr.get()->getType()))
            {
               tmpvec.push_back(uptr);
               isDone[j] = true;
               continue;
            }
         }
         if (tmpvec.empty())
            tmp.push_back(tptr);
         else
            tmp.push_back(tptr->CombineAdd(tmpvec));
      }
      swap(tmp,tvals); // swap back;
   }


   {
      // Combine all additions, including ((x+y)+z)
      const size_t len = tvals.size();
      std::vector<EqObjPtr> tmp;
      tmp.reserve(2*len);

      for (size_t i = 0; i < len; ++i) 
      {
         Add *foo = dynamic_cast<Add *>(tvals[i].get());
         if (foo == nullptr)
         {
            tmp.push_back(tvals[i]);
         }
         else
         {
            size_t sublen = foo->values.size();
            for (size_t j=0; j < sublen; ++j)
            {
               tmp.push_back(foo->values[j]);
            }
         }
      }
      swap(tmp,tvals); // swap back;
   }

   {
      // Remove zero entries
      const size_t len = tvals.size();
      std::vector<EqObjPtr> tmp;
      tmp.reserve(len);
      for (size_t i = 0; i < len; ++i) 
      {
         EqObjPtr tptr = tvals[i] -> Simplify();
         if (!tptr->isZero())
            tmp.push_back(tptr);
      }
      swap(tmp,tvals); // swap back;
   }

   // Adds terms with different factors
   {
      size_t len=tvals.size();
      std::vector <EqObjPtr> tmp;
      tmp.reserve(len);

      std::vector<bool> isDone(len,false);
      for (size_t i=0; i < len; ++i)
      {
         if (isDone[i])
            continue;
         isDone[i]=true;

         EqObjPtr tptr = tvals[i];

         if (tptr->isZero())
            continue;

         const std::string name = tptr->getUnscaledValue()->stringValue();
//       cout << name << endl;

         std::vector<EqObjPtr> factvec;
         factvec.push_back(tptr->getScale());

         for (size_t j=(i+1); j < len; ++j)
         {
            if (isDone[j])
               continue;

            EqObjPtr uptr = tvals[j];

            if (uptr->isZero())
            {
               isDone[j] = true;
               continue;
            }

            if (name == uptr->getUnscaledValue()->stringValue())
            {
               isDone[j] = true;
               factvec.push_back(uptr->getScale());
            }
         }


         EqObjPtr fact = EqObjPtr(new Add(factvec))->Simplify();

         if (fact->isOne())
            tmp.push_back(tptr->getUnscaledValue());
         else
            tmp.push_back((fact*tptr->getUnscaledValue())->Simplify());
      }
      swap(tmp,tvals); // swap back;
   }

   if (tvals.empty())
      return con(0);
   else if (tvals.size()==1)
      return tvals[0];

   return EqObjPtr(new Add(tvals));
}

EqObjPtr Add::CombineProduct(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return shared_from_this();

   y.push_back(shared_from_this());
   return (EqObjPtr(new Product(y)));
}

EqObjPtr Add::CombineAdd(std::vector<EqObjPtr> y)
{
   std::vector<EqObjPtr> tmp = values;
   const size_t len = y.size();
   for (size_t i=0; i < len; ++i)
   {
      Add *Y = static_cast<Add *>(y[i].get());
      const size_t len2 = (Y->values).size();
      for (size_t j=0; j < len2; ++j)
         tmp.push_back(Y->values[j]);
   }

   return EqObjPtr(new Add(tmp));
}

EqObjPtr Add::clone()
{
    const size_t len = values.size();
    std::vector<EqObjPtr> tmp(len);
    for (size_t i = 0; i < len; ++i)
    {
        tmp[i] = values[i]->clone();
    }
    EquationObject *n = new Add(tmp);
    return EqObjPtr(n);
}

EqObjPtr Add::subst(const std::string &str, EqObjPtr eqo)
{
    if (str == this->stringValue())
        return eqo;
    else
    {
        std::vector<EqObjPtr> new_values(values.size());
        for (size_t i=0; i < values.size(); ++i)
        {
            new_values[i] = values[i]->subst(str, eqo);
        }
        return EqObjPtr(new Add(new_values));
    }
}

bool Add::isZero()
{
   return false;
}

bool Add::isOne()
{
   return false;
}

EqObjPtr Add::getScale()
{
   return con(1.0);
}

EqObjPtr Add::getUnscaledValue()
{
   return shared_from_this();
}

double Add::getSign()
{
    return 1.0;
}

EqObjPtr Add::getUnsignedValue()
{
   return shared_from_this();
}

EqObjPtr Add::expand()
{
    const size_t len = values.size();
    std::vector<EqObjPtr> tmp(len);
    for (size_t i = 0; i < len; ++i)
    {
        tmp[i] = values[i]->expand();
    }
    EquationObject *n = new Add(tmp);
    return EqObjPtr(n);
}

EqObjPtr Add::getReciprocal()
{
  assert(false);
  return con(0);
}

}

