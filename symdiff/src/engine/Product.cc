/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "Product.hh"
#include "Functions.hh"
#include <sstream>

namespace Eqo {
Product::Product(EqObjPtr one, EqObjPtr two) : EquationObject(PRODUCT_OBJ)
{

   values.resize(2);
   values[0]=one;
   values[1]=two;
}

Product::Product(std::vector<EqObjPtr> one) : EquationObject(PRODUCT_OBJ)
{
   values = one;
   SortEqVector(values);
}

std::string Product::createStringValue() const
{
    const size_t len = values.size();
    std::ostringstream os;

    os << "(";
    if (len > 0)
    {
        const std::string &s = values[0]->stringValue();
        size_t beg = 1;
        if (len > 1 && s == "(-1)")
        {
            os << "-" << values[1];
            beg = 2;
        }
        else
        {
            os << s;
        }

        for (size_t i=beg; i < len; ++i)
            os << " * " << values[i];
    }
    os << ")";

#if 0
    if (this->getSign() > 0.0)
    {
        os << "(";
        if (len > 0)
            os << values[0];
        for (size_t i=1; i < len; ++i)
            os << " * " << values[i];
        os << ")";
    }
    else
    {
        os << "-" << this->getUnsignedValue()->stringValue();
    }
#endif

    return os.str();
}

EqObjPtr Product::Derivative(EqObjPtr foo)
{
//   cout << "in  :" << EqObjPtr(new Product(values)) << " " << foo << endl;

   const size_t len=values.size();
   std::vector<EqObjPtr> vec_add;
   vec_add.reserve(len);

   // Apply chain rule for each term
   for (size_t i = 0; i < len; ++i)
   {
      std::vector<EqObjPtr> vec_prod;
      vec_prod.reserve(len);

      EqObjPtr dptr = values[i]->Derivative(foo);
      if (dptr->isZero())
         continue;

      if (!dptr->isOne())
         vec_prod.push_back(dptr);

      for (size_t j = 0; j < len; ++j)
      {
         if (i == j)
            continue;
         vec_prod.push_back(values[j]);
      }
      if (vec_prod.empty())
         continue;
      else if (vec_prod.size()==1)
         vec_add.push_back(vec_prod[0]);
      else
         vec_add.push_back(EqObjPtr(new Product(vec_prod)));
   }

   EqObjPtr out;
   if (vec_add.empty())
      out = con(0);
   else if (vec_add.size()==1)
      out = vec_add[0];
   else
      out = EqObjPtr(new Add(vec_add));

//   cout << "out  :" << out << endl;
   return out;
}

    // Performs the various simplification rules.
void Product::ProductVecSimplify(std::vector<EqObjPtr> &tvals)
{
   // When each section is done swap it back into tvals
   {
      // Combine Products together
      const size_t len = tvals.size();
      std::vector<bool> isDone(len, false);
      std::vector<EqObjPtr> tmp;
      tmp.reserve(2*tvals.size());

      // Convert variable's to pow's for combination
      std::vector<EqObjPtr> tmp2;
      tmp2.reserve(len);
      for (size_t i = 0; i < len; ++i)
      {
         if (tvals[i]->getType()==VARIABLE_OBJ)
            tmp2.push_back(pow(tvals[i],con(1)));
         else
            tmp2.push_back(tvals[i]);
      }
      tvals.swap(tmp2);

      // identifies objects of the same type and tries to combine them
      // Calls the CombineProduct for each type
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

            // Combine elements of the same type
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
            tmp.push_back(tptr->CombineProduct(tmpvec));
      }
      swap(tmp,tvals); // swap back;
   }

   // Remove Nested products
   {
      const size_t len = tvals.size();
      std::vector<EqObjPtr> tmp;
      tmp.reserve(2*len);

      for (size_t i = 0; i < len; ++i) 
      {
         Product *foo = dynamic_cast<Product *>(tvals[i].get());
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
      const size_t len = tvals.size();
      std::vector<EqObjPtr> tmp;
      // break down all factors and remove ones
      tmp.reserve(len);
      bool hasOne = false;
      for (size_t i = 0; i < len; ++i) 
      {
         if (!tvals[i]->isOne())
            tmp.push_back(tvals[i]->Simplify());
         else
            hasOne = true;
      }
      if (tmp.empty() && hasOne)
         tmp.push_back(con(1));

      swap(tmp,tvals); // swap back;
   }

}

EqObjPtr Product::Simplify()
{
   if (this->isZero())
      return con(0);

   const size_t tlen = values.size();

   if (tlen==1)
      return values[0]->Simplify();

   std::vector<EqObjPtr> tvals;
   tvals.reserve (tlen);
   for (size_t i=0; i < tlen; ++i)
   {
      tvals.push_back(values[i]->Simplify());
   }

   ProductVecSimplify(tvals);

    EqObjPtr out;
    if (tvals.size()==1)
        out = tvals[0];
    else
        out = EqObjPtr(new Product(tvals));

    // now try method where everything is a pow (Catch (1-x)/(1-x) = (1-x)^1*(1-x)^(-1) = 1)
    // start with already simplified std::vector
    if (tvals.size() > 1)
    {
        std::vector<EqObjPtr> tmp;
        tmp.reserve(tvals.size());
        for (size_t i = 0; i < tvals.size(); ++i)
        {
            if (tvals[i]->getType() != POW_OBJ)
                tmp.push_back(pow(tvals[i], con(1)));
            else
                tmp.push_back(tvals[i]);
        }
        SortEqVector(tmp);
        ProductVecSimplify(tmp);

        EqObjPtr teq;
        if (tmp.size() == 1)
            teq = tmp[0];
        else
            teq = EqObjPtr(new Product(tmp));

        // If the result is longer, don't use it
        if (teq->stringValue().size() < out->stringValue().size())
            out = teq;
    }

    {
        /// now if we are a pow, expand out until a list of 2 factors
        /// x^3 = x^2 * x
        /// Need to catch when exponent ends up zero
        std::vector<EqObjPtr> tmp;
        tmp.reserve(tvals.size());
        for (size_t i = 0; i < tvals.size(); ++i)
        {
            if (tvals[i]->getType() == POW_OBJ)
            {
                Pow *foo = static_cast<Pow *>(tvals[i].get());
                EqObjPtr e = foo->exponent;
                EqObjPtr b = foo->base;

                if (e->isOne())
                {
                    tmp.push_back(b);
                    
                }
                else {
                    if (b->getType() == PRODUCT_OBJ)
                    {
                        Product *bar = static_cast<Product *>(b.get());
                        /// See if we already have a helper function for adding product values to list
                        std::vector<EqObjPtr> xx = bar->values;
                        for (size_t j=0; j < xx.size(); ++j)
                        {
                            tmp.push_back(xx[j]);
                        }
                        tmp.push_back(pow(b,(e - con(1))));

                    }
                    else
                    {
                        tmp.push_back(b);
                        tmp.push_back(pow(b,(e - con(1))));
                    }
                }

            }
            else
            {
                tmp.push_back(tvals[i]->clone());
            }
        }

        size_t j = 0;
        /// Keep reducing while it makes sense
        do
        {
            j = tmp.size();
            ProductVecSimplify(tmp);
        } while (tmp.size() < j);

        EqObjPtr teq;
        if (tmp.size() == 1)
            teq = tmp[0];
        else
            teq = EqObjPtr(new Product(tmp));
        if (teq->stringValue().size() < out->stringValue().size())
            out = teq;

    }

   return out;
}

EqObjPtr Product::CombineProduct(std::vector<EqObjPtr> y)
{
   std::vector<EqObjPtr> tmp = values;
   const size_t len = y.size();
   for (size_t i=0; i < len; ++i)
   {
      Product *Y = static_cast<Product *>(y[i].get());
      const size_t len2 = (Y->values).size();
      for (size_t j=0; j < len2; ++j)
         tmp.push_back(Y->values[j]);
   }

   return EqObjPtr(new Product(tmp));
}

EqObjPtr Product::CombineAdd(std::vector<EqObjPtr> y)
{
   if (y.empty())
      return (EqObjPtr(new Product(values)));

   y.push_back(EqObjPtr(new Product(values)));

   return (EqObjPtr(new Add(y)));
}

bool Product::isZero()
{
   size_t len = values.size();
   for (size_t i=0; i<len; ++i)
   {
      if (values[i]->isZero())
         return true;
   }

   return false;
}

bool Product::isOne()
{
   if (values.empty())
      return false;

   const size_t len = values.size();

   for (size_t i=0; i<len; ++i)
   {
      if (!values[i]->isOne())
         return false;
   }

   return true;
}

/// factors out constants scales
//  /todo would be to factor for arbitrary type
EqObjPtr Product::getScale()
{
   const size_t len = values.size();
   std::vector<EqObjPtr> tmp;
   tmp.reserve(len+1);
   for (size_t i=0; i < len; ++i)
   {
      if (values[i]->getType()==CONST_OBJ)
         tmp.push_back(values[i]);
   }

   if (tmp.empty())
      return con(1);

   return EqObjPtr(new Product(tmp))->Simplify();
}

EqObjPtr Product::getUnscaledValue()
{
   const size_t len = values.size();
   std::vector<EqObjPtr> tmp;
   tmp.reserve(len+1);
   for (size_t i=0; i < len; ++i)
   {
      if (values[i]->getType()!=CONST_OBJ)
         tmp.push_back(values[i]);
   }

   if (tmp.empty())
      return con(1);
   else if (tmp.size()==1)
      return EqObjPtr(tmp[0]);

   // Is this a clone or another reference to the original underlying object
   return EqObjPtr(new Product(tmp));
}

double Product::getSign()
{
    // do I already have one
    int    foundConst = 0;
    double foundSign = 1.0;
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (values[i]->getType() == CONST_OBJ)
        {
            foundSign = values[i]->getSign();
            ++foundConst;
        }

        // only have a negative sign if only one const obj exists
        if (foundConst > 1)
        {
            foundSign = 1.0;
            break;
        }
    }
    return foundSign;
}

EqObjPtr Product::getUnsignedValue()
{
    EqObjPtr ret;
    const double sign = this->getSign();
    if (sign >= 0.0)
    {
        // return a new smart pointer with references to smart pointers to these values
        ret = shared_from_this();
    }
    else
    {
        EqObjPtr unscaled = this->getUnscaledValue();
        if (unscaled->getType() == CONST_OBJ)
        {
            Constant *c = static_cast<Constant *>(unscaled.get());
            // We expect the value to be signed
                if (c->dvalue < 0.0)
                {
                    ret = con(-c->getDoubleValue());
                }
                else
                {
                    ret = unscaled->clone();
                }
        }
        else // This must be a product
        {
            EqObjPtr scaled = this->getScale();
            Constant *s = static_cast<Constant *>(scaled.get());
            const double val = s->getDoubleValue();

            EqObjPtr negval;
                negval = con(-(s->dvalue));


            if (val == -1.0)
            {
                ret = unscaled;
            }
            else
            {
                // Want to prevent something like -8*(a*x^2)
//              ret = con(-val) * unscaled;
                if (unscaled->getType() == PRODUCT_OBJ)
                {
                    Product *Y = static_cast<Product *>(unscaled.get());
                    typedef std::vector<EqObjPtr> EqVec;
                    EqVec &uvals = Y->values;
                    EqVec newvals;
                    newvals.reserve(uvals.size()+1);
                    newvals.push_back(negval);
                    for (size_t i = 0; i < uvals.size(); ++i)
                    {
                        // Is the clone necessary, well just in case
                        newvals.push_back(uvals[i]->clone());
                    }
                    EquationObject *eq = new Product(newvals);
                    ret = EqObjPtr(eq);
                }
                else
                {
                    ret = negval * unscaled;
                }
            }
        }
    }

    return ret;
}

EqObjPtr Product::clone()
{
    const size_t len = values.size();
    std::vector<EqObjPtr> tmp(len);
    for (size_t i = 0; i < len; ++i)
    {
        tmp[i] = values[i]->clone();
    }
    EquationObject *n = new Product(tmp);
    return EqObjPtr(n);
}

EqObjPtr Product::subst(const std::string &str, EqObjPtr eqo)
{
    if (str == this->stringValue())
        return eqo;

    //// TODO: back port this to the original adiff
    else if (str == this->getUnscaledValue()->stringValue())
    {
      return this->getScale() * eqo;
    }
    else
    {
        std::vector<EqObjPtr> new_values(values.size());
        for (size_t i=0; i < values.size(); ++i)
        {
            new_values[i] = values[i]->subst(str, eqo);
        }
        return EqObjPtr(new Product(new_values));
    }
}

// expand out all products with additions
EqObjPtr Product::expand()
{
    const size_t len = values.size();
    std::vector<EqObjPtr> tmp;
    tmp.reserve(len);
    std::vector<EqObjPtr> adds;
    adds.reserve(len);
    for (size_t i = 0; i < len; ++i)
    {
        if (values[i]->getType() == ADD_OBJ)
            adds.push_back(values[i]->expand());
        else
            tmp.push_back(values[i]->expand());
            
    }
    EqObjPtr scale = con(1);
    if (tmp.size() == 1)
    {
        scale = tmp[0];
    }
    else if (tmp.size() > 1)
    {
        EquationObject *n = new Product(tmp); 
        scale = EqObjPtr(n);
    }

    EqObjPtr out;
    if (adds.empty())
    {
        out = scale;
    }
    else
    {
        std::vector<EqObjPtr> pout;
        // break out into separate function
        // do the expansion here
        // pout is the result of the multiplication of scale
        // with each object within an add
        for (size_t i = 0; i < adds.size(); ++i)
        {
            // use static cast since we know this is an add
            Add *Y = static_cast<Add *>(adds[i].get());
            // get reference
            std::vector<EqObjPtr> &v = Y->values;
            if (scale->isOne())
            {
                pout = v; // copy
            }
            else
            {
                for (size_t j = 0; j < v.size(); ++j)
                {
                    pout.push_back(scale * v[j]);
                }
            }
            EquationObject *foo = new Add(pout);
            scale = EqObjPtr(foo);
            pout.clear(); // reuse for the next pass
        }
        out = scale;
    }

    return out;
}

EqObjPtr Product::getReciprocal()
{
  assert(false);
  return con(0);
}

}
