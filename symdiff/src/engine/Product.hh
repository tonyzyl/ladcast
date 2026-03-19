/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef PRODUCT_HH
#define PRODUCT_HH
#include "EquationObject.hh"
namespace Eqo {
class Product : public EquationObject {
    public:
        friend class Pow;
        Product(EqObjPtr, EqObjPtr);
        Product(std::vector<EqObjPtr>);

        EqObjPtr Derivative(EqObjPtr);
        EqObjPtr Simplify();

        EqObjPtr CombineProduct(std::vector<EqObjPtr>);
        EqObjPtr CombineAdd(std::vector<EqObjPtr>);

        bool isZero();
        bool isOne();

        EqObjPtr getScale();
        EqObjPtr getUnscaledValue();

        double getSign();
        EqObjPtr getUnsignedValue();

        EqObjPtr clone();
        EqObjPtr subst(const std::string &, EqObjPtr);

        EqObjPtr expand();

        void ProductVecSimplify(std::vector<EqObjPtr> &);

        bool hasReciprocal() {return false;}
        EqObjPtr getReciprocal();

        std::set<std::string> getReferencedType(Eqo::EqObjType rt)
        {
            return UniteReferencedType(rt, values);
        }

        std::vector<EqObjPtr> getArgs() {
          return createArgs(values);
        }

        std::string getName() const
        {
          return "*";
        }

    private:
        std::string createStringValue() const;

        Product(const Product &);
        Product operator=(const Product &);

        std::vector<EqObjPtr> values;
};
}
#endif
