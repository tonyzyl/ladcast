/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef CONSTANT_HH
#define CONSTANT_HH
#include "EquationObject.hh"
namespace Eqo {
class Constant : public EquationObject {
    public:
        Constant(double);
        ~Constant() {};


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

        bool hasReciprocal()
        {
          return !(this->isZero());
        }

        EqObjPtr getReciprocal();

        double getDoubleValue()
        {
          return dvalue;
        }

        std::set<std::string> getReferencedType(Eqo::EqObjType rt) {
            return CreateSetIfThisType(rt);
        }

        friend class Pow;
        friend class Product;

        std::vector<EqObjPtr> getArgs() {
          return std::vector<EqObjPtr>();
        }

        std::string getName() const
        {
          return EquationObject::stringValue();
        }
    private:
        std::string createStringValue() const;

        Constant(const Constant &);
        Constant operator=(const Constant &);

        const double dvalue;
};
}
#endif

