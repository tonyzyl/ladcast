/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef POW_HH
#define POW_HH
#include "EquationObject.hh"
namespace Eqo {
class Pow : public EquationObject {
    public:
        Pow(EqObjPtr, EqObjPtr);

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

        bool hasReciprocal() {return false;}
        EqObjPtr getReciprocal();
        std::set<std::string> getReferencedType(Eqo::EqObjType rt)
        {
            return UniteReferencedType(rt, base, exponent);
        }

        friend class Log;
        friend class Product;

        std::vector<EqObjPtr> getArgs() {
          return createArgs(base, exponent);
        }

        std::string getName() const
        {
          return "pow";
        }
    private:
        std::string createStringValue() const;

        Pow(const Pow &);
        Pow operator=(const Pow &);

        const EqObjPtr base;
        const EqObjPtr exponent;
};

}
#endif

