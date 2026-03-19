/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef EXPONENT_HH
#define EXPONENT_HH
#include "EquationObject.hh"
namespace Eqo {
class Exponent : public EquationObject {
    public:
        Exponent(EqObjPtr);

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
            return value->getReferencedType(rt);
        }

        std::vector<EqObjPtr> getArgs() {
          return createArgs(value);
        }

        std::string getName() const
        {
          return "exp";
        }

        friend class Log;
    private:
        std::string createStringValue() const;

        Exponent(const Exponent &);
        Exponent operator=(const Exponent &);

        const EqObjPtr value;
};

}
#endif

