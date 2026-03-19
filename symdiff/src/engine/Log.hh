/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef LOG_HH
#define LOG_HH
#include "EquationObject.hh"
namespace Eqo {
class Log : public EquationObject {
    public:
        Log(EqObjPtr);

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

        friend class Exponent;

        std::vector<EqObjPtr> getArgs() {
          return createArgs(value);
        }

        std::string getName() const
        {
          return "log";
        }

    private:
        std::string createStringValue() const;

        Log(const Log &);
        Log operator=(const Log &);

        const EqObjPtr value;
};
}
#endif

