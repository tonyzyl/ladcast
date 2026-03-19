/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef ADD_HH
#define ADD_HH
#include "EquationObject.hh"
namespace Eqo {
class Add : public EquationObject {
    public:
        Add(EqObjPtr, EqObjPtr);
        Add(std::vector<EqObjPtr>);

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
            return UniteReferencedType(rt, values);
        }

        friend class Product; // expansion of (a + ...)*(b + ...)

        std::vector<EqObjPtr> getArgs() {
          return createArgs(values);
        }

        std::string getName() const
        {
          return "+";
        }
    private:
        std::string createStringValue() const;

        Add(const Add &);
        Add operator=(const Add &);

        std::vector<EqObjPtr> values;
};
}
#endif
