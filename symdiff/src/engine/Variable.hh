/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef VARIABLE_HH
#define VARIABLE_HH
#include "EquationObject.hh"

namespace Eqo {
class Variable : public EquationObject {
    public:
        const std::string &getVariableName() const {
          return value;
        }

        std::string getName() const
        {
          return getVariableName();
        }

        Variable(std::string);
        ~Variable() {};
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

        std::vector<EqObjPtr> getArgs() {
          return std::vector<EqObjPtr>();
        }

        bool hasReciprocal() {return false;}
        EqObjPtr getReciprocal();
        std::set<std::string> getReferencedType(Eqo::EqObjType rt) {
            return CreateSetIfThisType(rt);
        }

    private:
        std::string createStringValue() const;

        Variable(const Variable &);
        Variable operator=(const Variable &);

        const std::string value;
};
}
#endif

