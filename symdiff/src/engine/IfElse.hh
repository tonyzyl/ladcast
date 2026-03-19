/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef IF_ELSE_HH
#define IF_ELSE_HH
#include "EquationObject.hh"

namespace Eqo {
class IfObj : public EquationObject
{
    public:
        IfObj(EqObjPtr /*test*/, EqObjPtr /*result*/);
        ~IfObj() {};

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
          return createArgs(test, arg);
        }

        bool hasReciprocal() {return false;}
        EqObjPtr getReciprocal();

        std::set<std::string> getReferencedType(Eqo::EqObjType rt);

        std::string getName() const
        {
          return "if";
        }

    private:
        std::string createStringValue() const;

        IfObj(const IfObj &);
        IfObj operator=(const IfObj &);

        const EqObjPtr test;
        const EqObjPtr arg;
};

class IfElseObj : public EquationObject
{
    public:
        IfElseObj(EqObjPtr /*test*/, EqObjPtr /*left*/, EqObjPtr /*right*/);
        ~IfElseObj() {};

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
          return createArgs(test, arg1, arg2);
        }

        bool hasReciprocal() {return false;}
        EqObjPtr getReciprocal();

        std::set<std::string> getReferencedType(Eqo::EqObjType rt);

        std::string getName() const
        {
          return "ifelse";
        }
    private:
        std::string createStringValue() const;

        IfElseObj(const IfElseObj &);
        IfElseObj operator=(const IfElseObj &);

        EqObjPtr test;
        EqObjPtr arg1;
        EqObjPtr arg2;
};
}

#endif
