/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "Functions.hh"

namespace Eqo
{
UnaryTblEntry UnaryTable[] = {
    {"simplify",    Simplify,         "simplify(obj)         -- Simplify as much as possible"},
    {"exp",         exp,              "exp(obj)              -- Exponentiation with respect to e"},
    {"expand",        Expand,          "expand(obj)          -- Perform expansion of products with sums"},
    {"log",         log,              "log(obj)              -- Natural logarithm"},
    {"unscaledval", getUnscaledValue, "unscaledval(obj)      -- Get value without constant scaling"},
    {"scale",       getConstantFactor,"scale(obj)            -- Get constant factor"},
    {"unsignedval", getUnsignedValue, "unsignedval(obj)      -- Get unsigned value"},
    {"sign",        getSign,          "sign(obj)             -- Get sign as 1 or -1"},
    {nullptr, nullptr, nullptr}
};

BinaryTblEntry BinaryTable[] = {
    {"pow",  pow,  "pow(obj1, obj2)       -- Raise obj1 to the power of obj2"},
    {"diff", diff, "diff(obj1, obj2)      -- Take derivative  of obj1 with respect to obj2"},
    {nullptr, nullptr, nullptr}
};

EqUnaryFuncPtr getUnaryFuncPtr(std::string x) {
    size_t i=0; 
    while (UnaryTable[i].name != nullptr)
    {
        if (x == UnaryTable[i].name)
            return UnaryTable[i].func;
        ++i;
    }
    return nullptr;
}

EqBinaryFuncPtr getBinaryFuncPtr(std::string x) {
    size_t i=0; 
    while (BinaryTable[i].name != nullptr)
    {
        if (x == BinaryTable[i].name)
            return BinaryTable[i].func;
        ++i;
    }
    return nullptr;
}
}

