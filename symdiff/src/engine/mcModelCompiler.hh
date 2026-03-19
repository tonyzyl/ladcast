/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef MC_MODEL_COMPILER_HH
#define MC_MODEL_COMPILER_HH
#include "EquationObject.hh"
#include <string>
Eqo::EqObjPtr ClearModel(const std::string &);

Eqo::EqObjPtr DeclareModel(const std::string &);

Eqo::EqObjPtr GetModelValue(const std::string &);

Eqo::EqObjPtr DefineModel(const std::string &, Eqo::EqObjPtr);
#endif

