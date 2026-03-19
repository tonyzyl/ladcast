/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef ENGINEAPI_HH
#define ENGINEAPI_HH
#include <memory>
#include "SymdiffImport.hh"
#include <string>
#include <vector>
#include <set>
namespace Eqo {
class EquationObject;
typedef std::shared_ptr<EquationObject> EqObjPtr;
}

namespace EngineAPI {
enum EqObjType {CONST_OBJ=0, VARIABLE_OBJ, ADD_OBJ, PRODUCT_OBJ, EXPONENT_OBJ, POW_OBJ, LOG_OBJ, MODEL_OBJ, USERFUNC_OBJ, BLOGICAL_OBJ, ULOGICAL_OBJ, IF_OBJ, IFELSE_OBJ, NUM_TYPES};

typedef Eqo::EqObjPtr (*EqUnaryFuncPtr) (Eqo::EqObjPtr);
typedef Eqo::EqObjPtr (*EqBinaryFuncPtr) (Eqo::EqObjPtr, Eqo::EqObjPtr);

Eqo::EqObjPtr DLL_PROTECTED createUnaryLogical(const std::string &/*unaryOp*/, Eqo::EqObjPtr /*arg*/);
Eqo::EqObjPtr DLL_PROTECTED createBinaryLogical(const std::string &/*unaryOp*/, Eqo::EqObjPtr /*arg*/, Eqo::EqObjPtr /*arg*/);
Eqo::EqObjPtr DLL_PROTECTED createIfObj(Eqo::EqObjPtr /*test*/, Eqo::EqObjPtr /*result*/);
Eqo::EqObjPtr DLL_PROTECTED createIfElseObj(Eqo::EqObjPtr /*test*/, Eqo::EqObjPtr /*left*/, Eqo::EqObjPtr /*right*/);
/// Non member functions for intuitive math operations
Eqo::EqObjPtr DLL_PROTECTED operator* (Eqo::EqObjPtr x, Eqo::EqObjPtr y);
Eqo::EqObjPtr DLL_PROTECTED pow (Eqo::EqObjPtr x, Eqo::EqObjPtr y);
Eqo::EqObjPtr DLL_PROTECTED operator/ (Eqo::EqObjPtr x, Eqo::EqObjPtr y);
Eqo::EqObjPtr DLL_PROTECTED operator+ (Eqo::EqObjPtr x, Eqo::EqObjPtr y);
Eqo::EqObjPtr DLL_PROTECTED operator- (Eqo::EqObjPtr x, Eqo::EqObjPtr y);

Eqo::EqObjPtr DLL_PROTECTED exp (Eqo::EqObjPtr x);

Eqo::EqObjPtr DLL_PROTECTED pow (Eqo::EqObjPtr x, Eqo::EqObjPtr y);

Eqo::EqObjPtr DLL_PROTECTED log (Eqo::EqObjPtr x);

Eqo::EqObjPtr DLL_PROTECTED diff(Eqo::EqObjPtr x, Eqo::EqObjPtr y);

Eqo::EqObjPtr DLL_PROTECTED con(double x);

Eqo::EqObjPtr DLL_PROTECTED mod(const char *x);

Eqo::EqObjPtr DLL_PROTECTED mod(const std::string &x);

Eqo::EqObjPtr DLL_PROTECTED var(const char *x);

Eqo::EqObjPtr DLL_PROTECTED var(const std::string &x);

Eqo::EqObjPtr DLL_PROTECTED sqrt(Eqo::EqObjPtr x);

Eqo::EqObjPtr DLL_PROTECTED reciprocal_sqrt(Eqo::EqObjPtr x);

/// how we print things
std::ostream & operator<< (std::ostream &os, Eqo::EqObjPtr foo);

/// keeps simplifying expression until string value doesn't change
Eqo::EqObjPtr DLL_PROTECTED Simplify(Eqo::EqObjPtr x);

Eqo::EqObjPtr DLL_PROTECTED getConstantFactor(Eqo::EqObjPtr x);

Eqo::EqObjPtr DLL_PROTECTED Expand(Eqo::EqObjPtr x);

Eqo::EqObjPtr DLL_PROTECTED getUnscaledValue(Eqo::EqObjPtr x);

Eqo::EqObjPtr DLL_PROTECTED getUnsignedValue(Eqo::EqObjPtr x);

Eqo::EqObjPtr DLL_PROTECTED getSign(Eqo::EqObjPtr x);

/// Sorts in order of constants, variables, then other objects
void SortEqVector(std::vector<Eqo::EqObjPtr> &v1);

Eqo::EqObjPtr DLL_PROTECTED subst(Eqo::EqObjPtr a, Eqo::EqObjPtr b, Eqo::EqObjPtr c);

Eqo::EqObjPtr DLL_PROTECTED getNegation(Eqo::EqObjPtr x);

bool DLL_PROTECTED isBinaryFunction(const std::string &str);
bool DLL_PROTECTED isUnaryFunction(const std::string &str);

EqBinaryFuncPtr DLL_PROTECTED getBinaryFuncPtr(const std::string &);
EqUnaryFuncPtr DLL_PROTECTED getUnaryFuncPtr(const std::string &);

std::string DLL_PROTECTED getName(Eqo::EqObjPtr x);
std::string DLL_PROTECTED getType(Eqo::EqObjPtr x);
std::set<std::string> DLL_PROTECTED getReferencedType(Eqo::EqObjPtr x, EqObjType t);
double DLL_PROTECTED getDoubleValue(Eqo::EqObjPtr x);
std::string DLL_PROTECTED getStringValue(Eqo::EqObjPtr x);
EqObjType DLL_PROTECTED getEnumeratedType(Eqo::EqObjPtr x);
std::vector<Eqo::EqObjPtr> DLL_PROTECTED getArgs(Eqo::EqObjPtr x);

typedef bool (*inModelListCallback_ptr)(const std::string &);
void DLL_PROTECTED SetModelListCallBack(inModelListCallback_ptr);

typedef Eqo::EqObjPtr (*modelDerivativeRule_ptr)(Eqo::EqObjPtr, Eqo::EqObjPtr);
void DLL_PROTECTED SetDerivativeRule(modelDerivativeRule_ptr);

void DLL_PROTECTED ResetAllData();
}
#endif
