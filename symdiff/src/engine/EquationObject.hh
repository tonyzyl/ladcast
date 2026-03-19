/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef EQUATIONOBJECT_H
#define EQUATIONOBJECT_H

#include <memory>
#include "SymdiffImport.hh"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>

namespace Eqo {
class EquationObject;
typedef std::shared_ptr<EquationObject> EqObjPtr;
/**
 Enumerated type quickly tells what type the derived class is.  May
 be faster than using dynamic_cast.  This is passed in the constructor
 of the derived class to the base class.
*/
enum EqObjType {CONST_OBJ=0, VARIABLE_OBJ, ADD_OBJ, PRODUCT_OBJ, EXPONENT_OBJ, POW_OBJ, LOG_OBJ, MODEL_OBJ, USERFUNC_OBJ, BLOGICAL_OBJ, ULOGICAL_OBJ, IF_OBJ, IFELSE_OBJ, NUM_TYPES};

const char * const EqObjNames[] = {
    "constant",
    "variable",
    "add",
    "product",
    "exp",
    "pow",
    "log",
    "model",
    "userfunc",
    "blogical",
    "ulogical",
    "if",
    "ifelse"
};

typedef std::shared_ptr<EquationObject> EqObjPtr;
std::set<std::string> UniteReferencedType(EqObjType, const std::vector<EqObjPtr> &x);
std::set<std::string> UniteReferencedType(EqObjType, const EqObjPtr &op1, const EqObjPtr &op2);

/**
   This is the abstract base class which provides the interface for all objects.
*/
class EquationObject : public std::enable_shared_from_this<EquationObject>
{
   public:
      typedef std::shared_ptr<EquationObject> EqObjPtr;

      virtual std::vector<EqObjPtr> getArgs() = 0;

      EqObjType getType() {return eqtype;}
      virtual ~EquationObject() {};

      virtual EqObjPtr Derivative(EqObjPtr)=0;
      virtual EqObjPtr Simplify()=0;

      const std::string &stringValue() const;
      /// Right now for name only
      virtual std::string getName() const = 0;

      virtual std::set<std::string> getReferencedType(Eqo::EqObjType)=0;

      // Both of these should be for std::vector<EqObjPtr>
      // Each EquationObject should know how to combine itself
      // with others of the same type
      // to do, create default method for simple types
      virtual EqObjPtr CombineProduct(std::vector<EqObjPtr>) = 0;
      virtual EqObjPtr CombineAdd(std::vector<EqObjPtr>) = 0;

      virtual bool isZero()=0;
      virtual bool isOne()=0;

      virtual EqObjPtr getScale()=0;
      virtual EqObjPtr getUnscaledValue()=0;

      virtual double getSign() = 0;
      virtual EqObjPtr getUnsignedValue() = 0;

      // creates a clone copy of original structure recursively
      // note that constant and variables do not need to clone
      // their values since they are simple data
      virtual EqObjPtr clone() = 0;

      // Substitute the first expression for the
      // non virtual since I want to check each for their string value
      // \todo make non virtual later
      // probably won't work for anything very well, except for strings.
      // If substitution is successful, it always returns a clone to the substitution
      //
      // Note that even though this changes the contents of the equation object, you
      // must always set the result equal to the return value, in case the top object
      // is replaced
// Really need to reconsider subst interface to not be string
      virtual EqObjPtr subst(const std::string &, EqObjPtr) = 0;

      // Expand out all products with additions
      virtual EqObjPtr expand() = 0;

      // csc = 1/sin
      virtual bool hasReciprocal() = 0;
      virtual EqObjPtr getReciprocal() = 0;

      /// TODO: for more powerful user defined functions
      /// hasInverse
      /// hasAnnhilater
      /// hasCombiner
   protected:
      virtual std::string createStringValue() const=0;

      EquationObject(EqObjType foo) : eqtype(foo) {}

      std::vector<EqObjPtr> createArgs(EqObjPtr x) {
        std::vector<EqObjPtr> out;
        out.push_back(x);
        return out;
      }

      std::set<std::string> CreateSetIfThisType(EqObjType rt)
      {
        std::set<std::string> ret;
        if (eqtype == rt)
        {
            ret.insert(this->stringValue());
        }
        return ret;
      }

      std::vector<EqObjPtr> createArgs(EqObjPtr x, EqObjPtr y) {
        std::vector<EqObjPtr> out;
        out.reserve(2);
        out.push_back(x);
        out.push_back(y);
        return out;
      }

      std::vector<EqObjPtr> createArgs(EqObjPtr x, EqObjPtr y, EqObjPtr z) {
        std::vector<EqObjPtr> out;
        out.reserve(3);
        out.push_back(x);
        out.push_back(y);
        out.push_back(z);
        return out;
      }

      std::vector<EqObjPtr> createArgs(std::vector<EqObjPtr> &x) {
        return x;
      }

   private:
      EquationObject(const EquationObject &);
      //Cannot return abstract class
      //EquationObject operator=(const EquationObject &);

      EqObjType   eqtype;
      mutable std::string stringValue_;
};





/**
 * The following is to define macro variables
 */
extern DLL_PROTECTED std::map<std::string, EqObjPtr> variableMap;



/**
 * The following entries are to create tables of available
 * commands for my parser
 */
typedef EqObjPtr (*EqUnaryFuncPtr) (EqObjPtr);
typedef EqObjPtr (*EqBinaryFuncPtr) (EqObjPtr, EqObjPtr);

struct BinaryTblEntry {
    const char *name;
    EqBinaryFuncPtr func;
    const char *desc;
};

struct UnaryTblEntry {
    const char *name;
    EqUnaryFuncPtr func;
    const char *desc;
};

extern UnaryTblEntry UnaryTable[];
extern BinaryTblEntry BinaryTable[];

EqBinaryFuncPtr getBinaryFuncPtr(std::string);
EqUnaryFuncPtr getUnaryFuncPtr(std::string);

/// Print help
EqObjPtr printHelp();

/// how we print things
inline std::ostream & operator<< (std::ostream &os, EqObjPtr foo)
{
   os << foo->stringValue();
   return os;
}


}/* end namespace Eqo */

#endif

