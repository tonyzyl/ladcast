/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef MC_CONTEXT_HH
#define MC_CONTEXT_HH
#include <map>
#include <string>
#include <memory>
#include "SymdiffImport.hh"

namespace Eqo
{
class EquationObject;
typedef std::shared_ptr<EquationObject> EqObjPtr;
}

typedef std::map<std::string, Eqo::EqObjPtr > ModelMap_t;
typedef bool (*inModelListCallback_ptr)(const std::string &);
typedef Eqo::EqObjPtr (*modelDerivativeRule_ptr)(Eqo::EqObjPtr, Eqo::EqObjPtr);

namespace Eqo
{
extern Eqo::EqObjPtr DLL_PROTECTED default_derivative_rule(Eqo::EqObjPtr, Eqo::EqObjPtr);
}

//// Be able to store rules in the future.
class ContextData {
  public:
    ContextData() : callback_func_ptr_(nullptr), model_derivative_rule_ptr_(Eqo::default_derivative_rule)
    {
    }

    ModelMap_t model_list_;

    inModelListCallback_ptr callback_func_ptr_;
    modelDerivativeRule_ptr model_derivative_rule_ptr_;
  private:
//    ContextData();
//    ~ContextData();
    ContextData(const ContextData &);
    const ContextData &operator=(const ContextData &);
};

class DLL_PROTECTED Context {
  public:
    static Context &GetInstance();
    static void DestroyInstance();

    Eqo::EqObjPtr EvaluateModelDerivative(Eqo::EqObjPtr, Eqo::EqObjPtr);
    Eqo::EqObjPtr FindInModelList(const std::string &);
    bool          IsInModelList(const std::string &);
    void          SetModelListCallBack(inModelListCallback_ptr);
    void          SetDerivativeRule(modelDerivativeRule_ptr);
    ModelMap_t  &GetModelMap();

    void          ClearAllModels();
    void          ClearModel(const std::string &);
    void          DeclareModel(const std::string &);
    void          DefineModel(const std::string &, Eqo::EqObjPtr);
    
  private:
    Context();
    ~Context();
    Context(const Context &);
    const Context &operator=(const Context &);

    ContextData context_data_;

    static Context *instance_;
};
#endif

