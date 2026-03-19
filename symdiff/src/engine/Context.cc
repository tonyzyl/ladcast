/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/
#include "Context.hh"

Context *Context::instance_;

Context::Context()
{
}

Context::~Context()
{
}

ModelMap_t &Context::GetModelMap()
{
  return context_data_.model_list_;
}

Context &Context::GetInstance()
{
  if (!instance_)
  {
    instance_ = new Context();
  }
  return *instance_;
}

void Context::DestroyInstance()
{
  delete instance_;
  instance_ = nullptr;
}


void Context::SetModelListCallBack(inModelListCallback_ptr foo)
{
    context_data_.callback_func_ptr_ = foo;
}

/*
   Returns true if the model by the given name is in the ModelList
 */
bool Context::IsInModelList(const std::string &str)
{
  if (context_data_.callback_func_ptr_)
  {
    return context_data_.callback_func_ptr_(str);
  }
  else
  {
     ModelMap_t::iterator it, end=context_data_.model_list_.end();
     for (it=context_data_.model_list_.begin(); it != end; ++it)
     {
        if ((*it).first == str)
           return true;
     }
  }
 return false;
}

/*
   If it is in the model list, this will return a pointer to the model with the given
   name.
 */
Eqo::EqObjPtr Context::FindInModelList(const std::string &str)
{
  Eqo::EqObjPtr ret;
  ModelMap_t::iterator it = context_data_.model_list_.find(str);
  if (it != context_data_.model_list_.end())
  {
    ret = it->second;
  }

  return ret;
}

void Context::ClearAllModels()
{
  context_data_.model_list_.clear();
}

void Context::ClearModel(const std::string &str)
{
  ModelMap_t::iterator it = context_data_.model_list_.find(str);
  if (it != context_data_.model_list_.end())
  {
    context_data_.model_list_.erase(it);
  }
}

void Context::DeclareModel(const std::string &s)
{
  if (!FindInModelList(s))
  {
//#warning "decide if we should zero out an existing model"
    context_data_.model_list_.insert(std::make_pair(s, Eqo::EqObjPtr()));
  }
}

void Context::DefineModel(const std::string &s, Eqo::EqObjPtr e)
{
  GetModelMap()[s] = e;
}

Eqo::EqObjPtr Context::EvaluateModelDerivative(Eqo::EqObjPtr x, Eqo::EqObjPtr y)
{
  return context_data_.model_derivative_rule_ptr_(x, y);
}

void Context::SetDerivativeRule(modelDerivativeRule_ptr x)
{
  context_data_.model_derivative_rule_ptr_ = x;
}


