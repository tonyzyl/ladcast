/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "ProcessModelOrder.hh"
#include "EquationObject.hh"
#include <set>

const ModelNameVector_t &ProcessModelOrder::GetModelVector(const ModelMap_t &model_list, const ModelNameVector_t &input_list)
{
  processedList_.clear();
  processedList_.reserve(input_list.size());
  processStatusMap_.clear();
  for (ModelNameVector_t::const_iterator it = input_list.begin(); it != input_list.end(); ++it)
  {
    const std::string &model_name = *it;
    const ProcessStatus_t status  = processStatusMap_[model_name];

    ModelMap_t::const_iterator mit = model_list.find(model_name);

    if (status == DONE)
    {
    }
    else if (mit == model_list.end())
    {
      errorString_ += model_name + " was not declared or defined as a model\n";

    }
    else if (!(mit->second))
    {
      errorString_ += model_name + " was not defined as a model\n";
    }
    //// impossible for this model to be in processing
    else if (status == UNTOUCHED)
    {

      processModelVector(model_list, mit->second);
    }

    processStatusMap_[model_name] = DONE;
    processedList_.push_back(model_name);
  }

  return processedList_;
}

/*
  Only print equations
  Called recursively.  Refactor, if possible, using iteration.
*/
void ProcessModelOrder::processModelVector(const ModelMap_t &model_list, Eqo::EqObjPtr eq)
{
  std::set<std::string> mset;

  const Eqo::EqObjType equation_type = eq->getType();
  if (equation_type == Eqo::MODEL_OBJ)
  {
    const std::string &mname = eq->stringValue();
    const ProcessStatus_t status = processStatusMap_[mname];
    if (status == DONE)
    {
       return;
    }
    else if (status == PROCESSING)
    {
      errorString_ += mname + " is being processed in terms of itself\n";
      processStatusMap_[mname] = DONE;
      return;
    }
    else if (status == UNTOUCHED)
    {
      processStatusMap_[mname] = PROCESSING;
    }

    ModelMap_t::const_iterator mit = model_list.find(mname);
    if (mit != model_list.end())
    {
      if (mit->second)
      {
        mset = (mit->second)->getReferencedType(Eqo::MODEL_OBJ);
      }
    }

  }
  else
  {
     //// Get all models referenced by this object
     mset = eq->getReferencedType(Eqo::MODEL_OBJ);
  }

  std::set<std::string>::iterator it =mset.begin();
  std::set<std::string>::iterator end=mset.end();
  for ( ; it != end; ++it)
  {
    const std::string &sname = *it;
    // Model was processed, we are done
    if (processStatusMap_[sname] == DONE)
    {
       continue;
    }

    ModelMap_t::const_iterator mit = model_list.find(sname);
    if (mit != model_list.end())
    {
      //// print out all models referenced by this equation
      Eqo::EqObjPtr x = mit->second;
      if (x)
      {
        processModelVector(model_list, x);
      }
      processedList_.push_back(sname);
      processStatusMap_[sname] = DONE;

    }
  }

  //// If we are a model, go ahead and print out our equation
  if (equation_type == Eqo::MODEL_OBJ)
  {
    const std::string &mname = eq->stringValue();
    processedList_.push_back(mname);
    processStatusMap_[mname] = DONE;
  }
}

