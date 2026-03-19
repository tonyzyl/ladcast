/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "SubExpr.hh"
#include "EquationObject.hh"
#include "mcModel.hh"
#include "Functions.hh"

#include <iostream>
#include <sstream>

const std::string &SubExprData::getName(SubExpr &subexpr) const {
  if (name.empty())
  {
    name = subexpr.GetUniqueName();
  }
  return name;
}

SubExprData::SubExprData(Eqo::EqObjPtr x)
{
  refcount = 1;
  equation = x->clone();
}

Eqo::EqObjPtr SubExprData::getEquation() const
{
  return equation->clone();
}

SubExpr::SubExpr() : unique_counter_(0)
{
}

std::string SubExpr::GetUniqueName()
{
  std::ostringstream os;
  os << "unique" << unique_counter_++;
  return os.str();
}


/// Recursively scan subexpressions and add to map
/// only do this on unscaled values
void SubExpr::scanSubexpression(Eqo::EqObjPtr &eq)
{
  Eqo::EqObjPtr te = eq->getUnscaledValue();
  Eqo::EqObjType t = te->getType();
  //// These are too simple to cache
  if (t == Eqo::MODEL_OBJ)
  {
    const std::string &model_name = te->stringValue();
    const ProcessStatus_t status  = processStatusMap_[model_name];
    if (status == PROCESSING)
    {
      errorString_ += model_name + " being processed in terms of itself\n";
    }
    return;
  }
  else if (t == Eqo::VARIABLE_OBJ || t == Eqo::CONST_OBJ)
  {
    return;
  }

  const std::string &eqstring = te->stringValue();

  std::map<std::string, SubExprData>::iterator it = SubExprMap.find(eqstring);
  if (it != SubExprMap.end())
  {
    (it->second).addCount();
  }
  else
  {
    SubExprMap.insert(std::make_pair(eqstring, SubExprData(te)));
    SubExprOrdered.push_back(eqstring);
  }
  //// Since this is the first time, we will look for temporaries which may be shared as well
  //// If they are truly unique, they will only be visited once since the parent will only be descended into once
  //// Keep going  TODO: see what we are doing in devsim
  std::vector<Eqo::EqObjPtr> vec = te->getArgs();
  for (size_t i = 0; i < vec.size(); ++i)
  {
    scanSubexpression(vec[i]);
  }
}

bool SubExpr::removeZeros(ModelMap_t &ModelMap)
{
  bool done = true;

  ModelNameVector_t zeros;
  zeros.reserve(ModelMap.size());

  for (ModelMap_t::const_iterator it = ModelMap.begin(); it != ModelMap.end(); ++it)
  {
    Eqo::EqObjPtr x = it->second;
    if (x)
    {
      if (x->isZero())
      {
        zeros.push_back(it->first);
      }
    }
  }

  for (ModelNameVector_t::const_iterator it = zeros.begin(); it != zeros.end(); ++it)
  {
    ModelMap.erase(*it);
  }

  for (ModelMap_t::iterator it = ModelMap.begin(); it != ModelMap.end(); ++it)
  {
//    const std::string &name = it->first;
    Eqo::EqObjPtr x = it->second;
    if (x)
    {
      const std::set<std::string> &reftype = x->getReferencedType(Eqo::MODEL_OBJ);
      for (ModelNameVector_t::const_iterator jt = zeros.begin(); jt != zeros.end(); ++jt)
      {
        if (reftype.count(*jt))
        {
          x = x->subst(*jt, Eqo::con(0.0));
        }
      }
      x = x->Simplify();
      if ((it->second)->stringValue() != x->stringValue())
      {
        done = false;
        it->second = x;
      }
    }
  }
  return done;
}

void SubExpr::RemoveZeros(ModelMap_t &ModelMap)
{
  size_t mlsize;
  do
  {
    mlsize = ModelMap.size();
    removeZeros(ModelMap);
  }
  while (mlsize != ModelMap.size());
}

void SubExpr::CreateSubexpressions(ModelMap_t &ModelMap)
{
  size_t mlsize;
  do
  {
    mlsize = ModelMap.size();
    createSubexpressions(ModelMap);
  }
  while (mlsize != ModelMap.size());
}

void SubExpr::createSubexpressions(ModelMap_t &ModelMap)
{
  SubExprMap.clear();
  processStatusMap_.clear();

   /*** Start with whole model list, assume all derivatives created ***/
  for (ModelMap_t::iterator semit = ModelMap.begin(); semit != ModelMap.end(); ++semit)
  {
    const std::string &model_name = semit->first;

    const ProcessStatus_t status = processStatusMap_[model_name];

    if (status == PROCESSING)
    {
      errorString_ += model_name + " being processed in terms of itself\n";
      processStatusMap_[model_name] = DONE;
    }
    else if (status == UNTOUCHED)
    {
      /// what if two models have the same definition?
      /// Implicity handle case where model has no definition
      if (semit->second)
      {
        scanSubexpression(semit->second);
      }
      processStatusMap_[model_name] = DONE;
    }
    else if (status == DONE)
    {
    }
  }

  /*** now scan, and if ref count is greater than 1, then create unique name and substitute ***/
  std::vector<std::string> final_list;
  final_list.reserve(SubExprOrdered.size());
  for (std::vector<std::string>::iterator it = SubExprOrdered.begin(); it != SubExprOrdered.end(); ++it)
  {
    //// This is unique so should not be temporary
    std::map<std::string, SubExprData>::iterator jt = SubExprMap.find(*it);
    if (jt != SubExprMap.end())
    {
//      std::cerr << "Ref count " << *it << " " << (jt->second).getCount() << "\n";
      if ((jt->second).getCount() > 1)
      {
        final_list.push_back(*it);
      }
    }
  }

  //// Create temporaries
  for (size_t i = 0; i < final_list.size(); ++i)
  {
    const SubExprData &subexpr = SubExprMap[final_list[i]];
    ModelMap.insert(std::make_pair(subexpr.getName(*this), subexpr.getEquation()));
  }

  for (ModelMap_t::iterator semit = ModelMap.begin(); semit != ModelMap.end(); ++semit)
  {
    for (size_t i = 0; i < final_list.size(); ++i)
    {
      const SubExprData &subexpr = SubExprMap[final_list[i]];
      /// make sure we don't substitute something for itself
      if (subexpr.getName(*this) != semit->first)
      {
        if (semit->second)
        {
          semit->second = (semit->second)->subst(final_list[i], Eqo::mod(subexpr.getName(*this)));
        }
      }
    }
  }
}

