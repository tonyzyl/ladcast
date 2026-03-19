/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef SUBEXPR_HH
#define SUBEXPR_HH
#include <memory>
#include "SymdiffImport.hh"

#include <string>
#include <map>
#include <vector>



namespace Eqo
{
class EquationObject;
typedef std::shared_ptr<EquationObject> EqObjPtr;
}

typedef std::vector<std::string> ModelNameVector_t;
typedef std::map<std::string, Eqo::EqObjPtr > ModelMap_t;


class SubExpr;

class SubExprData
{
  public:
    SubExprData() : refcount(0) {}
    explicit SubExprData(Eqo::EqObjPtr x);

    void addCount()
    {
      ++refcount;
    }
    size_t getCount() const
    {
      return refcount;
    }

    Eqo::EqObjPtr getEquation() const;

    const std::string &getName(SubExpr &subexpr) const;

  private:
    size_t refcount;
    Eqo::EqObjPtr equation;
    mutable std::string name;
};

/// Creates sub expressions on the fly
class DLL_PROTECTED SubExpr
{
  public:
    SubExpr();

    std::string GetUniqueName();

    const std::string &GetErrorString() const
    {
      return errorString_;
    }

    void CreateSubexpressions(ModelMap_t &);
    void RemoveZeros(ModelMap_t &);

  private:
    bool removeZeros(ModelMap_t &);
    void scanSubexpression(Eqo::EqObjPtr &);
    void createSubexpressions(ModelMap_t &);

    enum ProcessStatus_t {UNTOUCHED = 0, PROCESSING, DONE};

    /// Use this for detecting cycles
    std::map<std::string, ProcessStatus_t> processStatusMap_;

    /// report errors if there is a problem
    std::string errorString_;

    std::map<std::string, SubExprData> SubExprMap;
    std::vector<std::string>       SubExprOrdered;

    size_t unique_counter_;
};


#endif

