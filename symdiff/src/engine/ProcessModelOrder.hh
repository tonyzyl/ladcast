/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef PROCESSED_MODEL_ORDER
#define PROCESSED_MODEL_ORDER
#include <string>
#include <map>
#include <vector>
#include <list>

#include <memory>
#include "SymdiffImport.hh"


namespace Eqo
{
class EquationObject;
typedef std::shared_ptr<EquationObject> EqObjPtr;
}

typedef std::map<std::string, Eqo::EqObjPtr > ModelMap_t;
typedef std::list<std::pair<std::string, Eqo::EqObjPtr> > ModelList_t;
typedef std::vector<std::string> ModelNameVector_t;


class DLL_PROTECTED ProcessModelOrder
{
  public:
//    ProcessModelOrder();

    const std::string &GetErrorString() const
    {
      return errorString_;
    }

    //// first list is all of our models
    //// second is list of model names we want together
    const ModelNameVector_t &GetModelVector(const ModelMap_t &, const ModelNameVector_t &);

  private:
    void processModelVector(const ModelMap_t &, Eqo::EqObjPtr);

    enum ProcessStatus_t {UNTOUCHED = 0, PROCESSING, DONE};

    /// Use this for detecting cycles
    std::map<std::string, ProcessStatus_t> processStatusMap_;
    /// This is the final list that we create as part of the operation
    ModelNameVector_t processedList_;
    std::string errorString_;

};
#endif

