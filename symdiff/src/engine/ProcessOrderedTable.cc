/***
Copyright 2013 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "ProcessOrderedTable.hh"
#include "EquationObject.hh"


void ProcessOrderedTable::run(Eqo::EqObjPtr eq)
{
  errors_.clear();
  orderedTable_.clear();
  orderedIndex_.clear();
  processStatusMap_.clear();
  processOrderedTable(eq);
  for (size_t i = 0; i < orderedTable_.size(); ++i)
  {
    std::vector<size_t> &refs = orderedTable_[i].indexes_;
    if (!refs.empty())
    {
      for (size_t j = 0; j < refs.size(); ++j)
      {
        orderedTable_[refs[j]].references_.push_back(i);
      }
    }
  }
  /*
     it should be noted that the existence of 2 different instances
     of the string value in the index means there is a type mismatch
  */
  for (OrderedIndex_t::const_iterator it = orderedIndex_.begin(); it != orderedIndex_.end(); ++it)
  {
    if (it->second.size() > 1)
    {
      errors_.push_back(it->first);
    }
  }
}

namespace {
bool CompareRows(const OrderedTableData &r1, const OrderedTableData &r2)
{
  bool ret = true;
  if (r1.ptr_ != r2.ptr_)
  {
    const size_t len = r1.indexes_.size();
    if (r1.ptr_->getType() != r2.ptr_->getType())
    {
      ret = false;
    }
    else if (len != r2.indexes_.size())
    {
      ret = false;
    }
    else
    {
      /*
        For the same type and args, each sub index will match
      */
      for (size_t i = 0; i < len; ++i)
      {
        if (r1.indexes_[i] != r2.indexes_[i])
        {
          ret = false;
          break;
        }
      }
    }
  }
  return ret;
}

size_t FindMatchedRow(const OrderedTable_t &table, const std::vector<size_t> &indexes, OrderedTableData &newrow)
{
  size_t ret = size_t(-1);
  size_t len = indexes.size();
  for (size_t i = 0; i < len; ++i)
  {
    if (CompareRows(table[indexes[i]], newrow))
    {
      ret = indexes[i];
      break;
    }
  }
  return ret;
}
}

/* make this return the indexes of the args processed */
/* does not handle when sub types (model/variable) have different types */
/* looks like it also handles subexpression elimination as well */
size_t ProcessOrderedTable::processOrderedTable(Eqo::EqObjPtr eq) {
  /* are we already in the table */
  const std::string &value = eq->stringValue();

  /* this implicitly creates if it doesn't exist */
  std::vector<size_t> &myindexes = orderedIndex_[value];

  size_t myindex = size_t(-1);

  OrderedTableData newrow;
  newrow.value_ = value;
  newrow.ptr_   = eq;

  // This first test just does a pointer match
  myindex = FindMatchedRow(orderedTable_, myindexes, newrow);

  /* a deep comparison to make sure sub arguments are in the table */
  if (myindex == size_t(-1))
  {
    const std::vector<Eqo::EqObjPtr> &args = eq->getArgs();
    newrow.indexes_.reserve(args.size());
    // This is our current index
    for (std::vector<Eqo::EqObjPtr>::const_iterator it = args.begin(); it != args.end(); ++it)
    {
      newrow.indexes_.push_back(processOrderedTable(*it));
    }

    myindex = FindMatchedRow(orderedTable_, myindexes, newrow);
    if (myindex == size_t(-1))
    {
      myindex = orderedTable_.size();
      // implicitly adds to the existing index row
      myindexes.push_back(myindex);
      orderedTable_.push_back(newrow);
    }
  }
  return myindex;
}

