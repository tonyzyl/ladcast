/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "EquationObject.hh"
#include <string>
#include <sstream>
#include <iostream>
#include <typeinfo>
#include <cmath>
#include <utility>
#include <algorithm>
#include <map>
#include <iomanip>


namespace Eqo {
typedef std::shared_ptr<EquationObject> EqObjPtr;

std::map<std::string, EqObjPtr> variableMap;


const std::string &EquationObject::stringValue() const
{
  if (stringValue_.empty())
  {
    stringValue_ = this->createStringValue();
  }

  return stringValue_;
}




std::set<std::string> UniteReferencedType(EqObjType rt, const EqObjPtr &op1, const EqObjPtr &op2)
{
   std::set<std::string> out;

   if (op1->getType() == rt)
   {
         out.insert(op1->stringValue());
   }
   else
   {
      std::set<std::string> rm= op1->getReferencedType(rt);
      std::set<std::string>::iterator it=rm.begin();
      std::set<std::string>::iterator end=rm.end();
      for ( ; it != end; ++it)
            out.insert(*it);
   }

   if (op2->getType() == rt)
   {
         out.insert(op2->stringValue());
   }
   else
   {
      std::set<std::string> rm= op2->getReferencedType(rt);
      std::set<std::string>::iterator it=rm.begin();
      std::set<std::string>::iterator end=rm.end();
      for ( ; it != end; ++it)
            out.insert(*it);
   }
   return out;
}

std::set<std::string> UniteReferencedType(EqObjType rt, const std::vector<EqObjPtr> &x)
{
   std::set<std::string> out;
   for (size_t i=0; i < x.size(); ++i)
   {
      if (x[i]->getType() == rt)
      {
            out.insert(x[i]->stringValue());
      }
      else
      {
         std::set<std::string> rm= x[i]->getReferencedType(rt);
         std::set<std::string>::iterator it=rm.begin();
         std::set<std::string>::iterator end=rm.end();
         for ( ; it != end; ++it)
         {
            out.insert(*it);
         }
      }
   }
   return out;
}

} // always close the namespace

