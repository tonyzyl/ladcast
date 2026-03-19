/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "mcTest.hh"
#include "EngineAPI.hh"
#include "mcModelCompiler.hh"
#include "EquationObject.hh"
#include "UserFunc.hh"
#include "mcModel.hh"
#include "Context.hh"
#include "Functions.hh"

#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <iomanip>
#include <fstream>

using std::cerr;
using std::endl;
using std::ostringstream;
using std::ofstream;
using std::string;

void print_hh (ofstream &);
void print_cc(ofstream &, const string &);

void GenerateEquationDerivatives();
void PrintModelList();
void CreateTotalNodeList();

enum assembletype_t {DCASSEMBLE, ACASSEMBLE};
void PrintAssemblyRoutine(ofstream &, assembletype_t);


/**
 *  Creates error messages for use when a delimiter is not specified
 */
Eqo::EqObjPtr CreateErrorMessage(const Eqo::EqObjPtr &e1, const std::string &t1, const Eqo::EqObjPtr &e2, const std::string &t2)
{
    std::string s1 = e1->stringValue();
    std::string s2 = e2->stringValue();
    std::ostringstream os;
    os << "The " << t1 << " \"" << s1 <<
	"\" and " << t2 << " " << "\"" << s2 << "\" must be separated by an operator (e.g. \""
	<< s1 << " * " << s2 << "\")";
    mcerror(os.str().c_str());
    return Eqo::con(0);
}

/// Handles when one is a function name
Eqo::EqObjPtr CreateErrorMessageWithOtherType(const Eqo::EqObjPtr &e1, const std::string &t1, const std::string &e2, const std::string &t2)
{
    std::string s1 = e1->stringValue();
    std::ostringstream os;
    os << "The " << t1 << " \"" << s1 <<
	"\" and " << t2 << " " << "\"" << e2 << "\" must be separated by an operator (e.g. \""
	<< s1 << " * " << e2 << " . . .\")";
    mcerror(os.str().c_str());
    return Eqo::con(0);
}

Eqo::EqObjPtr CreateBuiltInError(const std::string &func)
{
    std::ostringstream os;
    os << func << " is a built in function and cannot be overridden";
    std::string foo = os.str();
    mcerror(foo.c_str());
    return Eqo::con(0);
}

Eqo::EqObjPtr CreateParenthesisError(const std::string &name, const std::string &type, const std::string &example)
{
    std::ostringstream os;
    os << "\"" << name <<
	"\" is a " << type << " function requiring parenthesis around its arguments.  For example: \"" << example << "\"";
    std::string foo = os.str();
    mcerror(foo.c_str());
    return Eqo::con(0);
}

Eqo::EqObjPtr processUnaryFunction(const std::string &function, const std::vector<Eqo::EqObjPtr> &args)
{
    Eqo::EqObjPtr ret = Eqo::con(0);
    if (args.size() == 1)
    {
	ret = Eqo::getUnaryFuncPtr(function)(args[0]) ;
    }
    else
    {
	std::ostringstream os;
	os << "Function \""
	    << function << "\" expects 1 argument and you provided " <<  args.size() << ":";
	/// Copy and pasted
	for (size_t i = 0; i < args.size(); ++i)
	{
	    if (i != 0)
	    {
		os << ",";
	    }
	    os << " \"" << args[i] << "\"";
	}
	std::string foo = os.str();
	mcerror(foo.c_str());
	ret = Eqo::con(0);
    }
    return ret;
}

Eqo::EqObjPtr processBinaryFunction(const std::string &function, const std::vector<Eqo::EqObjPtr> &args)
{
    Eqo::EqObjPtr ret = Eqo::con(0);
    if (args.size() == 2)
    {
	ret = EngineAPI::getBinaryFuncPtr(function)(args[0], args[1]) ;
    }
    else
    {
	std::ostringstream os;
	os << "Function \""
	    << function << "\" expects 2 arguments and you provided " <<  args.size() << ":";
	/// Copy and pasted
	for (size_t i = 0; i < args.size(); ++i)
	{
	    if (i != 0)
	    {
		os << ",";
	    }
	    os << " \"" << args[i] << "\"";
	}
	std::string foo = os.str();
	mcerror(foo.c_str());
	ret = Eqo::con(0);
    }
    return ret;
}

Eqo::EqObjPtr processSubstFunction(const std::string &function, const std::vector<Eqo::EqObjPtr> &args)
{
    Eqo::EqObjPtr ret = Eqo::con(0);
    if (args.size() == 3)
    {
	ret = EngineAPI::subst(args[0], args[1], args[2]); /*cerr << args->stringValue() << endl;*/
    }
    else
    {
	std::ostringstream os;
	os << "Function \""
	    << function << "\" expects 3 arguments and you provided " <<  args.size() << ":";
	/// Copy and pasted
	for (size_t i = 0; i < args.size(); ++i)
	{
	    if (i != 0)
	    {
		os << ",";
	    }
	    os << " \"" << args[i] << "\"";
	}
	std::string foo = os.str();
	mcerror(foo.c_str());
	ret = Eqo::con(0);
    }
    return ret;
}

Eqo::EqObjPtr processClearFunction()
{
   Eqo::UserFuncMap.clear();
   Context &context = Context::GetInstance();
   context.ClearAllModels();
   return Eqo::con(0);
}

Eqo::EqObjPtr GetMacro(const std::string &s)
{
    if (!Eqo::variableMap.count(s))
    {
	Eqo::variableMap[s] = Eqo::con(0);
    }
    return Eqo::variableMap[s]->clone();
}

Eqo::EqObjPtr DefineMacro(const std::string &s, Eqo::EqObjPtr eq)
{
    /* it is probably impossible to reference $3, except through $_ but just in case */
    Eqo::variableMap[s] = eq->clone();
    return eq->clone();
}

Eqo::EqObjPtr DeclareUserFunction(const std::string &fname, const std::vector<std::string> &args)
{
    Eqo::EqObjPtr ret = Eqo::con(0);
    if (Eqo::UserFuncMap.count(fname))
    {
	std::ostringstream os;
	os << fname << " has already been declared with "
	    << Eqo::UserFuncMap[fname].size() << " argument(s)";
	std::string foo = os.str();
	mcerror(foo.c_str());
    }
    else
    {
	const size_t nargs = args.size();
	Eqo::CreateUserFunc(fname, args.size());
	std::vector<Eqo::EqObjPtr> f;
	for (size_t i = 0; i < nargs; ++i)
	{
	    f.push_back(EngineAPI::var(args[i]));
	}
	ret = Eqo::EqObjPtr(new Eqo::UserFunc(fname, f));
    }
    return ret;
}

Eqo::EqObjPtr DefineUserFunction(const std::string &fname, std::vector<std::string> args, std::vector<Eqo::EqObjPtr> objs)
{
    Eqo::EqObjPtr ret = Eqo::con(0);
    std::ostringstream os;

    std::map<std::string, size_t> inputArgs;
    for (size_t i = 0; i < args.size(); ++i)
    {
	inputArgs[args[i]] += 1;
    }
    for (std::map<std::string, size_t>::iterator it = inputArgs.begin(); it != inputArgs.end(); ++it)
    {
	if (it->second > 1)
	{
	    os << " Variable \"" << it->first << "\" is used " << it->second << " times.  It may only be used once in the variable list for defining functions.";
	}
    }

    std::set<std::string> uniqueReferencedArgs;
    for (size_t i = 0; i < objs.size(); ++i)
    {
	std::set<std::string> tmp = objs[i]->getReferencedType(Eqo::VARIABLE_OBJ);
	for (std::set<std::string>::iterator it = tmp.begin(); it != tmp.end(); ++it)
	{
	    if (!inputArgs.count(*it))
	    {
		os << " Variable \"" << *it << "\" is used in expression \"" << objs[i] << "\", even though it is not present in the variable list for defining functions."; 
	    }
	}
    }

    if (args.size() != objs.size())
    {
	os << "argument list not the same size as the object list "
	    << args.size() << " args versus " <<  objs.size() << " objects";
    }

    if (!os.str().empty())
    {
	std::string foo = os.str();
	mcerror(foo.c_str());
    }
    else
    {
	const size_t nargs = args.size();
	if (!Eqo::UserFuncMap.count(fname))
	{
	    Eqo::CreateUserFunc(fname, nargs);
	}
	Eqo::UserDiffInfoVec udiv;
	std::vector<Eqo::EqObjPtr> f;

	// create special reserved names
	for (size_t i = 0; i < nargs; ++i)
	{
	    std::string foo = args[i];
	    args[i] = "@" + foo; // create reserved name
	    for (size_t j = 0; j < nargs; ++j)
	    {
		objs[j]=objs[j]->subst(foo, EngineAPI::var(args[i]));
	    }
	}
	for (size_t i = 0; i < nargs; ++i)
	{
	    udiv.push_back(std::make_pair(args[i], objs[i]));
	    f.push_back(EngineAPI::var(args[i].substr(1)));
	}
	Eqo::SetUserFuncDiffs(fname, udiv);
	ret = Eqo::EqObjPtr(new Eqo::UserFunc(fname, f));
    }
    return ret;
}

Eqo::EqObjPtr ReDefineUserFunction(const std::string &fname, std::vector<std::string> args, std::vector<Eqo::EqObjPtr> objs)
{
    Eqo::EqObjPtr ret = Eqo::con(0);
    const size_t ulen = Eqo::UserFuncMap[fname].size();
    const size_t nargs = args.size();
    if (ulen != nargs)
    {
	std::ostringstream os;
	os << fname << " has been declared or defined with "
	    << ulen << " argument(s), but is being redefined with " <<
	    nargs << " arguments(s).";
	std::string foo = os.str();
	mcerror(foo.c_str());
    }
    else if (args.size() != objs.size())
    {
	std::ostringstream os;
	os << "argument list for " << fname << " is not the same size as the object list. "
	    << args.size() << " args versus " <<  objs.size() << " objects";
	std::string foo = os.str();
	mcerror(foo.c_str());
    }
    else
    {
	ret = DefineUserFunction(fname, args, objs);
    }
    return ret;
}

Eqo::EqObjPtr EvaluateUserFunction(const std::string &fname, std::vector<Eqo::EqObjPtr> objs)
{
    Eqo::EqObjPtr ret = Eqo::con(0);
    const size_t ulen = Eqo::UserFuncMap[fname].size();
    const size_t vlen = objs.size();
    if ( ulen != vlen)
    {
	std::ostringstream os;
	os << fname << " has been declared or defined with "
	    << ulen << " argument(s), but has been supplied with " <<
	    vlen << " arguments(s).";
	std::string foo = os.str();
	mcerror(foo.c_str());
    }
    else
    {
	std::vector<Eqo::EqObjPtr> f;
	const size_t nargs = objs.size();
	for (size_t i = 0; i < nargs; ++i)
	{
	    f.push_back(objs[i]);
	}
	ret = Eqo::EqObjPtr(new Eqo::UserFunc(fname, f));
    }
    return ret;
}

