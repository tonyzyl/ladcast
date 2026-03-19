/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef MC_TEST_HH
#define MC_TEST_HH

#include <string>
#include <vector>

#include "EquationObject.hh"

typedef struct {
    std::string str;
    double dval;
    Eqo::EqObjPtr eqptr;
    std::vector<std::string> vararg; // variable number of arguments
    std::vector<Eqo::EqObjPtr> varobj; // variable number of objects
} myyystype;

#define YYSTYPE myyystype

int mcerror(const char *);
int mclex();
int mcparse();
void mcrestart(FILE *);
extern FILE * mcin;
void prompt();
// flex stuff to parse string
#define yyconst const
typedef struct yy_buffer_state *YY_BUFFER_STATE;
YY_BUFFER_STATE mc_scan_string( yyconst char *yy_str );
void mc_delete_buffer( YY_BUFFER_STATE b );

Eqo::EqObjPtr CreateErrorMessage(const Eqo::EqObjPtr &, const std::string &, const Eqo::EqObjPtr &, const std::string &);
Eqo::EqObjPtr CreateErrorMessageWithOtherType(const Eqo::EqObjPtr &, const std::string &, const std::string &, const std::string &);
Eqo::EqObjPtr CreateBuiltInError(const std::string &);
Eqo::EqObjPtr CreateParenthesisError(const std::string &, const std::string &, const std::string &);

Eqo::EqObjPtr processUnaryFunction(const std::string &, const std::vector<Eqo::EqObjPtr> &);
Eqo::EqObjPtr processBinaryFunction(const std::string &, const std::vector<Eqo::EqObjPtr> &);
Eqo::EqObjPtr processSubstFunction(const std::string &, const std::vector<Eqo::EqObjPtr> &);
Eqo::EqObjPtr processClearFunction();

Eqo::EqObjPtr DefineMacro(const std::string &, Eqo::EqObjPtr);
Eqo::EqObjPtr GetMacro(const std::string &s);



Eqo::EqObjPtr DeclareUserFunction(const std::string &, const std::vector<std::string> &);
Eqo::EqObjPtr DefineUserFunction(const std::string &, std::vector<std::string>, std::vector<Eqo::EqObjPtr>);
Eqo::EqObjPtr ReDefineUserFunction(const std::string &, std::vector<std::string>, std::vector<Eqo::EqObjPtr>);
Eqo::EqObjPtr EvaluateUserFunction(const std::string &, std::vector<Eqo::EqObjPtr>);

#endif
