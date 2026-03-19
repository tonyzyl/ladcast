%{
/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

// must include test.hh first
// to use struct instead of union.
#include <cstdlib>
#include "mcTest.hh"
#include "mcParser.hh"
#include "EngineAPI.hh"
// For isatty
#ifdef _WIN32
#include <io.h>
#define YY_NO_UNISTD_H
#else
#include <unistd.h>
#endif 
#include "EquationObject.hh"
#include "UserFunc.hh"
//#include "FunctionTables.hh"
#include "Context.hh"
#define YY_NO_UNPUT
// remove clang compiler warning about deprecated keyword
#define register
%}
%option noyywrap
%option nounput
/*%option stack*/

integer [0-9]+ 
float1 [0-9]*\.[0-9]+
float2 [0-9]+\.[0-9]*
float3 [0-9]+\.?[0-9]*[eE][+-]?[0-9]+

/* Unicode goodness from http://osdir.com/ml/lex.flex.general/2004-03/msg00023.html */
U80    [\200-\277]
U81    [\300-\337]{U80}
U82    [\340-\357]{U80}{2}
U83    [\360-\367]{U80}{3}
U84    [\370-\373]{U80}{4}
U85    [\374-\375]{U80}{5}
UTF8   {U81}|{U82}|{U83}|{U84}|{U85}


%%
[ \t\r\n]*	    ;

[\n\;]	{return EOL;}

{float1}|{float2}|{float3}    {mclval.dval = atof(yytext); return FLOAT;}

{integer}   {mclval.dval = atoi(yytext); return INT;}

subst   {mclval.str=yytext; return SUBST;}

define   {mclval.str=yytext; return DEFUSERFUNC;}

declare   {mclval.str=yytext; return DECUSERFUNC;}

declare_model   {mclval.str=yytext; return DECMODEL;}

model_value   {mclval.str=yytext; return MODEL_VALUE_CMD;/*Should this return a value, or should we have a top level command test*/}

define_model   {mclval.str=yytext; return DEFMODEL;}

clear_model   {mclval.str=yytext; return CLEARMODEL;}

clear   {mclval.str=yytext; return CLEARUSERFUNC;}

\$_	       {mclval.str = yytext; return MACRO;}


\<         {mclval.str=yytext; return LT;}
\<=        {mclval.str=yytext; return LTE;}
\>         {mclval.str=yytext; return GT;}
\>=        {mclval.str=yytext; return GTE;}
==         {mclval.str=yytext; return EQ;}
!=         {mclval.str=yytext; return NEQ;}
!          {mclval.str=yytext; return NOT;}
&&         {mclval.str=yytext; return AND;}
\|\|       {mclval.str=yytext; return OR;}

\$([A-Za-z]|{UTF8})([A-Za-z_0-9]|{UTF8})*  {
                mclval.str = yytext; return MACRO;
                }
if {
      mclval.str = yytext;
      return IF;
   }

ifelse {
      mclval.str = yytext;
      return IFELSE;
   }

([A-Za-z]|{UTF8})([A-Za-z_0-9:@.]|{UTF8})*  {
                mclval.str = yytext;

		if (EngineAPI::isUnaryFunction(mclval.str))
		    return UNARYFUNC;
		else if (EngineAPI::isBinaryFunction(mclval.str))
		    return BINARYFUNC;
		else if (Eqo::UserFuncMap.count(mclval.str))
		    return USERFUNC;
	        else if (Context::GetInstance().IsInModelList(mclval.str))
		    return MODELNAME;
		else
		    return VAR;
		    }

.   {return yytext[0];}
%%


