%{
/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "mcTest.hh"
#include "EngineAPI.hh"
#include "mcModel.hh"
#include "mcModelCompiler.hh"
#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sstream>

using std::cout;
using std::cerr;
using std::endl;
int mcerror(const char *msg);

%}

%token EOL 
%token <dval> INT FLOAT
%token <str> VAR UNARYFUNC BINARYFUNC MODELNAME SUBST DEFUSERFUNC DECUSERFUNC DECMODEL DEFMODEL CLEARMODEL USERFUNC CLEARUSERFUNC MACRO
%token <str> MODEL_VALUE_CMD
%token <str> LT GT EQ NEQ LTE GTE NOT AND OR IF IFELSE
%type  <eqptr> expression_list number variable expression userfunc macro macrodef errors modelptr modelcmd
%type  <vararg> vararg
%type  <varobj> varobj
%type <str> all;

%left OR
%left AND
%left EQ NEQ
%left LT LTE GT GTE
%right '='
%left '-' '+'
%left '*' '/'
%right '^'
%nonassoc UMINUS UPLUS NOT

%%
all : expression_list {$$ = $1->stringValue();}
	| {$$ = ""; Eqo::variableMap["$_"] = EngineAPI::con(0.0);}
    ;

expression_list : expression_list expression end_of_line {$$=$2; Eqo::variableMap["$_"] = $$; /*cout << $2 << endl;*/} |
    expression end_of_line {$$=$1; Eqo::variableMap["$_"] = $$; /*cout << $1 << endl;*/ }
    ;

end_of_line : EOL | end_of_line EOL;

expression :
	   IF '(' varobj ')' {
		if ($3.size() == 2)
		{
		    $$ = EngineAPI::createIfObj($3[0], $3[1]);
		}
		else
		{
		    std::ostringstream os;
		    os << "Function \""
			<< $1 << "\" expects 2 arguments and you provided " <<  $3.size();
		    std::string foo = os.str();
		    yyerror(foo.c_str());
		    $$ = EngineAPI::con(0.0);
		}
	    } |
	   IFELSE '(' varobj ')' {
		if ($3.size() == 3)
		{
		    $$ = EngineAPI::createIfElseObj($3[0], $3[1], $3[2]);
		}
		else
		{
		    std::ostringstream os;
		    os << "Function \""
			<< $1 << "\" expects 3 arguments and you provided " <<  $3.size();
		    std::string foo = os.str();
		    yyerror(foo.c_str());
		    $$ = EngineAPI::con(0.0);
		}
	    } |
	   expression EQ  expression {$$ = EngineAPI::createBinaryLogical($2, $1, $3 );} |
	   expression NEQ expression {$$ = EngineAPI::createBinaryLogical($2, $1, $3 );} |
	   expression LT  expression {$$ = EngineAPI::createBinaryLogical($2, $1, $3 );} |
	   expression LTE expression {$$ = EngineAPI::createBinaryLogical($2, $1, $3 );} |
	   expression GT  expression {$$ = EngineAPI::createBinaryLogical($2, $1, $3 );} |
	   expression GTE expression {$$ = EngineAPI::createBinaryLogical($2, $1, $3 );} |
	   expression AND expression {$$ = EngineAPI::createBinaryLogical($2, $1, $3 );} |
	   expression OR  expression {$$ = EngineAPI::createBinaryLogical($2, $1, $3 );} |
	   NOT expression {$$ = EngineAPI::createUnaryLogical($1, $2);} |
	   expression '^' expression {$$ = EngineAPI::getBinaryFuncPtr("pow")($1, $3);} |
	   expression '+' expression {$$ = EngineAPI::operator+($1, $3);} |
	   expression '-' expression {$$ = EngineAPI::operator-($1, $3);} |
	   expression '*' expression {$$ = EngineAPI::operator*($1, $3);} |
	   expression '/' expression {$$ = EngineAPI::operator/($1, $3);} |
	   '-' expression %prec UMINUS {$$ = EngineAPI::getNegation($2);} |
	   '+' expression %prec UPLUS  {$$ = $2;} |
	   UNARYFUNC '(' varobj ')' {$$ = processUnaryFunction($1, $3);} |
	   BINARYFUNC '(' varobj ')' {$$ = processBinaryFunction($1, $3);} |
	   SUBST '(' varobj ')' {$$ = processSubstFunction($1, $3);} |
	   '(' expression ')' {$$ = $2;} |
	   CLEARUSERFUNC {$$ = processClearFunction();} |
	   userfunc |
           modelcmd |
	   number |
	   modelptr |
	   variable |
	   macro |
	   macrodef |
	   errors
	    ;

errors :
	number variable {$$ = CreateErrorMessage($1, "number", $2, "variable");} |
	variable number {$$ = CreateErrorMessage($1, "variable", $2, "number");} |
	number number {$$ = CreateErrorMessage($1, "number", $2, "number");} |
	number UNARYFUNC {$$ = CreateErrorMessageWithOtherType($1, "number", $2, "unary function");} |
	number BINARYFUNC {$$ = CreateErrorMessageWithOtherType($1, "number", $2, "binary function");} |
	number SUBST {$$ = CreateErrorMessageWithOtherType($1, "number", $2, "function");} |
	number '(' {$$ = CreateErrorMessageWithOtherType($1, "number", "(", "parenthetical expression");} |
	variable variable {$$ = CreateErrorMessage($1, "variable", $2, "variable");} |
	variable UNARYFUNC {$$ = CreateErrorMessageWithOtherType($1, "variable", $2, "unary function");} |
	variable BINARYFUNC {$$ = CreateErrorMessageWithOtherType($1, "variable", $2, "binary function");} |
	variable SUBST {$$ = CreateErrorMessageWithOtherType($1, "variable", $2, "function");} |
	variable '(' {$$ = CreateErrorMessageWithOtherType($1, "variable", "(", "parenthetical expression");} |
	UNARYFUNC {
	    $$ = CreateParenthesisError($1, "unary", $1+"(x)");
	} |
	BINARYFUNC {
	    $$ = CreateParenthesisError($1, "binary", $1+"(x, y)");
	} |
	SUBST {
	    $$ = CreateParenthesisError($1, "3 argument", $1+"(x, x, z)");
	} |
	USERFUNC {
	    $$ = CreateParenthesisError($1, "user defined", "myfunction(x, y, z)");
	} |
	DECUSERFUNC {
	    $$ = CreateParenthesisError($1, "built in", $1+"(f(x))");
	} |
	DEFUSERFUNC {
	    $$ = CreateParenthesisError($1, "built in", $1+"(f(x), 1/x)");
	}
	;

macrodef : MACRO '=' expression {
		$$ = DefineMacro($1, $3);
	   };

macro :
	   MACRO {
		$$ = GetMacro($1);
	   };

modelcmd : DECMODEL '(' VAR ')' {
            $$ = DeclareModel($3);
          } |
           DECMODEL '(' MODELNAME ')' {
            $$ = DeclareModel($3);
          } |
           DEFMODEL '(' VAR ',' expression ')' {
            $$ = DefineModel($3, $5);
          } |
           DEFMODEL '(' MODELNAME ',' expression ')' {
            $$ = DefineModel($3, $5);
          } |
            MODEL_VALUE_CMD '(' MODELNAME ')' {
            $$ = GetModelValue($3);
          } |
           CLEARMODEL '(' VAR ')' {
            $$ = ClearModel($3);
          } |
           CLEARMODEL '(' MODELNAME ')' {
            $$ = ClearModel($3);
          }
          ;

userfunc :
	    DEFUSERFUNC '(' VAR '(' vararg ')' ',' varobj ')' {
		$$ = DefineUserFunction($3, $5, $8);
	    } |
	    DEFUSERFUNC '(' USERFUNC '(' vararg ')' ',' varobj ')' {
		$$ = ReDefineUserFunction($3, $5, $8);
	    } |
	   USERFUNC '(' varobj ')' {
		$$ = EvaluateUserFunction($1, $3);
	   } |
	   DECUSERFUNC '(' VAR '(' vararg ')' ')' {
		$$ = DeclareUserFunction($3, $5);
	   } |
	   DECUSERFUNC '(' USERFUNC '(' vararg ')' ')' {
		$$ = DeclareUserFunction($3, $5);
	   } |
	   DECUSERFUNC '(' UNARYFUNC  '(' vararg ')' ')' {$$ = CreateBuiltInError($3);} |
	   DECUSERFUNC '(' BINARYFUNC '(' vararg ')' ')' {$$ = CreateBuiltInError($3);} |
	   DEFUSERFUNC '(' UNARYFUNC  '(' vararg ')' ')' {$$ = CreateBuiltInError($3);} |
	   DEFUSERFUNC '(' BINARYFUNC '(' vararg ')' ')' {$$ = CreateBuiltInError($3);}
	   ;

varobj :    expression {
		std::vector<Eqo::EqObjPtr> foo;
		foo.push_back($1);
		$$ = foo;
	    } |
	    varobj ',' expression {
		$$ = $1;
		$$.push_back($3);
	    }
	    ;

vararg :    VAR {
		std::vector<std::string> foo;
		foo.push_back($1);
		$$ = foo;
	    } |
	    vararg ',' VAR {
		$$ = $1;
		$$.push_back($3);
	    }
	    ;

number :    INT {$$ = EngineAPI::con($1);} |
	    FLOAT {$$ = EngineAPI::con($1);}
	  ;

variable :    VAR {$$ = EngineAPI::var($1);}
	  ;

modelptr    :    MODELNAME {$$ = Eqo::mod($1);}
	  ;

%%
