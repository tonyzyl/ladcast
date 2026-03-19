from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

def maptest(arg):
    print("")
    print(arg);
    try:
        out = symdiff(arg)
    except SymdiffError as x:
        out = x
    print(out)

maptest("a*x")
maptest("simplify(a*a)")
maptest("simplify(pow(x, 0.0))")
maptest("simplify(pow(x, 1.0))")
maptest("simplify(pow(1.0, x))")
maptest("simplify(exp(1.0))")
maptest("simplify(exp(0.0))")
maptest("simplify(exp(log(x)))")
maptest("simplify(log(exp(x)))")
maptest("simplify(log(1.0))")

maptest("(exp(pow(c,pow(x,pow(c,x)))))")
maptest("diff((exp(pow(c,pow(x,pow(c,x))))),x)")
maptest("simplify(diff((exp(pow(c,pow(x,pow(c,x))))),x))")
maptest("simplify((((((((1 * pow(x,(-1))) * pow(c,x)) + ((log(c) * pow(c,x)) * log(x))) * pow(x,pow(c,x))) * log(c)) * pow(c,pow(x,pow(c,x)))) * exp(pow(c,pow(x,pow(c,x))))))")

maptest("simplify(a*5*pow(a,-1))")

maptest("c*c*c*c")
maptest("simplify(c*c*c*c)")
maptest("diff(c*c*x*x,x)")

maptest("simplify(log(pow(x,y)))")

maptest("simplify(a*a+a*a*a)")
maptest("simplify(a*a+a*a)")
maptest("simplify(a+a+a*a)")
maptest("simplify(a+a+a+a*a)")

maptest("simplify(y*pow(x,(-1)) * pow(x,y))");
maptest("simplify(pow(x,(-1)) * pow(x,y)*y)");
maptest("simplify(pow(x,(-1)) * pow(x,y)*x)");

maptest("unscaledval(2*x)")
maptest("scale(2*x)")

maptest("a - a")
maptest("a + -a")
maptest("a - b")

maptest("simplify(a - a)")
maptest("simplify(a + -a)")
maptest("simplify(a - b)")

maptest("-simplify(a - b)")
maptest("simplify(-simplify(a - b))")

maptest("(-1*3*4*-3*x)");
maptest("simplify(-1*3*4*-3*x)");

maptest("-a");
maptest("a++-+a")
maptest("a+*+a");
maptest("diff(-1*3*4*(-3)*x,x)");
maptest("diff(simplify(-1*3*4*(-3)*x),x)");
maptest("diff((-1*3*4*(-3)*x),x)");

maptest("(a *(-a))");
maptest("(a * -a)");
maptest("(a * (+a))");
maptest("(a * +a)");

maptest("expand((a)*(c+d))")
maptest("expand((a + b)*(c+d))")
maptest("expand((a+b)*(c+d)*(e+f+g))")
maptest("expand((a-1)*(b+1))")
maptest("expand((a-1)*(a+1))")

maptest("simplify((a-1)*(a+1)/(a+1))")
maptest("simplify((a-1)*(a+1)^2/(a+1))")
maptest("simplify((a-1)*(a+1)^2/(a+1)^3)")

maptest("expand((a-b)^-2)")
maptest("expand((a-b)^+2)")
maptest("expand(((a-b)^2)/((a-b)^2))")
maptest("expand((a-b)^2/(a-b)^2)")
maptest("simplify(expand((a-b)^2/(a-b)^2))")
maptest("simplify((a-b)^2/expand((a-b)^2))")

maptest("unscaledval(-1*10)")
maptest("scale(-1*10)")
maptest("unsignedval(-1*10)")
maptest("sign(-1*10)")
maptest("(a-3)")
maptest("simplify(a-3)")
maptest("2-10")
maptest("unscaledval(-10)")
maptest("scale(-10)")
maptest("sign(-10)")
maptest("unsignedval(-10)")
maptest("simplify(-8*a*x^2 -3*b*c^2)")

maptest("simplify(x,y)")
maptest("subst(y)")
