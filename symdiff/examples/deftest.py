from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

commands = [
    "define(f(x,y),y^2, 2*y*x+3*y^2)",
    "simplify(diff(f(x,y),x))",
    "simplify(diff(f(x,y),y))",
    "simplify(diff(f(y,x),y))",
    "simplify(diff(f(y,x),x))",

    "declare(cos(x))",
    "define(sin(x),cos(x))",
    "define(cos(x),-sin(x))",
    "simplify(diff(sin(x),x))",
    "simplify(diff(sin(y),x))",
    "simplify(diff(sin(x^2),x))",
    "simplify(diff(sin(y^2),y))",

    "clear",
    "define(f(x,y),diff(x*y^2+y^3,x),diff(x*y^2+y^3,y))",
    "simplify(diff(f(x,y),x))",
    "simplify(diff(f(x,y),y))",
    "simplify(diff(f(y,x),y))",
    "simplify(diff(f(y,x),x))",

    "clear",
    "declare(g(x,y,z))",
    "define(f(x,y,z),g(y,z,x),1,0)",
    "diff(f(x,y,z),x)",
    "diff(f(a,b,c),x)",
    "diff(f(a,b,c),a)",
    "diff(f(a,b,c),b)",
    "diff(f(a,b,c),c)",

    "clear",
    "a+b",
    "$_*$_",
    "expand($_)",
    "simplify($_)",
    "$a = $_ +1",
    "$b = $_ - $a",
    "$c = simplify($_)",
    "$c = simplify(expand($_))",
]

for i in commands:
    print(symdiff(i))
