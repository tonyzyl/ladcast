package require symdifftcl
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

proc callSymdiff {arg} {
puts [format "\"%s\"\n\"%s\"\n\"%s\"\n\n" $arg [symdiff "$arg"] [symdiff "diff($arg,x)"]]
}

callSymdiff "exp(0)"
callSymdiff "simplify(exp(0))"
callSymdiff "exp(1)"
callSymdiff "exp(2)"

callSymdiff "pow(x,0)"
callSymdiff "simplify(pow(x,0))"
callSymdiff "pow(x,1)"
callSymdiff "simplify(pow(x,1))"
callSymdiff "pow(x*y,1)"
callSymdiff "pow(x*y,2)"
callSymdiff "pow(x,2)"
callSymdiff "pow(2,x)"
callSymdiff "pow(2,x*y)"
callSymdiff "pow(2,3)"
callSymdiff "pow(0,x)"
callSymdiff "simplify(pow(0,x))"
callSymdiff "pow(1,x)"
callSymdiff "simplify(pow(1,x))"

callSymdiff "exp(x)"
callSymdiff "exp(x*y)"
callSymdiff "log(x)"
callSymdiff "log(x*y)"
callSymdiff "log(1)"
callSymdiff "x*y"
callSymdiff "x"
callSymdiff "0*x"
callSymdiff "0*x*y"
callSymdiff "x*y*z"
callSymdiff "0+x+y"
callSymdiff "pow(x*y,z)"
callSymdiff "pow(y,x*z)"
callSymdiff "pow(x*y,x*z)"
