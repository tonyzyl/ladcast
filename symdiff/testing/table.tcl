package require symdifftcl
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff "declare_model(x)"
set i 0
set x [symdiff_table "a*x + b*c"]
foreach j $x {
    puts "$i $j"
    incr i
}

