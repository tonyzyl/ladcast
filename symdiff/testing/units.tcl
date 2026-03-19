package require symdifftcl
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff {define_model(E, m * c^2)}
symdiff {$a = model_value(E);
        $a = subst($a, m, Kg);
        $a = subst($a, c, m/s);
        $a = simplify($_);}
puts [symdiff {$a}]
