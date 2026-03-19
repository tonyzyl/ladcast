package require symdifftcl
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff "define_model(a,b + c)"
symdiff "diff(a,b)"
puts [symdiff "model_value(a__b)"]
symdiff "define_model(a,c)"
symdiff "diff(a,b)"
puts [symdiff "model_value(a__b)"]

