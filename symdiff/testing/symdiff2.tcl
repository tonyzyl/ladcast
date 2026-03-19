package require symdifftcl
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff "declare_model(x)"
puts [symdiff "diff(x,y)"]
puts [symdiff "diff(x,x)"]
puts [symdiff "define_model(z, 2 * a + b *c)"]
puts [symdiff "diff(z, x)"]
puts [symdiff "diff(z, a)"]
puts [symdiff "model_value(z)"]
puts [symdiff "model_value(z__x)"]
puts [symdiff "model_value(z__a)"]
puts [symdiff "define_model(qq, 2 * a)"]
puts [symdiff "model_value(qq)"]
puts [symdiff "define_model(qq, model_value(qq)+ b)"]
puts [symdiff "model_value(qq)"]
# this is a recursive definition and should be caught if ever printed out
puts [symdiff "define_model(qq, model_value(qq)+ qq)"]
puts [symdiff "model_value(qq)"]
puts [symdiff "define_model(qq, simplify(model_value(qq)))"]
puts [symdiff "model_value(qq)"]

