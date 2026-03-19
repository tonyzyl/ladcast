from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

print(symdiff('declare_model(x)'))
print(symdiff('define_model(y, a*x)'))
print(symdiff('diff(y, x)'))
print(symdiff('clear_model(x)'))
print(ordered_list('y'))
print(symdiff('declare_model(x)'))
print(ordered_list('y'))
print(symdiff('define_model(x,y)'))
# catch cycles
try:
    print(ordered_list('y'))
except SymdiffError as x:
    print(x)

for i in model_list():
    print(i + ' ' + symdiff('model_value(%s)' % i))
    symdiff('clear_model(%s)' % i)

print("")
print(symdiff('declare_model(x)'))
print(symdiff('define_model(y,x^2)'))
print(symdiff('diff(y,x)'))
print(symdiff('clear_model(y)'))
print(symdiff('model_value(y__x)'))
print(symdiff('diff(model_value(y__x),z)'))

for i in model_list():
    print(i + ' ' + symdiff('model_value(%s)' % i))

print("")
symdiff('clear_model(x)')
symdiff('clear_model(x__z)')

for i in model_list():
    print(i + ' ' + symdiff('model_value(%s)' % i))
    symdiff('clear_model(%s)' % i)

