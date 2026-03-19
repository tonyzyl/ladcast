from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff('declare_model(y)')
symdiff('define_model(x, 3 * y + z)')
symdiff('diff(x, z)')
print(('%s' % symdiff('model_value(x)')))
print(('%s' % symdiff('model_value(x__z)')))
print(('%s' % symdiff('model_value(y__z)')))
