from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff('define_model(x, 0)')
symdiff('define_model(y, x + z)')
remove_zeros()
print(('%s' % symdiff('model_value(y)')))
