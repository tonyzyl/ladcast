from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff('declare_model(x)')
symdiff('declare_model(y)')
l = model_list();
for i in l:
    print(('%s' % i))
