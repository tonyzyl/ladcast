from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff('define_model(x, y * z)')
symdiff('define_model(z, y * z)')
subexpression()
l = model_list()
for i in l:
    print(("%s, %s" % (i, symdiff('model_value(%s)' % i))))
