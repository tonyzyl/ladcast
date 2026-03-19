from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff('declare_model(x)')
for i, v in enumerate(symdiff_table('a*x + b*c')):
    print('%s %s' % (i, v))
