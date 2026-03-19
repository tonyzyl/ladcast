from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff('define_model(b, a)')
symdiff('define_model(d, b * c)')
mylist = ordered_list('d');
for i in mylist:
    print(('%s' % i))
