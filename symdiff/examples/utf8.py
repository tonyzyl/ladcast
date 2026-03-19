# -*- coding: utf-8 -*-
from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

alpha = 'α'
print(alpha)
x = symdiff('%s' % alpha);
x = symdiff('diff(%s^2, %s)' % (alpha, alpha));
print(x)
