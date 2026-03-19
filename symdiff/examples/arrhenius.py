from symdiff import *
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff('define_model(arrhenius, simplify(Cf * pow(T, Etar) * exp(-Ea/T)))')
symdiff('define_model(KC1      , simplify(A3 * R * T * exp((A1 * (-pow(T,(-1)) + pow(A2,(-1))) * pow(R,(-1))))))')
symdiff('define_model(kb_model , simplify(arrhenius / KC1))')
symdiff('declare_model(fprod)')

symdiff('define_model(rprod    , simplify(H * CN * AR))')
symdiff('define_model(omega_dot_H     , simplify((vr_H - vf_H) * (arrhenius * fprod - kb_model * rprod)))')
symdiff('diff(omega_dot_H, T)')
symdiff('diff(omega_dot_H, H)')
symdiff('diff(omega_dot_H, HCN)')
symdiff('diff(omega_dot_H, AR)')
symdiff('diff(omega_dot_H, CN)')

subexpression()
remove_zeros()
mylist = ordered_list('omega_dot_H', 'omega_dot_H__T', 'omega_dot_H__H', 'omega_dot_H__HCN', 'omega_dot_H__AR', 'omega_dot_H__CN')
for i in mylist:
    mv = symdiff("model_value(%s);" % i)
    print("%s, %s" % (i, mv))

