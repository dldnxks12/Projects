function Mout = Inertia(i, k, n, U11, U12, U13, U21, U22, U23, U31, U32, U33, J1, J2, J3)
Mtemp = 0;

for j = max(i, k):n

    cmd = sprintf("Mtemp = simplify(trace(U%d%d*J%d*U%d%d.'))+Mtemp;", j,k,j,j,i);
    eval(cmd);

end
Mout = Mtemp;

end
