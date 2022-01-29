function Cout = CC2(i, k, m, n, U11, U12, U21, U22, U111, U112, U121, U122, U211, U212,U221, U222, J1, J2)

Htemp = 0;

for j = max([i, k, m]):n


    cmd = sprintf("Htemp = simplify(trace(U%d%d%d*J%d*U%d%d.')) + Htemp;" , j,k,m,j,j,i);
    eval(cmd);

end

Cout = Htemp;

end