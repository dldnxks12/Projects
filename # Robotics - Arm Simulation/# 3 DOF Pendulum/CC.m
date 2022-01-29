function C = CC(i, k, m, n, U11, U12, U13, U21, U22, U23, U31, U32, U33,U111, U112, U113, U121, U122, U123, U131, U132, U133, U211, U212, U213, U221, U222, U223, U231, U232, U233, U311, U312, U313, U321, U322, U323, U331, U332, U333, J1, J2, J3)

CTemp = 0;

for j = max([i, k, m]):n

    cmd = sprintf("CTemp = simplify(trace(U%d%d%d*J%d*U%d%d.')) + CTemp;" , j,k,m,j,j,i);
    eval(cmd);
end

C = CTemp;
end