function [J] = get_Jacobian(th1, th2)

global L1 L2

X1 = -L1*sin(th1) -L2*sin(th1 + th2);
X2 = -L2*sin(th1 + th2);

Y1 = L1*cos(th1) + L2*cos(th1 + th2);
Y2 = L2*cos(th1 + th2);


J = [X1 X2; Y1 Y2];

end