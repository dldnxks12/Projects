function [J] = get_Jacobian2(th1, th2, th3)

global L1 L2 L3

X1 = -L1*sin(th1)-L2*sin(th1+th2)-L3*sin(th1+th2+th3);
X2 = -L2*sin(th1+th2)-L3*sin(th1+th2+th3);
X3 = -L3*sin(th1+th2+th3);

Y1 = L1*cos(th1)+L2*cos(th1+th2)+L3*cos(th1+th2+th3);
Y2 = L2*cos(th1+th2)+L3*cos(th1+th2+th3);
Y3 = L3*cos(th1+th2+th3);

Z1 = 0;
Z2 = 0;
Z3 = 0;

J = [X1 X2 X3; Y1 Y2 Y3; Z1 Z2 Z3];

end