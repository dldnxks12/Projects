function dydt = link2(t, y)
global tau1 tau2

% y(1) : q1
% y(2) : dq1
% y(3) : q2
% y(4) : dq2

%ddy1 = ((l2 + lm2)*(tau1 - Fv1*y(2) - Fs1*sign(y(2)) + g*m2*r2*cos(y(1) + y(3)) + g*l1*m2*cos(y(1)) + g*m1*r1*cos(y(1))))/(l1*l2 + l1*lm2 + l2*lm1 + l2*lm2 + lm1*lm2 + l1^2*l2*m2 + l1^2*lm2*m2 - l1^2*m2^2*r2^2*cos(y(3))^2 + 2*l1*lm2*m2*r2*cos(y(3))) - ((l2 + l1*m2*r2*cos(y(3)))*(tau2 - Fv2*y(4) - Fs2*sign(y(4)) + g*m2*r2*cos(y(1) + y(3))))/(l1*l2 + l1*lm2 + l2*lm1 + l2*lm2 + lm1*lm2 + l1^2*l2*m2 + l1^2*lm2*m2 - l1^2*m2^2*r2^2*cos(y(3))^2 + 2*l1*lm2*m2*r2*cos(y(3)));
%ddy2 = ((tau2 - Fv2*y(4) - Fs2*sign(y(4)) + g*m2*r2*cos(y(1) + y(3)))*(l1 + l2 + lm1 + l1^2*m2 + 2*l1*m2*r2*cos(y(3))))/(l1*l2 + l1*lm2 + l2*lm1 + l2*lm2 + lm1*lm2 + l1^2*l2*m2 + l1^2*lm2*m2 - l1^2*m2^2*r2^2*cos(y(3))^2 + 2*l1*lm2*m2*r2*cos(y(3))) - ((l2 + l1*m2*r2*cos(y(3)))*(tau1 - Fv1*y(2) - Fs1*sign(y(2)) + g*m2*r2*cos(y(1) + y(3)) + g*l1*m2*cos(y(1)) + g*m1*r1*cos(y(1))))/(l1*l2 + l1*lm2 + l2*lm1 + l2*lm2 + lm1*lm2 + l1^2*l2*m2 + l1^2*lm2*m2 - l1^2*m2^2*r2^2*cos(y(3))^2 + 2*l1*lm2*m2*r2*cos(y(3)));

ddy1_ =  (1000*(tau1 - y(2)/10 + (4903*cos(y(1) + y(3)))/25000 + (14709*cos(y(1)))/12500 - sign(y(2))/10 + (y(2)*y(4)*sin(y(3)))/100 + (y(4)*sin(y(3))*(y(2) + y(4)))/100))/(10*cos(y(3)) - cos(y(3))^2 + 175) + (100*(cos(y(3)) + 5)*((sin(y(3))*y(2)^2)/100 + y(4)/10 - tau2 - (4903*cos(y(1) + y(3)))/25000 + sign(y(4))/10))/(10*cos(y(3)) - cos(y(3))^2 + 175);
ddy2_ = - (200*(cos(y(3)) + 10)*((sin(y(3))*y(2)^2)/100 + y(4)/10 - tau2 - (4903*cos(y(1) + y(3)))/25000 + sign(y(4))/10))/(10*cos(y(3)) - cos(y(3))^2 + 175) - (100*(cos(y(3)) + 5)*(tau1 - y(2)/10 + (4903*cos(y(1) + y(3)))/25000 + (14709*cos(y(1)))/12500 - sign(y(2))/10 + (y(2)*y(4)*sin(y(3)))/100 + (y(4)*sin(y(3))*(y(2) + y(4)))/100))/(10*cos(y(3)) - cos(y(3))^2 + 175);
dydt = [y(2); ddy1_; y(4); ddy2_];