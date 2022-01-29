function dydt = one_link2(t, y) % y : 각도, 각속도
global l lm m g r Fs Fv tau;

% 각가속도
% y(1) : 각도
% y(2) : 각속도

ddy = ( tau - m*g*r*sin(y(1)) - Fs*sign(y(2)) - Fv*y(2) ) / (l+lm);

dydt = [y(2); ddy];