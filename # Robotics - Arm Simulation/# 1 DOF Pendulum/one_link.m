function dydt = one_link(t, y)
global m l g u;

dydt = [y(2); -g/l*sin(y(1)) + u*(1/m*l*l)];