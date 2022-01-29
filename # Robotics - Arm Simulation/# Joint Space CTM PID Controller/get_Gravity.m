function G = get_Gravity(theta)
global m l g L;

G = m*L*g*cos(theta);
% G = l*g/(L*sin(theta));
