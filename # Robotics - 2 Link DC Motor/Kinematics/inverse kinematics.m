clc;
clear;

%% Parameter
syms l1 l2 theta1 theta2 x y

%% Transformation matrix

% Test
%l1 = 0.5;
%l2 = 0.45;
%x = 0.5;
%y = 0.45;
%theta1 = 0*(pi/180);
%theta2 = 90*(pi/180);

c2 = ( ((x^2) + (y^2)) - ((l1^2) + (l2^2)) ) / 2*l1*l2;
s2 = sqrt(1-(c2^2));

theta2 = atan2(s2, c2);

theta_2 = theta2*180/pi;

c1 = ( ((l1 + l2*c2)*x)+ ((l2*s2)*y) ) / ( (l1+l2*c2)^2 + (l2*s2)^2 );
s1 = ( -((l2*s2)*x) + ((l1+l2*c2)*y) ) / ( ((l1+l2*c2)^2) + ((l2*s2)^2) );

theta1 = atan2(s1,c2);
theta_1 = theta1*180/pi;
