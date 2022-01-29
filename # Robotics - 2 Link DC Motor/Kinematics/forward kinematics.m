clc;
clear;

%% Parameter
syms l1 l2 theta1 theta2

%% Transformation matrix

% Test
l1 = 0.5;
l2 = 0.45;
theta1 = 0*(pi/180);
theta2 = 90*(pi/180);

T01 = [ cos(theta1) -sin(theta1) 0 0; sin(theta1) cos(theta1) 0 0; 0 0 1 0; 0 0 0 1];

T12 = [ cos(theta2) -sin(theta2) 0 l1; sin(theta2) cos(theta2) 0 0; 0 0 1 0; 0 0 0 1];

T23 = [ 1 0 0 l2; 0 1 0 0; 0 0 1 0; 0 0 0 1];

T03 = T01*T12*T23;
