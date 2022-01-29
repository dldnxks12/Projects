%% 2014741002 로봇학부 이종수 Robot Control Term Project
clc;
clear all
close all

DR = deg2rad(1);
RD = rad2deg(1);

%% DH Parameter

syms L1 L2 L3 m1 m2 m3 lc1 lc2 lc3 lm1 lm2 lm3 r1 r2 r3 lz1 lz2 lz3
syms th1 th2 th3 dth1 dth2 dth3 ddth1 ddth2 ddth3

l1xx = 0;
l1yy = lz1; % lz1 = lc1 + m1r1^2 --- Parallel Axis Theorem
l1zz = lz1;

l2xx = 0;
l2yy = lz2;
l2zz = lz2;

% For 3 DOF Arm
l3xx = 0;
l3yy = lz3;
l3zz = lz3;

d1 = 0; d2 = 0; d3 = 0; % Link Offset
a1 = L1; a2 = L2; a3 = L3;% Link Length
al1 = 0; al2 = 0; al3 = 0;% Link Twist

DH = [ th1, d1, a1, al1;
       th2, d2, a2, al2;
       th3, d3, a3, al3 ];

%% Homogenous Transformation Matrix

T01 = HT(th1, d1, a1, al1);
T12 = HT(th2, d2, a2, al2);
T23 = HT(th3, d3, a3, al3);

T02 = T01*T12;
T03 = T02*T23;
T13 = T12*T23;

%% Q Matrix (미분 행렬)

% r0i의 미분 V0i를 구하기 위해 Homogenous Transformation Matrix의 미분이 필요
% d/dt(r0i) = d/dt(T0i * rii) ....
Qr = [0 -1 0 0;
      1 0 0 0;
      0 0 0 0;
      0 0 0 0 ];

Q1 = Qr;
Q2 = Qr;
Q3 = Qr;

% Classical DH 에서는 Q Matrix를 앞에 곱
%dT01t1 = Qr*T01; % or We can use dT01dt = diff(Tq1)

%% U Matrix

% U ij
U11 = Q1*T01;
U12 = zeros(4,4);

U21 = Q1*T02;
U22 = T01*Q2*T12;

% For 3 DOF Arm

U13 = zeros(4,4);
U23 = zeros(4,4);

U31 = Q1*T03;
U32 = T01*Q2*T13;
U33 = T02*Q3*T23;

%% Pseudo Inertia Matrix (J)

% X bari : ith 좌표계에서본 COG

% J1 Matrix
J1(1,1) = 1/2*(-l1xx + l1yy + l1zz);
J1(2,2) = 1/2*(l1xx - l1yy + l1zz);
J1(3,3) = 1/2*(l1xx + l1yy - l1zz);
J1(4,4) = m1;

J1(1,4) = -m1 * (L1 - r1);
J1(4,1) = J1(1,4);

% J2 Matrix

J2(1,1) = 1/2*(-l2xx + l2yy + l2zz);
J2(2,2) = 1/2*(l2xx - l2yy + l2zz);
J2(3,3) = 1/2*(l2xx + l2yy - l2zz);
J2(4,4) = m2;

J2(1,4) = -m2 * (L2 - r2);
J2(4,1) = J2(1,4);

% J3 Matrix for 3 DOF Arm

J3(1,1) = 1/2*(-l3xx + l3yy + l3zz);
J3(2,2) = 1/2*(l3xx - l3yy + l3zz);
J3(3,3) = 1/2*(l3xx + l3yy - l3zz);
J3(4,4) = m3;

J3(1,4) = -m3 * (L3 - r3);
J3(4,1) = J3(1,4);

%% Motion of Equation - Inertia Matrix (D)

% Dik --- tau i = summation(k=1 ~ k=n) D ik
D11 = trace(U11*J1*U11.')...
    + trace(U21*J2*U21.')...
    + trace(U31*J3*U31.');

D12 = trace(U22*J2*U21.')...
    + trace(U32*J3*U31.');

D13 = trace(U33*J3*U31.');

D21 = trace(U21*J2*U22.')...
    + trace(U31*J3*U32.');

D22 = trace(U22*J2*U22.')...
    + trace(U32*J3*U32.');

D23 = trace(U33*J3*U32.');

D31 = trace(U31*J3*U33.');
D32 = trace(U32*J3*U33.');
D33 = trace(U33*J3*U33.');

M11 = simplify(D11); % simplify --- 정리해주는 함수
M12 = simplify(D12); % ok
M13 = simplify(D13); % ok
M21 = simplify(D21); % ok
M22 = simplify(D22);
M23 = simplify(D23); % ok

M31 = simplify(D31); % ok
M32 = simplify(D32); % ok
M33 = simplify(D33);

M = [M11 M12 M13;
     M21 M22 M23;
     M31 M32 M33];
%% Motion of Equation - Inertia Matrix (D) - Function

n = 3;
for i = 1:n
    for k = 1:n
        nM(i,k) = Inertia(i, k, n, U11, U12, U13, U21, U22, U23, U31, U32, U33, J1, J2, J3);
    end
end

%% Get Uijk matrix
n = 3;
for i = 1:n
    for j = 1:n
        for k = 1:n
            cmd = sprintf("U%d%d%d = dUdq(i,j,k,T01,T12,T23,T02,T03,T13, Q1, Q2, Q3);", i,j,k);
            eval(cmd);
        end
    end
end

%% Coriolis & Centrifugal Matrix  (H)

n = 3;
for i = 1:n
    for k = 1:n
        for m = 1:n
            cmd = sprintf("h%d%d%d = CC(i, k, m, n, U11, U12, U13, U21, U22, U23, U31, U32, U33,U111, U112, U113, U121, U122, U123, U131, U132, U133, U211, U212, U213, U221, U222, U223, U231, U232, U233, U311, U312, U313, U321, U322, U323, U331, U332, U333, J1, J2, J3);" ,  i, k, m);
            eval(cmd);
        end
    end
end

%% H Matrix of 3 DOF Arm

h1 = h111*dth1*dth1 + h112*dth1*dth2 + h113*dth1*dth3 + h121*dth2*dth1 + h122*dth2*dth2 + h123*dth2*dth3 + h131*dth3*dth1 + h132*dth3*dth2 + h133*dth3*dth3;
h2 = h211*dth1*dth1 + h212*dth1*dth2 + h213*dth1*dth3 + h221*dth2*dth1 + h222*dth2*dth2 + h223*dth2*dth3 + h231*dth3*dth1 + h232*dth3*dth2 + h233*dth3*dth3;
h3 = h311*dth1*dth1 + h312*dth1*dth2 + h313*dth1*dth3 + h321*dth2*dth1 + h322*dth2*dth2 + h323*dth2*dth3 + h331*dth3*dth1 + h332*dth3*dth2 + h333*dth3*dth3;

h = simplify([h1; h2; h3]);

%% Gravity Matrix

syms g

% ith 번째 링크에서 본 i번 째 링크의 COG 까지 거리
r11 = [-(L1-r1); 0; 0; 1];
r22 = [-(L2-r2); 0; 0; 1];
r33 = [-(L3-r3); 0; 0; 1];
gv = [0 -g 0 0];

G1 = -(m1*gv*U11*r11 + ...
    m2*gv*U21*r22 + ...
    m3*gv*U31*r33);

G2 = -(m2*gv*U22*r22 + ...
    m3*gv*U32*r33);

G3 = -(m3*gv*U33*r33);

G = simplify([G1; G2; G3]);


%% Simulation

syms tau1 tau2 tau3

DDTH = inv(M)*([tau1; tau2; tau3] - h - G);

%%
dydt = simplify([dth1; DDTH(1); dth2; DDTH(2); dth3; DDTH(3)]);

% dydt 변수를 file로 저장할 수 있다.
matlabFunction(dydt, 'file', 'three_link2.m', 'Optimize', true);

%% Simulation Start

clear all
close all

global lz1 lz2 lz3 L1 L2 L3 g m1 m2 m3 r1 r2 r3 tau1 tau2 tau3

L1 = 0.5; L2 = 0.5; L3 = 0.5;
r1 = 0.1; r2 = 0.1; r3 = 0.1;
m1 = 0.2; m2 = 0.2; m3 = 0.2;

lz1 = 0.05; lz2 = 0.05; lz3 = 0.05;

g = 9.806;

dt = 0.005; ft = 15;

% 초기 각도, 각속도
q1 = -pi/2; dq1 = 0;
q2 = pi/2; dq2 = 0;
q3 = pi/4; dq3 = 0;

data = [];

n = 1;

FG = figure('Position', [300 300 400 400], 'Color', [1 1 1]);
AX = axes('parent', FG);

hold on
grid on

axis([-1.5 1.5 -1.5 1.5]);

x1 = L1*cos(q1);
y1 = L1*sin(q1);
Px1 = [0 x1];
Py1 = [0 y1];

x2 = L2*cos(q1+q2);
y2 = L2*sin(q1+q2);
Px2 = [x1 x1+x2];
Py2 = [y1 y1+y2];

x3 = L3*cos(q1+q2+q3);
y3 = L3*sin(q1+q2+q3);
Px3 = [x1+x2 x1+x2+x3];
Py3 = [y1+y2 y1+y2+y3];

p1 = plot(Px1, Py1, '-ob', 'LineWidth', 3);
p2 = plot(Px2, Py2, '-og', 'LineWidth', 3);
p3 = plot(Px3, Py3, '-or', 'LineWidth', 3);

for cnt=0:dt:ft

    tau1 = 0.0;
    tau2 = 0.0;
    tau3 = 0.0;

    % 시간과 State를 넘겨줌
    [t, y] = ode45('three_link2', [0 dt], [q1; dq1; q2; dq2; q3; dq3]);

    index = length(y);

    q1 = y(index, 1);
    dq1 = y(index, 2);
    q2 = y(index, 3);
    dq2 = y(index,4);
    q3 = y(index, 5);
    dq3 = y(index,6);

    x1 = L1*cos(q1);
    y1 = L1*sin(q1);
    Px1 = [0 x1];
    Py1 = [0 y1];

    x2 = L2*cos(q1+q2);
    y2 = L2*sin(q1+q2);
    Px2 = [x1 x1+x2];
    Py2 = [y1 y1+y2];

    x3 = L3*cos(q1+q2+q3);
    y3 = L3*sin(q1+q2+q3);
    Px3 = [x1+x2 x1+x2+x3];
    Py3 = [y1+y2 y1+y2+y3];

    n = n+1;

    cmd = sprintf('Time : %2.2f', cnt);
    clc
    disp(cmd)
    if rem(n, 10) == 0
        set(p1, 'XData', Px1, 'YData', Py1)
        set(p2, 'XData', Px2, 'YData', Py2)
        set(p3, 'XData', Px3, 'YData', Py3)
        drawnow
    end
end
