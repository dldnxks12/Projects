%% 2014741002 로봇학부 이종수 Robot Control Term Project
clc;
clear all
close all

DR = deg2rad(1);
RD = rad2deg(1);

%% DH Parameter

syms L1 L2 m1 m2 lc1 lc2 lm1 lm2 r1 r2 lz1 lz2
syms th1 th2 dth1 dth2 ddth1 ddth2

l1xx = 0;
l1yy = lz1; % lz1 = lc1 + m1r1^2 --- Parallel Axis Theorem
l1zz = lz1;

l2xx = 0;
l2yy = lz2;
l2zz = lz2;

d1 = 0; d2 = 0; % Link Offset
a1 = L1; a2 = L2; % Link Length
al1 = 0; al2 = 0; % Link Twist


% th1 th2 : Joint Angle

DH = [ th1, d1, a1, al1;
       th2, d2, a2, al2 ];

%% Homogenous Transformation Matrix

T01 = HT(th1, d1, a1, al1);
T12 = HT(th2, d2, a2, al2);
T02 = T01*T12;


%% Q Matrix (미분 행렬)

% r0i의 미분 V0i를 구하기 위해 Homogenous Transformation Matrix의 미분이 필요
% d/dt(r0i) = d/dt(T0i * rii) ....
Qr = [0 -1 0 0;
      1 0 0 0;
      0 0 0 0;
      0 0 0 0 ];

Q1 = Qr;
Q2 = Qr;

%% U Matrix

% U ij
U11 = Q1*T01;
U12 = zeros(4,4);

U21 = Q1*T02;
U22 = T01*Q2*T12;
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


%% Motion of Equation - Inertia Matrix (D)

% Dik --- tau i = summation(k=1 ~ k=n) D ik
D11 = trace(U11*J1*U11.')...
    + trace(U21*J2*U21.');

D12 = trace(U22*J2*U21.');

D21 = trace(U21*J2*U22.');

D22 = trace(U22*J2*U22.');

M11 = simplify(D11); % simplify --- 정리해주는 함수
M12 = simplify(D12); % ok
M21 = simplify(D21); % ok
M22 = simplify(D22);


M = [M11 M12 ;
     M21 M22 ];
%% Motion of Equation - Inertia Matrix (D) - Function

n = 2;
for i = 1:n
    for k = 1:n
        nM(i,k) = Inertia2(i, k, n, U11, U12, U21, U22, J1, J2);
    end
end

%% Get Uijk matrix
n = 2;
for i = 1:n
    for j = 1:n
        for k = 1:n
            cmd = sprintf("U%d%d%d = dUdq2(i,j,k,T01,T12,T02, Q1, Q2);", i,j,k);
            eval(cmd);
        end
    end
end

%% 2 DOF Coriolis & Centrifugal

n = 2;
for i = 1:n
    for k = 1:n
        for m = 1:n
            cmd = sprintf("h%d%d%d = CC2(i, k, m, n, U11, U12, U21, U22, U111, U112, U121, U122, U211, U212,U221, U222, J1, J2);" ,  i, k, m);
            eval(cmd);
        end
    end
end

%% Check H of 2DOF

h1 = (dth1^2)*h111 + (dth1*dth2)*(h112 + h121) + (dth2^2)*h122;
h2 = (dth1^2)*h211 + (dth1*dth2)*(h212 + h221) + (dth2^2)*h222;

h = simplify([h1; h2]);

%% Test - ok

L2 = L1;
r1 = L1/2;
r2 = L2/2;

lz1 = 1/3*m1*L1^2;
lz2 = 1/3*m2*L2^2;

sh1 = simplify(eval(h1));
sh2 = simplify(eval(h2));

%% Gravity Matrix

syms g

% ith 번째 링크에서 본 i번 째 링크의 COG 까지 거리
r11 = [-(L1-r1); 0; 0; 1];
r22 = [-(L2-r2); 0; 0; 1];
gv = [0 -g 0 0];

G1 = -(m1*gv*U11*r11 + ...
    m2*gv*U21*r22 );

G2 = -(m2*gv*U22*r22);

G = simplify([G1; G2]);

%% Simulation

syms tau1 tau2
DDTH = inv(M)*([tau1; tau2] - h - G);

dydt = simplify([dth1; DDTH(1); dth2; DDTH(2)]);

% dydt 변수를 file로 저장할 수 있다.
matlabFunction(dydt, 'file', 'two_link.m', 'Optimize', false);

%% Simulation Start

clear all
close all

global lz1 lz2 L1 L2 g m1 m2 r1 r2 tau1 tau2

L1 = 0.5; L2 = 0.5; L3 = 0.5;
r1 = 0.1; r2 = 0.1; r3 = 0.1;
m1 = 0.2; m2 = 0.2; m3 = 0.2;

lz1 = 0.05; lz2 = 0.05; lz3 = 0.05;

g = 9.806;

dt = 0.005; ft = 5;

% 초기 각도, 각속도
q1 = -pi/2; dq1 = 0;
q2 = pi/2; dq2 = 0;

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

p1 = plot(Px1, Py1, '-ob', 'LineWidth', 3);
p2 = plot(Px2, Py2, '-ob', 'LineWidth', 3);

for cnt=0:dt:ft

    tau1 = 0.0;
    tau2 = 0.0;

    % 시간과 State를 넘겨줌
    [t, y] = ode45('two_link',[0 dt], [q1; dq1; q2; dq2]);

    index = length(y);

    q1 = y(index, 1);
    dq1 = y(index, 2);
    q2 = y(index, 3);
    dq2 = y(index,4);

    x1 = L1*cos(q1);
    y1 = L1*sin(q1);
    Px1 = [0 x1];
    Py1 = [0 y1];

    x2 = L2*cos(q1+q2);
    y2 = L2*sin(q1+q2);
    Px2 = [x1 x1+x2];
    Py2 = [y1 y1+y2];

    n = n+1;

    cmd = sprintf('Time : %2.2f', cnt);

    clc
    disp(cmd)
    if rem(n, 10) == 0
        set(p1, 'xData', Px1, 'yData', Py1)
        set(p2, 'xData', Px2, 'yData', Py2)
        drawnow
    end
end











