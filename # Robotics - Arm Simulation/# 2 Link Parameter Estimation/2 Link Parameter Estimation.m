clc
clear all
close all
%%
syms q1 dq1 ddq1 q2 dq2 ddq2
global l1 l2 lm1 lm2 m1 m2 r1 r2 Fs1 Fs2 Fv1 Fv2 g tau1 tau2 i1 i2

l1 = 0.05;
l2 = l1;

i1 = 0.5;
i2 = i1;

lm1 = 0.05;
lm2 = lm1;

m1 = 0.2;
m2 = m1;

g = 9.806;

Fs1 = 0.1;
Fs2 = Fs1;

Fv1 = 0.1;
Fv2 = Fv1;

r1 = 0.1;
r2 = r1;

M = [l1+l2+(m2*(i1^2))+2*m2*r2*i1*cos(q2)+lm1 l2+m2*r2*i1*cos(q2); l2+m2*r2*i1*cos(q2) l2+lm2];
C = [-m2*r2*i1*sin(q2)*dq2 -m2*r2*i1*sin(q2)*(dq1+dq2); m2*r2*i1*sin(q2)*dq1 0];
G = [-m1*r1*g*cos(q1)-m2*i1*g*cos(q1)-m2*r2*g*cos(q1+q2); -m2*r2*g*cos(q1+q2)];
D = [Fs1*sign(dq1)+Fv1*dq1; Fs2*sign(dq2)+Fv2*dq2];

test_theta = [l1+(m2*(i1^2))+lm1; l2; m2*r2*i1; m1*r1+m2*i1; m2*r2; lm2; Fs1; Fs2; Fv1; Fv2];


%% W1_Int*theta = C_transpose*dq - G - D

W1_theta = C.'*[dq1; dq2] - G - D;
W1_int = [0, 0, 0, g*cos(q1), g*cos(q1+q2), 0, -sign(dq1), 0, -dq1, 0; 0, 0, -sin(q2)*(dq1+dq2)*dq1, 0, g*cos(q1+q2), 0, 0, -sign(dq2), 0, -dq2];

Check1 = simplify(W1_theta - (W1_int*test_theta));
%% W2

W2_theta = M*[dq1; dq2];
W2 = [dq1, dq1+dq2, 2*cos(q2)*dq1+cos(q2)*dq2, 0, 0, 0, 0, 0, 0, 0; 0, dq1+dq2, cos(q2)*dq1, 0, 0, dq2, 0, 0, 0, 0];

Check2 = simplify(W2_theta - (W2*test_theta));

%% Find tau1, tau2 for ode45

syms tau1 tau2
DDTH = inv(M)*([tau1; tau2] - C*[dq1; dq2] - G - D);

%%

dt = 0.005; ft = 5;

q1 = pi/4; dq1 = 0;
q2 = pi/4; dq2 = 0;

W1 = [0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0];
theta = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0]; % Regressor --- 추정할 파라미터들

Y = zeros(1, 2 , 10);
u = zeros(1, 2, 1);
T = zeros(1, 2, 1);

data = [];
time = [];
n = 2;

for cnt=0:dt:ft

    % 무작위 입력 값
    tau1 = sin(cnt) + cos(10*cnt);
    tau2 = cos(cnt) + sin(10*cnt);

    [t, y] = ode45('link2', [0 dt], [q1; dq1; q2; dq2]);

    index = length(y);

    q1 = y(index, 1);
    dq1 = y(index, 2);
    q2 = y(index, 3);
    dq2 = y(index, 4);

    W1_int = [0, 0, 0, g*cos(q1), g*cos(q1+q2), 0, -sign(dq1), 0, -dq1, 0; 0, 0, -sin(q2)*(dq1+dq2)*dq1, 0, g*cos(q1+q2), 0, 0, -sign(dq2), 0, -dq2];

    W1 = W1 + W1_int*dt;
    W2 = [dq1, dq1+dq2, 2*cos(q2)*dq1+cos(q2)*dq2, 0, 0, 0, 0, 0, 0, 0; 0, dq1+dq2, cos(q2)*dq1, 0, 0, dq2, 0, 0, 0, 0];

    % Error Minimization

   % Y(n, :, :) = W2 - W1;
   % u(n, :) = u(n-1, :) + [tau1; tau2]*dt;
   % ytemp1 = 0;
   % ytemp2 = 0;

   % for i=1:n
   %     K = squeeze(Y(i, :, :));
   %     ytemp1 = ytemp1 + K.'*K; %
   %     ytemp2 = ytemp2 + K.'*u(i,:);
   % end

    T(1, 1, 1) = tau1*dt;
    T(1, 2, 1) = tau2*dt;

    Y(n, :, :) = W2 - W1;
    u(n, :, :) = u(n-1, :, :) + T;
    ytemp1 = 0;
    ytemp2 = 0;

    for i=1:n
        K = squeeze(Y(i, :, :));
        ytemp1 = ytemp1 + K.'*K;
        ytemp2 = ytemp2 + K.'*u(i,:,:).';
    end
    theta = (inv(ytemp1)*ytemp2);

    time(n, :) = cnt;
    data(n, :) = theta;

    n = n+1;

    cmd = sprintf("Time : %2.2f", cnt);
    clc
    disp(cmd)

end

%%

FG2 = figure('Position', [300 300 600 300], 'Color', [1 1 1]);
AX2 = axes('parent', FG2);

plot(time(:, 1), data(:,2), 'r', 'linew', 2);

AX2.XLabel.String = 'time';
AX2.YLabel.String = 'Parameter';
AX2.FontSize = 12;
