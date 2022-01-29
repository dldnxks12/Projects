clc
clear all
close all

syms d dq ddq

global l lm m g r Fs Fv tau

r = 0.2;
m = 0.5;
g = 9.806;
l = 0.05;
lm = 0.05;

Fs = 0.1;
Fv = 0.1;

dt = 0.005; ft = 5;
q = pi/4; dq = 0;

% + Momentum ... Mass Inertia x 각속도 (M = Ia)

% W1 Integral
W1_int = [0 0 0 0];
theta = [0; 0; 0; 0]; % Regressor
%P = eye(4);
u = 0;

data = [];
time = [];
n = 2;

%%
for cnt=0:dt:ft

    tau = sin(cnt) + cos(10*cnt); % 무작위 값

    [t, y] = ode45('one_link2', [0 dt], [q; dq]);

    index = length(y);

    q = y(index, 1);
    dq = y(index, 2);

    W1_int = W1_int + [0 -g*sin(q) -sign(dq) -dq]*dt;
    W2 = [dq 0 0 0];

    %Y = W2 - W1_int;
    %u = u + tau*dt;

    % Error Minimization

    Y(n, :) = W2 - W1_int;
    u(n, :) = (u(n-1,:)) + (tau*dt);

    % regressor theta에 대한 1번 term, 2번 term
    ytemp1 = eye(4);
    ytemp2 = 0;

    for i=1:n
        ytemp1 = ytemp1 + Y(i,:).'*Y(i,:);
        ytemp2 = ytemp2 + Y(i,:).'*u(i,:);
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

plot(time(:, 1), data(:,1), 'r', 'linew', 2);
grid on
hold on
plot(time(:, 1), data(:,2), 'g', 'linew', 2);
plot(time(:, 1), data(:,3), 'b', 'linew', 2);
plot(time(:, 1), data(:,4), 'y', 'linew', 2);

legend({'l+lm', 'mr', 'Fs', 'Fv'}, 'location', 'best')
AX2.XLabel.String = 'time';
AX2.YLabel.String = 'Parameter';
AX2.FontSize = 12;
