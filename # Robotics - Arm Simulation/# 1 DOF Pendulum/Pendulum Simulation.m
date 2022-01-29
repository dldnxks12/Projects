clc
clear all
close all

global m l g u;
% m : mass , l : length of pendulum, g : graivity , u : control input

m = 1; l = 1; g = 9.8148; n = 1; u = 0;

dt = 0.005;
ft = 5;
q = pi/4; % 45도
dq = 0;

data = [];

%% Make Figure
FG = figure('Position', [300 300 600 600], 'Color', [ 1 1 1]);
AX = axes('parent', FG);

hold on
grid on
axis([-1.5 1.5 -1.5 1.5])

%% Plot
Px = [0,1];
Py = [0,0];

p = plot(Px, Py, 'Linewidth', 3, 'Color', 'b')'


%% ODE45

for cnt = 0:dt:ft % 0~5 까지 dt만큼

    % 해당 시간의 dq와 ddq를 적분해서 q, dq를 return
    [t,y] = ode45('one_link', [0 dt], [q; dq]);

    index = length(y);

    q = y(index, 1);
    dq = y(index, 2);

    % Forward Kinematics
    x = l*sin(q);
    y = -l*cos(q);

    Px = [0, x];
    Py = [0, y];

    data(n, 1) = cnt;
    data(n, 2) = q;
    data(n, 3) = dq;
    n = n+1;

    cmd = sprintf("Time : %2.2f", cnt);
    clc
    disp(cmd)

    if rem(n, 10) == 0
        set(p, 'xData', Px, 'yData', Py)
        drawnow
    end
end

%%

FG2 = figure('Position', [300 300 600 300] , 'Color', [1 1 1]);
AX2 = axes('parent', FG2);

grid on
plot(data(:, 1), data(:, 2), 'r') % 시간에 따른 Q
hold on
plot(data(:, 1), data(:, 3), 'b') % 시간에 따른 dQ
title('1 DOF Pendulum');
legend({'Angle', 'Angular Vel'})














