%%
clc
clear all
close all

global lz1 lz2 L1 L2 g m1 m2 r1 r2 tau1 tau2;

% Simulation Parameters
delta_t  = 0.005;     % [sec] : Sampling time
start_t  = 0.000;     % [sec] : Start time
finish_t = 5.000;     % [sec] : End time

g        = 9.8148;    % [m/s^2] : Gravitational Acceleration

% Roobot Parameters

m1       = 0.2000;      % [kg]    : Link Mass
m2       = 0.2000;      % [kg]    : Link Mass
r1       = 0.10;
r2       = 0.10;
L1       = 0.5000;      % [m]     : Link Length
L2       = 0.5000;      % [m]     : Link Length

lz1       = 0.05;   % [kgm^2] : Link Inertia
lz2       = 0.05;   % [kgm^2] : Link Inertia

tau1     = 0.0000;      % [Nm]    : Control Torque
tau2     = 0.0000;      % [Nm]    : Control Torque
tau      = [tau1
            tau2]; % [Nm]    : Control Torque

init_q1  = -pi/2;       % [rad]    : Initial Joint Angle
init_q2  = pi/2;        % [rad]    : Initial Joint Angle
init_dq1 = 0.00;        % [rad/s]  : Initial Joint Angular Vel
init_dq2 = 0.00;        % [rad/s]  : Initial Joint Angular Vel

q       = [init_q1
           init_q2];       % [rad]    : Joint Angle
dq      = [init_dq1
           init_dq2];     % [rad/s]  : Joint Angular Vel

init_X = get_Kinematics(q(1), q(2)); % [m] : Init End-Effector Position
X     = init_X;                      % [m] : Current Position
dX    = [0; 0];                      % [m] : Current Velocity
X_d   = init_X;                      % [m] : Target End-Effector Position
dX_d  = [0; 0];                      % [m] : Target End-Effector Velocity
ddX_d = [0; 0];                      % [m] : Target End-Effector Acceleration

% Contoller Gain

Wn    = 60; % 20 -> 60 good -> 120 better
Kp    = Wn^2;              % P Gain
Kv    = 2*Wn;              % D Gain
Ki    = Kp*Kv*0.005;         % I Gain --- Ki < Kp*Kv !!

%% Simulation

n = 1;
sin_t = 0;
pre_J = 0;
pre_X = 0;
Integral_temp = 0;
for time = start_t:delta_t:finish_t
    % Robot Model
        % Set Trajectory 45에서 시작 -> 2초에 걸쳐 45도를 이동하여 90도로
        if time < 1
            X_d   = init_X;
            dX_d  = [0; 0];
            ddX_d = [0; 0];

        % 반경 0,1m 주기 1초 원 만들기 준비 --- 시작 위치로 올리기
        elseif time < 2.0
            X_d(1) = init_X(1);                    % x 위치는 보정 불필요
            if X_d(2) < init_X(2) + 0.1              % 1m 만큼 올라가기 때문에 1 보정
                X_d(2) = X_d(2) + (0.1/0.5)*delta_t; % (0.2/0.5)*delta_t : Target Velocity 0.2의 거리를 0.5초 만에 이동 --- 즉, 0.4 m/s X 적분 --- 0.4 * dt --- 위치 보정 !
            else
                X_d(2) = init_X(2) + 0.1;            % 목표 위치에 도달한다면 멈춰라
            end
            dX_d  = (X_d  - [simul_X_d_x(n-1) ; simul_X_d_y(n-1)])./delta_t;
            ddX_d = (dX_d - [simul_dX_d_x(n-1); simul_dX_d_y(n-1)])./delta_t;

        else % 1초의 주기로 1바퀴 돈다.
            % sin_t : 0.5초
            % sin(2*pi*t) : 주기 T
            % sin(2*pi*sin_t/2)  --- 2초에 1바퀴 (4 tick에 1바퀴)
            % sin(2*pi*sin_t)    --- 1초에 1바퀴
            % sin(2*pi*sin_t*2)  --- 0.5초에 1바퀴
            X_d   = [0.1 * sin((2*pi*sin_t)) + init_X(1);        % 초기 X위치 : 0.1 + Initial X
                     0.1 * cos((2*pi*sin_t)) + init_X(2);];      % 초기 Y위치 : 0.2 + Initial Y
            sin_t = sin_t + delta_t;

            dX_d  = (X_d  - [simul_X_d_x(n-1) ; simul_X_d_y(n-1)])./delta_t;
            ddX_d = (dX_d - [simul_dX_d_x(n-1); simul_dX_d_y(n-1)])./delta_t;

        end
        % Dynamics
        J     = get_Jacobian(q(1), q(2));     % 자코비안을 구해오는 코드 구현
        dJ    = (J - pre_J)/delta_t;          % Jacobian 미분
        pre_J = J;                            % 이전 Jacobian
        X     = get_Kinematics(q(1), q(2));   % Current Position
        dX    = J*dq;                         % Velocity

        % Mass Inertia Term
        D     =  [lz1 + lz2 - L1^2*m1 + L1^2*m2 - L2^2*m2 + 2*L1*m1*r1 + 2*L2*m2*r2 + 2*L1*m2*r2*cos(q(2)), - m2*L2^2 + 2*m2*r2*L2 + lz2 + L1*m2*r2*cos(q(2));
                                       - m2*L2^2 + 2*m2*r2*L2 + lz2 + L1*m2*r2*cos(q(2)),                     - m2*L2^2 + 2*m2*r2*L2 + lz2];

        H     = [-L1*dq(2)*m2*r2*sin(q(2))*(2*dq(1)+dq(2)); L1*dq(1)^2*m2*r2*sin(q(2))];                        % Corilois Term
        C     = [ g*(m1*r1*cos(q(1)) + m2*r2*cos(q(1) + q(2))+L1*m2*cos(q(1))); g*m2*r2*cos(q(1) + q(2))];      % Gravity Term


        % Control Torque
        Integral_temp = Integral_temp + (X_d - X)*delta_t;           % 위치 오차 값을 누적시키는 부분
        u = ddX_d + Kv*(dX_d - dX) + Kp*(X_d- X) + Ki*Integral_temp; % P I D
        ddq_ref = inv(J)*(u - dJ*dq); % 2 x 1

        tq_control = D*ddq_ref + H + C*0.9;

            % Inverse Dynamics
            tau = tq_control;
            tau1 = tau(1);
            tau2 = tau(2);
            [t, y] = ode45('two_link', [0 delta_t], [q(1); dq(1); q(2); dq(2);]);
            index = length(y);

            q  = [y(index, 1); y(index, 3)];
            dq = [y(index, 2); y(index, 4)];

    % Save Data
    simul_time(n)    = time;      % [sec]
    simul_q1(n)       = q(1);     % [rad]
    simul_q2(n)       = q(2);     % [rad]
    simul_dq1(n)      = dq(1);    % [rad/s]
    simul_dq2(n)      = dq(2);    % [rad/s]

    simul_X_x(n)        = X(1);        % [m]
    simul_X_y(n)        = X(2);        % [m]
    simul_dX_x(n)       = dX(1);       % [m/s]
    simul_dX_y(n)       = dX(2);       % [m/s]

    simul_X_d_x(n)      = X_d(1);      % [m]  --- Target Position
    simul_X_d_y(n)      = X_d(2);      % [m]  --- Target Position
    simul_dX_d_x(n)     = dX_d(1);     % [m/s]  --- Target Velocity
    simul_dX_d_y(n)     = dX_d(2);     % [m/s]  --- Target Velocity

    n = n+1;
end

font_size_label   = 20;
font_size_title   = 25;

linewidth_current = 3;
linewidth_target  = 5;

FG1 = figure('Position', [200 300 700 700], 'Color', [1 1 1]);
AX = axes('parent', FG1);

hold on
grid on

axis([-1.0 1.0 -1.6 0.4]);

x1 = L1*cos(init_q1);                    % [m] : Joint 1 X-axis Position
y1 = L1*sin(init_q1);                    % [m] : Joint 1 Y-axis Position
x2 = L2*cos(init_q1+init_q2);            % [m] : Joint 2 X-axis Position
y2 = L2*sin(init_q1+init_q2);            % [m] : Joint 2 Y-axis Position

Px1 = [0 x1];
Py1 = [0 y1];
Px2 = [x1 x1+x2];
Py2 = [y1 y1+y2];

p1 = plot(Px1, Py1, '-ob', 'Linewidth', linewidth_current);
p2 = plot(Px2, Py2, '-or', 'Linewidth', linewidth_current);


xlabel('X-axis (m)', 'fontsize', font_size_label)
ylabel('Y-axis (m)', 'fontsize', font_size_label)
title('2 DOF Robot', 'fontsize', font_size_title)


% Draw Robot !
n = 1;
for time = start_t:delta_t:finish_t
    q1 = simul_q1(n);
    q2 = simul_q2(n);

    x1 = L1*cos(q1);
    y1 = L1*sin(q1);
    x2 = L2*cos(q1+q2);
    y2 = L2*sin(q1+q2);

    Px1 = [0 x1]; Py1 = [0 y1];
    Px2 = [x1 x1+x2]; Py2 = [y1 y1+y2];

    set(p1, 'XData', Px1, 'YData', Py1);
    set(p2, 'XData', Px2, 'YData', Py2);
    drawnow
    n = n+1;
end

%% Draw Datas

% Draw Angle
FG2 = figure('Position', [900 700 600 300], 'Color', [1 1 1]);
plot(simul_time, simul_X_d_x, ':r', 'linewidth', 7); hold on % Current Data
plot(simul_time, simul_X_d_y, ':b', 'linewidth', 7); hold on

plot(simul_time, simul_X_x, 'k', 'linewidth', 3); hold on   % Target Data
plot(simul_time, simul_X_y, 'g', 'linewidth', 3); hold on

axis([start_t finish_t -1.25 1]);
xticks([start_t : 1 : finish_t]);
yticks([-0.8 : 0.1 : 0.8]);
grid on

legend({'tar_x', 'tar_y', 'cur_x', 'cur_y'}, 'location', 'best', 'orientation', 'horizontal', 'fontsize', 15)
xlabel('time (s)', 'fontsize', font_size_label)
ylabel('Position (m)', 'fontsize', font_size_label)
title('Cartesian Space PD CTM Controller', 'fontsize', font_size_title)

% Draw Velocity

FG3 = figure('Position', [900 300 600 300], 'Color', [1 1 1]);
plot(simul_time, simul_dX_d_x, ':r', 'linewidth', 7); hold on;
plot(simul_time, simul_dX_d_y, ':b', 'linewidth', 7); hold on;

plot(simul_time, simul_dX_x, 'k', 'linewidth', 3); hold on;
plot(simul_time, simul_dX_y, 'g', 'linewidth', 3); hold on;

legend({'tar_{dx}', 'tar_{dy}', 'cur_{dx}', 'cur_{dy}'}, 'location', 'best', 'orientation', 'horizontal', 'fontsize', 15)
axis([start_t finish_t -1.25 1.25]);
xticks([start_t : 1 : finish_t]);
yticks([-2.5 : 1 : 2.5]);
grid on

xlabel('time (s)', 'fontsize', font_size_label)
ylabel('Velocity (m/s)', 'fontsize', font_size_label)
title('Cartesian Space PD CTM Controller', 'fontsize', font_size_title)


