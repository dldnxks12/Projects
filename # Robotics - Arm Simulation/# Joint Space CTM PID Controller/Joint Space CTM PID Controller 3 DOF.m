%% 2014741002 로봇학부 이종수 TermProject #1
clc
clear all
close all

%%

global lz1 lz2 lz3 L1 L2 L3 g m1 m2 m3 r1 r2 r3 tau1 tau2 tau3;

% Simulation Parameters
delta_t  = 0.005;     % [sec] : Sampling time
start_t  = 0.000;     % [sec] : Start time
finish_t = 5.000;     % [sec] : End time

g        = 9.8148;    % [m/s^2] : Gravitational Acceleration

% Roobot Parameters

m1       = 0.2000;      % [kg]    : Link Mass
m2       = 0.2000;      % [kg]    : Link Mass
m3       = 0.2000;      % [kg]    : Link Mass

r1       = 0.10;        % [m]     : COG distance
r2       = 0.10;        % [m]     : COG distance
r3       = 0.10;        % [m]     : COG distance

L1       = 0.5000;      % [m]     : Link Length
L2       = 0.5000;      % [m]     : Link Length
L3       = 0.5000;      % [m]     : Link Length

lz1       = (m1*L1^2)/3;   % [kgm^2] : Link Inertia
lz2       = (m2*L2^2)/3;   % [kgm^2] : Link Inertia
lz3       = (m3*L3^2)/3;   % [kgm^2] : Link Inertia

tau1     = 0.0000;      % [Nm]    : Control Torque
tau2     = 0.0000;      % [Nm]    : Control Torque
tau3     = 0.0000;      % [Nm]    : Control Torque

tau      = [tau1
            tau2
            tau3];      % [Nm]    : Control Torque

init_q1  = -pi/2;       % [rad]    : Initial Joint Angle
init_q2  = pi/4;        % [rad]    : Initial Joint Angle
init_q3  = pi/4;        % [rad]    : Initial Joint Angle

init_dq1 = 0.00;        % [rad/s]  : Initial Joint Angular Vel
init_dq2 = 0.00;        % [rad/s]  : Initial Joint Angular Vel
init_dq3 = 0.00;        % [rad/s]  : Initial Joint Angular Vel

q       = [init_q1
           init_q2
           init_q3];       % [rad]    : Joint Angle

dq      = [init_dq1
           init_dq2
           init_dq3];     % [rad/s]  : Joint Angular Vel

% Target Parameters

q_d   = q;                      % [rad]        : Target Joint Angle
dq_d  = [0; 0; 0;];             % [rad/s]      : Target Joint Angular Velocity
ddq_d = [0; 0; 0;];             % [rad/s^2]    : Target Joint Angular Acceleration

% Contoller Gain
% 우리는 Critically Damped Condition으로 만들 것이기 때문에 Damping ratio를 1로 상정한다.
Wn    = 80;                 % [Hz] : Natural Frequency ----  20 initial -> 60 good -> 120 better
Kp    = Wn^2;                % P Gain
Kv    = 2*Wn;                % D Gain
Ki    = Kp*Kv*0.05;         % I Gain --- 일반적으로 Ki < Kp*Kv 의 범위에서 정한다.

%% Simulation

n = 1;
Integral_temp = 0; % Integral Term을 위한 임시 저장 메모리

for time = start_t:delta_t:finish_t
    % Robot Model
        % Set Trajectory
        % 시작하고서 1초 까지는 그 각도를 그대로 유지
        if time < 1
            q_d   = q;
            dq_d  = [0; 0; 0;];
            ddq_d = [0; 0; 0;];

        else % 1초 후에는 30 deg/s의 속도로 우리가 원하는 위치인 90도로 이동한다.

            % 마지막 Link의 위치를 90도로 바꿀 것이기 때문에 q_d(3)를 조정 !
            if(q_d(3) < 90*(pi/180))
                q_d(3) = q_d(3) + (60*pi/180/2)*delta_t;   %  목표 각속도 : 60 rad / 2 s == 30 deg/s
            else
                q_d(3) = 90*(pi/180); % 목표 위치에 다다랐았을 경우 해당 위치 유지
            end

            % 수치 미분을 통해 각속도, 각가속도를 계산
            dq_d  = (q_d  - [simul_q_d1(n-1); simul_q_d2(n-1); simul_q_d3(n-1)]  )  / delta_t;
            ddq_d = (dq_d - [simul_dq_d1(n-1); simul_dq_d2(n-1); simul_dq_d3(n-1)]  ) / delta_t;

        end

        D     =  get_MassInertia(q(2), q(3));                       % Mass Inertia Term
        H     =  get_CC(q(2), q(3), dq(1), dq(2), dq(3));           % Corilois Term
        C     =  get_Gravity2(q(1), q(2), q(3));                    % Gravity Term

        % Control Torque
        Integral_temp = Integral_temp + (q_d - q)*delta_t;            % 각도 오차 값을 누적시키는 부분

        % Error Dynamics로 생각할 수도 있음 PID 제어 수행 !
        u = ddq_d + Kv*(dq_d - dq) + Kp*(q_d- q) + Ki*Integral_temp;  % Servo Control Term

        % Control Partitioning을 이용하였으므로, FeedForward 수행 - M, H, C Term 보상
        tq_control = D*u + H + C*0.9; % 제어 입력

            % Inverse Dynamics
            tau = tq_control;
            tau1 = tau(1);
            tau2 = tau(2);
            tau3 = tau(3);

            % 3 DOF Link에 대한 수치 미분으로 해당 시간에 맞는 각도, 각속도 값 가져올 것
            [t, y] = ode45('three_link', [0 delta_t], [q(1); dq(1); q(2); dq(2); q(3); dq(3)]);
            % 가장 마지막 값을 가져올 것이기 때문에 index라는 변수에 값 저장
            index = length(y);

            % 업데이트를 위한 Joint 속성들 쭉 받아오기
            q  = [y(index, 1); y(index, 3); y(index, 5)];
            dq = [y(index, 2); y(index, 4); y(index, 6)];

    % Save Data ! --- Graph 그려서 데이터 변화 확인할 것 !
    simul_time(n)    = time;      % [sec]

    % 현재 각도들
    simul_q1(n)       = q(1);     % [rad]
    simul_q2(n)       = q(2);     % [rad]
    simul_q3(n)       = q(3);     % [rad]

    % Target 각도들
    simul_q_d1(n)       = q_d(1);     % [rad]
    simul_q_d2(n)       = q_d(2);     % [rad]
    simul_q_d3(n)       = q_d(3);     % [rad]

    % 현재 각속도들
    simul_dq1(n)      = dq(1);    % [rad/s]
    simul_dq2(n)      = dq(2);    % [rad/s]
    simul_dq3(n)      = dq(3);    % [rad/s]

    % Target 각속도들
    simul_dq_d1(n)      = dq_d(1);    % [rad/s]
    simul_dq_d2(n)      = dq_d(2);    % [rad/s]
    simul_dq_d3(n)      = dq_d(3);    % [rad/s]


    n = n+1;
end

% Plotting 속성 설정
font_size_label   = 20;
font_size_title   = 25;

linewidth_current = 3;
linewidth_target  = 5;

FG1 = figure('Position', [200 200 700 700], 'Color', [1 1 1]);
AX = axes('parent', FG1);

hold on
grid on

axis([-2.0 2.0 -2.0 2.0]);

% Forward Kinematic를 통해 초기 위치 값 Get
x1 = L1*cos(init_q1);                            % [m] : Joint 1 X-axis Position
y1 = L1*sin(init_q1);                            % [m] : Joint 1 Y-axis Position
x2 = L2*cos(init_q1+init_q2);                    % [m] : Joint 2 X-axis Position
y2 = L2*sin(init_q1+init_q2);                    % [m] : Joint 2 Y-axis Position
x3 = L3*cos(init_q1+init_q2+init_q3);            % [m] : Joint 3 X-axis Position
y3 = L3*sin(init_q1+init_q2+init_q3);            % [m] : Joint 3 Y-axis Position

% 초기 위치에 맞게 초기 그래프 그리기
Px1 = [0 x1];
Py1 = [0 y1];
Px2 = [x1 x1+x2];
Py2 = [y1 y1+y2];
Px3 = [x1+x2 x1+x2+x3];
Py3 = [y1+y2 y1+y2+y3];

p1 = plot(Px1, Py1, '-ob', 'Linewidth', linewidth_current);
p2 = plot(Px2, Py2, '-or', 'Linewidth', linewidth_current);
p3 = plot(Px3, Py3, '-ok', 'Linewidth', linewidth_current);

% Plotting 속성 설정
xlabel('X-axis (m)', 'fontsize', font_size_label)
ylabel('Y-axis (m)', 'fontsize', font_size_label)
title('3 DOF Robot (Joint Space Control)', 'fontsize', font_size_title)


% Draw Robot !
n = 1;
for time = start_t:delta_t:finish_t
    % 저장해둔 데이터 Array에서 하나씩 뽑아와서 그래프 그릴 것 - 마치 움직이듯이 그려진다.

    % 해당 시간의 각도값
    q1 = simul_q1(n);
    q2 = simul_q2(n);
    q3 = simul_q3(n);

    % Forward Kinematics를 통해 매 순간 위치를 추적
    x1 = L1*cos(q1);
    y1 = L1*sin(q1);
    x2 = L2*cos(q1+q2);
    y2 = L2*sin(q1+q2);
    x3 = L3*cos(q1+q2+q3);
    y3 = L3*sin(q1+q2+q3);

    % 추적한 값을 기반으로 그래프 업데이트
    Px1 = [0 x1]; Py1 = [0 y1];
    Px2 = [x1 x1+x2]; Py2 = [y1 y1+y2];
    Px3 = [x1+x2 x1+x2+x3]; Py3 = [y1+y2 y1+y2+y3];

    set(p1, 'XData', Px1, 'YData', Py1);
    set(p2, 'XData', Px2, 'YData', Py2);
    set(p3, 'XData', Px3, 'YData', Py3);
    drawnow
    n = n+1;
end

%% Draw Datas --- 데이터가 어떻게 변하는지, Overshoot, Undershoot, Steady State Error .. 등을 확인할 수 있다.

% 각도 그래프 그려보기!

% 첫 번째 Joint Angle
FG2 = figure('Position', [700 700 600 300], 'Color', [1 1 1]);
plot(simul_time, simul_q1*180/pi, 'r', 'linewidth', linewidth_current); hold on;
plot(simul_time, simul_q_d1*180/pi, ':k', 'linewidth', linewidth_target); hold on;

legend('Current', 'Desired')
axis([start_t finish_t 0 120]);
xticks([start_t : 1 : finish_t]);
yticks([-90 : 45 : 90]);
grid on

xlabel('time (s)', 'fontsize', font_size_label)
ylabel('1th Joint Angle (deg)', 'fontsize', font_size_label)
title('3 DOF Joint Space Controller', 'fontsize', font_size_title)


% 두 번째 Joint Angle
FG3 = figure('Position', [700 300 600 300], 'Color', [1 1 1]);
plot(simul_time, simul_q2*180/pi, 'r', 'linewidth', linewidth_current); hold on;
plot(simul_time, simul_q_d2*180/pi, ':k', 'linewidth', linewidth_target); hold on;

legend('Current', 'Desired')
axis([start_t finish_t 0 120]);
xticks([start_t : 1 : finish_t]);
yticks([0 : 45 : 90]);
grid on

xlabel('time (s)', 'fontsize', font_size_label)
ylabel('2th Joint Angle (deg)', 'fontsize', font_size_label)
title('3 DOF Joint Space Controller', 'fontsize', font_size_title)


% 세 번째 Joint Angle
FG4 = figure('Position', [700 0 600 300], 'Color', [1 1 1]);
plot(simul_time, simul_q3*180/pi, 'r', 'linewidth', linewidth_current); hold on;
plot(simul_time, simul_q_d3*180/pi, ':k', 'linewidth', linewidth_target); hold on;

legend('Current', 'Desired')
axis([start_t finish_t 0 120]);
xticks([start_t : 1 : finish_t]);
yticks([0 : 45 : 90]);
grid on

xlabel('time (s)', 'fontsize', font_size_label)
ylabel('3th Joint Angle (deg)', 'fontsize', font_size_label)
title('3 DOF Joint Space Controller', 'fontsize', font_size_title)

%%
% 첫 번째 Joint Angular Vel
FG2 = figure('Position', [1300 700 600 300], 'Color', [1 1 1]);
plot(simul_time, simul_dq1*180/pi, 'r', 'linewidth', linewidth_current); hold on;
plot(simul_time, simul_dq_d1*180/pi, ':k', 'linewidth', linewidth_target); hold on;

legend('Current', 'Desired')
axis([start_t finish_t 0 120]);
xticks([start_t : 1 : finish_t]);
yticks([-30 : 15 : 30]);
grid on

xlabel('time (s)', 'fontsize', font_size_label)
ylabel('1th Joint Angular Velocity (deg/sec)', 'fontsize', 10)
title('3 DOF Joint Space Controller', 'fontsize', font_size_title)


% 두 번째 Joint Angular Vel
FG3 = figure('Position', [1300 300 600 300], 'Color', [1 1 1]);
plot(simul_time, simul_dq2*180/pi, 'r', 'linewidth', linewidth_current); hold on;
plot(simul_time, simul_dq_d2*180/pi, ':k', 'linewidth', linewidth_target); hold on;

legend('Current', 'Desired')
axis([start_t finish_t 0 120]);
xticks([start_t : 1 : finish_t]);
yticks([-30 : 15 : 30]);
grid on

xlabel('time (s)', 'fontsize', font_size_label)
ylabel('2th Joint Angular Velocity (deg/sec)', 'fontsize', 10)
title('3 DOF Joint Space Controller', 'fontsize', font_size_title)


% 세 번째 Joint Angular Vel
FG4 = figure('Position', [1300 0 600 300], 'Color', [1 1 1]);
plot(simul_time, simul_dq3*180/pi, 'r', 'linewidth', linewidth_current); hold on;
plot(simul_time, simul_dq_d3*180/pi, ':k', 'linewidth', linewidth_target); hold on;

legend('Current', 'Desired')
axis([start_t finish_t 0 120]);
xticks([start_t : 1 : finish_t]);
yticks([-30 : 15 : 30]);
grid on

xlabel('time (s)', 'fontsize', font_size_label)
ylabel('3th Joint Angular Velocity (deg/sec)', 'fontsize', 10)
title('3 DOF Joint Space Controller', 'fontsize', font_size_title)



