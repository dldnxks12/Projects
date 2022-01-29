%% Computed Torque Method

% 어떤 위치까지 모터가 돌도록 하기 위해 필요한 Torque가 얼마냐? 를 계산하는 방법

% 기존의 방법은 Encoder를 이용해서 현재 각도를 읽고, 목표 각도와의 차이 즉 에러를 계산해서 오차를 줄여나가는 각도 기반의 제어
% 이번에 사용할 CTM은 토크 기반의 방법! --- 각도 기반은 시스템이 선형적이라는 가정 하였다. (사실은 중력의 영향을 받기 때문에 비선형이다!)
% CTM은 비선형 시스템을 선형화해서, 선형 제어기를 사용하기 위함이다! 

%%
clc 
clear all
close all

global m l L g tau;

% Simulation Parameters

delta_t  = 0.005;     % [sec] : Sampling time
start_t  = 0.000;     % [sec] : Start time
finish_t = 5.000;     % [sec] : End time

g        = 9.8148;    % [m/s^2] : Gravitational Acceleration

% Roobot Parameters

m       = 1.0000;      % [kg]    : Link Mass
L       = 1.0000;      % [m]     : Link Length
l       = (m*L^2)/3;   % [kgm^2] : Link Inertia
tau     = 0.0000;      % [Nm]    : Control Torque

init_q  = 0.00;        % [rad]    : Initial Joint Angle -------------- 초기 각도 0도로!
init_dq = 0.00;        % [rad/s]  : Initial Joint Angular Vel
q       = init_q;      % [rad]    : Joint Angle
dq      = init_dq;     % [rad/s]  : Joint Angular Vel

% Target Parameters

q_d   = init_q;
dq_d  = 0;
ddq_d = 0;

% Contoller Gain

Wn    = 80; % 20 -> 60 good -> 120 better 
Kp    = Wn^2;              % P Gain
Kv    = 2*Wn;              % D Gain 
Ki    = Kp*Kv*0.005;         % I Gain --- Ki < Kp*Kv !!
%% Simulation

n = 1;
Integral_temp = 0;              % I Term의 적분을 위한 변수 

for time = start_t:delta_t:finish_t
    % Robot Model
        % Set Trajectory 45에서 시작 -> 2초에 걸쳐 45도를 이동하여 90도로 
        if time < 1
            q_d   = init_q;
            dq_d  = 0.0;
            ddq_d = 0.0;
        else
            if(q_d < 90*(pi/180))
                q_d = q_d + (60*pi/180/2)*delta_t;   %  목표 각속도 : 60 rad / 2 s == 30 deg/s                              
            else
                q_d = 90*(pi/180);
            end
            
            dq_d  = (q_d  - simul_q_d(n-1))  / delta_t;
            ddq_d = (dq_d - simul_dq_d(n-1)) / delta_t;
        end
        % Dynamics
        G = get_Gravity(q);
        
        % Control Torque        
        Integral_temp = Integral_temp + (q_d - q)*delta_t;
        u = ddq_d + Kv*(dq_d - dq) + Kp*(q_d- q) + Ki*Integral_temp;
        tq_control = l*u + G*0.9;  % G = 중력보상 
        
            % Inverse Dynamics
            tau = tq_control;
            [t, y] = ode45('one_link', [0 delta_t], [q; dq]);
            index = length(y);

            q  = y(index, 1);
            dq = y(index, 2);

    % Save Data
    simul_time(n)    = time;  % [sec]
    simul_q(n)       = q;     % [rad]
    simul_dq(n)      = dq;   % [rad/s]
    simul_q_d(n)     = q_d;     % [rad]
    simul_dq_d(n)    = dq_d;   % [rad/s]    
    n = n+1;                      
end

font_size_label   = 20;
font_size_title   = 25;
linewidth_current = 3;
linewidth_target  = 5;

init_x = L*sin(init_q);
init_y = -L*cos(init_q);

FG1 = figure('Position', [200 300 700 700], 'Color', [1 1 1]);
AX = axes('parent', FG1);
hold on 

p = plot([0 0], [init_x init_y], '-ob', 'Linewidth', linewidth_current);
axis([-1.5 1.5 -1.5 1.5]);
grid on 

xlabel('X-axis (m)', 'fontsize', font_size_label)
ylabel('Y-axis (m)', 'fontsize', font_size_label)
title('1 DOF Robot', 'fontsize', font_size_title)

% Draw Robot !
n = 1;
for time = start_t:delta_t:finish_t
    q = simul_q(n);
    x = L*sin(q);
    y = -L*cos(q);
    
    Px = [0, x];
    Py = [0, y];    
    set(p, 'XData', Px, 'YData', Py);
    drawnow
    n = n+1;
end

%% Draw Datas

% Draw Angle
FG2 = figure('Position', [900 700 600 300], 'Color', [1 1 1]);
plot(simul_time, simul_q*180/pi, 'r', 'linewidth', linewidth_current);
hold on 
plot(simul_time, simul_q_d*180/pi, ':k', 'linewidth', linewidth_target);
hold on 
legend('Current', 'Desired')
axis([start_t finish_t 0 120]);
xticks([start_t : 1 : finish_t]);
yticks([0 : 45 : 90]);
grid on 

xlabel('time (s)', 'fontsize', font_size_label)
ylabel('Angle (deg)', 'fontsize', font_size_label)
title('Joint Space PD CTM Controller', 'fontsize', font_size_title)

% Draw Angular Velocity

FG3 = figure('Position', [900 300 600 300], 'Color', [1 1 1]);
plot(simul_time, simul_dq*180/pi, 'r', 'linewidth', linewidth_current);
hold on 
plot(simul_time, simul_dq_d*180/pi, ':k', 'linewidth', linewidth_target);
hold on 
legend('Current', 'Desired')
axis([start_t finish_t -90 90]);
xticks([start_t : 1 : finish_t]);
yticks([-60 0 60/2 60]);
grid on 

xlabel('time (s)', 'fontsize', font_size_label)
ylabel('Anglular Velocity (deg/s)', 'fontsize', font_size_label)
title('Joint Space PD CTM Controller', 'fontsize', font_size_title)


