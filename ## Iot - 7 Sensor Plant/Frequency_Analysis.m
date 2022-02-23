% Potentiometer 

%% Set parameters
    % Set simulation time

    sim_y = [2493,2493,2493,2493,2493,2493,2493,2493,2493,2493,2493,2493,2493,2376,1858,1855,2356,2493,2493,2493,2493,2493,2458,2319,2232,2238,2303,2493,2493,2493,1721,925,1543,2303,2493,2493,2493,2493,2493,2493,2376,1858,1855,2356,2493,2493,2493,2493,2493,2458,2319,2232,2238,2303,2493,2493,2493,1721,925];
    len = length(sim_y);
    sim_time = (0 : len-1);  
 
    
    // Sampling frequency -> IIR 에 들어간 거랑 똑같이 해야하는지??
    Fs = 1/0.001;       % Sampling Frequency 1/0.001 = 1000 Hz
    T =  1/Fs;          % Sampling Period 二쇨린 0.001
    L = length(sim_y);   
    T_vector = (0:L-1)*T; 
    fft_f = Fs*(0:(L/2))/L; % Frequency Range 
    fft_y_temp = abs(fft(sim_y)/L);
    fft_y = fft_y_temp(1:L/2+1);
    fft_y(2:end-1) = 2*fft_y(2:end-1);

%% Draw Graph

    figure('units', 'pixels', 'pos', [100 300 1600 1200], 'Color', [1,1,1]); 
    
    % Time - Domain 
    subplot(2,1,1) 
    Xmin = 0.0; XTick = 10.0; Xmax = len; 
    Ymin = 0.0; YTick = 100.0; Ymax = 2500.0;    
        plot(sim_time, sim_y, '-k', 'LineWidth', 2) 
        
        grid on; 
        axis([Xmin, Xmax, Ymin, Ymax]) 
            set(gca, 'XTick', [Xmin:XTick:Xmax]); 
            set(gca, 'YTick', [Ymin:YTick:Ymax]); 
        xlabel('Time (s)' , 'fontsize', 20); 
        ylabel('Magnitude', 'fontsize', 20); 
        title('Time Domain', 'fontsize', 25);  
    
    % Frequency Domain
    subplot(2,1,2) 
    
    Xmin = 0.0; Xmax = 500;
    Ymin = 0.0; Ymax = 2000.0;
   
        stem(fft_f, fft_y, '-k', 'LineWidth', 2); 
        
        grid on;
        
        axis([Xmin, Xmax, Ymin, Ymax]) 
        set(gca, 'XTick', [0 10.0 500.0]); 
        set(gca, 'YTick', [0 1.0 2.0]); 
    xlabel('Frequency (Hz)' , 'fontsize', 20); 
    ylabel('Magnitude', 'fontsize', 20); 
    title('Frequency Domain', 'fontsize', 25); 
       
        
    
