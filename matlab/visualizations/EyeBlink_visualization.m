eeglab;

sample_rate = EEG.srate;                  
T = 1/sample_rate;
L = length(EEG.data); % Length of signal
f = sample_rate*(0:(L/2))/L;
t = (0:L-1)*T;        % Time vector

function fft_data = get_fft(X, L)
    Y = fft(X);
    P2 = abs(Y/L).^2;
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    fft_data = 10 * log10(P1); 
end

function plot_data = plot_data(data, f, fig_title, xlabels, ylabels)
    n = length(data);
    figure;
    for i = 1:n
        subplot(n,1,i);
        plot(f(i),data(i));
        title(fig_title(i));
        xlabel(xlabels(i));
        ylabel(ylabels(i));
        xlim([0, 100]);
    end
end
