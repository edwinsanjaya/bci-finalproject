voluntary_data = RawData.Voluntary.Sub01.Trial02;
involuntary_data = RawData.Involuntary.Sub01.Trial02;

% test for only one channel
voluntary_data = voluntary_data(2,1:10000);
involuntary_data = involuntary_data(2,1:10000);

sample_rate = 256;                  
T = 1/sample_rate;
L = length(voluntary_data); % Length of signal
f = sample_rate*(0:(L/2))/L;
t = (0:L-1)*T;        % Time vector


fft_voluntary = get_fft(voluntary_data, L);
fft_involuntary = get_fft(involuntary_data, L);

plot_fft_Voluntary_Involuntary(fft_voluntary, fft_involuntary, f, ["Power spectrum Voluntary Raw", "Power spectrum Involuntary Raw"]);

bandpass_voluntary = bandpass(voluntary_data,[1 50], sample_rate, ImpulseResponse="iir", Steepness=0.95);
bandpass_involuntary = bandpass(involuntary_data,[1 50], sample_rate, ImpulseResponse="iir", Steepness=0.95);

fft_bandpass_voluntary = get_fft(bandpass_voluntary, L);
fft_bandpass_involuntary = get_fft(bandpass_involuntary, L);

plot_fft_Voluntary_Involuntary(fft_bandpass_voluntary, fft_bandpass_involuntary, f, ["Power spectrum Voluntary Bandpassed", "Power spectrum Involuntary Bandpass"]);


function fft_data = get_fft(X, L)
    Y = fft(X);
    P2 = abs(Y/L).^2;
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    fft_data = 10 * log10(P1); 
end

function plot_i_v_data = plot_fft_Voluntary_Involuntary(voluntary, involuntary, f, fig_title)
    figure;
    subplot(2,1,1);
    plot(f,voluntary);
    title(fig_title(1));
    xlabel("f (Hz)");
    ylabel("|P1(f)|");
    xlim([0, 100]);

    subplot(2,1,2);
    plot(f,involuntary);
    title(fig_title(2));
    xlabel("f (Hz)");
    ylabel("|P1(f)|");
    xlim([0, 100]);
end
