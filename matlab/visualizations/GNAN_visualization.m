sample_rate = EEG.srate;
main_data = EEG.data;

win_size = 250; % segment length
overlap = win_size - 100; % overlap value

n_channels = size(main_data, 1);
N = size(main_data, 2); % number of time points
n_epochs = size(main_data, 3);


% ERSP calculation
f = linspace(0, sample_rate/2, win_size/2+1); % frequency axis
n_freqs = length(f);
baseline = zeros(n_freqs, 1);

% Loop over each epoch
for i = 1:n_epochs
    epoch_baseline = squeeze(main_data(:, 1:win_size, i));
    
    % Compute the spectrogram of the epoch data
    [~, ~, ~, P] = spectrogram(epoch_baseline(:), win_size, overlap, [], sample_rate);
    
    if i == 1
        baseline = zeros(size(P));
    end
    baseline = baseline + P;
end

baseline_col = size(baseline, 2);
baseline_sum = sum(baseline, 2);
baseline = baseline_sum / (n_epochs * baseline_col);

ersp_total = zeros(size(baseline));

for i = 1:n_epochs
    epoch_data = squeeze(main_data(:, :, i));
    
    % Compute the spectrogram of the epoch data
    [~, ~, ~, P] = spectrogram(epoch_data(:), win_size, overlap, [], sample_rate);
    
    ersp_total = ersp_total + P;
end

ersp = 10*log10(abs(ersp_total).^2) - baseline;
ersp = ersp / n_epochs;


figure;
t = linspace(0, N/sample_rate, size(ersp, 2));
subplot(2,1,1);
imagesc(t, f, ersp);
ylim('auto');
colormap('jet');
colorbar;
xlabel('Time (s)');
ylabel('Frequency');
title('Average ERSP');

epoch_index = 2;
se_epoch_data = squeeze(main_data(:, :, epoch_index));

[~, f_s, t_s, P_s] = spectrogram(se_epoch_data(:), win_size, overlap, [], sample_rate);

ersp_s = 10*log10(abs(P_s).^2);

subplot(2,1,2);
imagesc(t_s, f_s, ersp_s);
ylim('auto');
colormap('jet');
colorbar;
xlabel('Time (s)');
ylabel('Frequency');
title('ERSP of a Single Epoch');


% Analysis on the time and frequency domain

% Generate a color map for differentiating channels
color_map = lines(n_channels);

% Plotting signals in the time domain
figure;
subplot(2, 1, 1);
hold on;
for i = 1:n_channels
    plot((0:N-1) / sample_rate, squeeze(main_data(i, :, epoch_index)), 'Color', color_map(i, :));
end
hold off;
xlabel('Time (s)');
ylabel('Amplitude');
title('Single Epoch Signals in the Time Domain');
legend({EEG.chanlocs.labels}, 'Location', 'eastoutside');

% Plotting signals in the frequency domain
subplot(2, 1, 2);
hold on;
for i = 1:n_channels
    Y = fft(squeeze(main_data(i, :, epoch_index)));
    f = (0:N/2) * sample_rate / N;
    P = abs(Y(1:N/2+1));
    plot(f, P, 'Color', color_map(i, :));
end
hold off;
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Single Epoch Signals in the Frequency Domain');
legend({EEG.chanlocs.labels}, 'Location', 'eastoutside');
