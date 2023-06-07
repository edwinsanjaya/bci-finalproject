main_data = epochs_blinks_ASR_invol;
n_channels = size(main_data, 1);
L = size(main_data, 2);
n_epochs = size(main_data, 3);

Fs = 250;               % Sampling frequency             
T = 1/Fs;
t = (0:L-1)*T;          % Time vector
f = Fs*(0:(L/2))/L;

output_data = zeros(length(f), size(main_data, 2), n_epochs); % Initialize as 3D matrix

win_size = 250; % segment length
overlap = win_size - 100; % overlap value

% Loop over each epoch
for e = 1:n_epochs
    epoch_data = squeeze(main_data(:, :, e));
    [~, ~, ~, P] = spectrogram(epoch_data(:), win_size, overlap, [], Fs);

    output_data(:,:,e) = abs(P).^2; % Store spectrogram values in the corresponding epoch
end

spectrogram_data = 10 * log10(mean(output_data, 3)); % Average across epochs

% Plotting the spectrogram
figure;
imagesc(t, f, spectrogram_data);
set(gca, 'YDir', 'normal');
colorbar;
colormap('jet');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram');

