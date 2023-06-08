eeglab;

% Create a copy
newEpochs = Epochs;

% For Voluntary
newEpochs.AllEpochs.Voluntary = permute(Epochs.AllEpochs.Voluntary, [2 1 3]);
newEpochs.AllEpochs.Voluntary(1,:,:) = [];
newEpochs.AllEpochs.Voluntary = permute(newEpochs.AllEpochs.Voluntary, [1 3 2]);

% For Involuntary
newEpochs.AllEpochs.Involuntary = permute(Epochs.AllEpochs.Involuntary, [2 1 3]);
newEpochs.AllEpochs.Involuntary(1,:,:) = [];
newEpochs.AllEpochs.Involuntary = permute(newEpochs.AllEpochs.Involuntary, [1 3 2]);

blinks_data=newEpochs.AllEpochs;
merged_data = cat(3, blinks_data.Voluntary, blinks_data.Involuntary);

% merged_data is the input array
original_size = size(merged_data);

% Reshape the array
reshaped_array_merged = reshape(merged_data, [original_size(1), prod(original_size(2:end))]);

samples=size(merged_data, 3);
sampling_rate=256; %Sample rate of the dataset of voluntary blinking
% Calculate time duration
timeDuration = (samples - 1) / sampling_rate; % Subtracting 1 to start from 0
% Generate time variable
time = linspace(0, timeDuration, samples);

% Assuming 'newEpochs' is your modified data

% Initialize EEGLAB structure for the merged data
EEG_merged = eeg_emptyset;

% Add your data
EEG_merged.data = merged_data;

% Update dimensions
EEG_merged.nbchan = size(EEG_merged.data, 1); % Number of channels
EEG_merged.pnts = size(EEG_merged.data, 2); % Number of points
EEG_merged.trials = size(EEG_merged.data, 3); % Number of trials
EEG_merged.srate = sampling_rate;
EEG_merged.times = time; %Time 

% Set the label
EEG_merged(1).labels = 'Merged Voluntary and Involuntary Data';
% Define the channel labels
chan_labels = {'Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1', 'O2'};
% Assign these channel labels to your data
for i = 1:length(chan_labels)
    EEG_merged.chanlocs(i).labels = chan_labels{i};
end

% Load the standard channel location file
EEG_merged = pop_chanedit(EEG_merged, 'lookup','standard-10-5-cap385.elp');

% Save the cleaned data as a .set file
outputFilenameSet = 'Neuro_merged_data.set';
pop_saveset(EEG_merged, 'filename', outputFilenameSet, 'filepath', '');


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
