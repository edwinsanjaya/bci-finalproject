% Create a copy of the Epochs Data for the Neuro Dataset
% The entire Epochs Raw Data can be requested to the respective author of
% the dataset
newEpochs = Epochs;

% For Voluntary Blinking we delete the first row related to EOG
newEpochs.AllEpochs.Voluntary = permute(Epochs.AllEpochs.Voluntary, [2 1 3]);
newEpochs.AllEpochs.Voluntary(1,:,:) = [];
newEpochs.AllEpochs.Voluntary = permute(newEpochs.AllEpochs.Voluntary, [1 3 2]);

% For Involuntary Blinking we delete the first row related to EOG
newEpochs.AllEpochs.Involuntary = permute(Epochs.AllEpochs.Involuntary, [2 1 3]);
newEpochs.AllEpochs.Involuntary(1,:,:) = [];
newEpochs.AllEpochs.Involuntary = permute(newEpochs.AllEpochs.Involuntary, [1 3 2]);

%Variables used to get the .mat file of the raw data of Neuro dataset
blinks_data=newEpochs.AllEpochs;
jap_blink_voluntary=blinks_data.Voluntary;
jap_blink_involuntary=blinks_data.Involuntary;

%Merged voluntary and involuntary blinking data to process the ICA
merged_data = cat(3, blinks_data.Voluntary, blinks_data.Involuntary);


%Processing the merged data to obtain the .set to use in the EEGLAB:
samples=size(merged_data, 3);
sampling_rate=256; %Sample rate of the Neuro dataset
% Calculate time duration
timeDuration = (samples - 1) / sampling_rate; % Subtracting 1 to start from 0
% Generate time variable
time = linspace(0, timeDuration, samples);

% Initialize EEGLAB structure for the merged data
EEG_merged = eeg_emptyset;

% Adding the merged data
EEG_merged.data = merged_data;

% Update dimensions
EEG_merged.nbchan = size(EEG_merged.data, 1); % Number of channels
EEG_merged.pnts = size(EEG_merged.data, 2); % Number of points
EEG_merged.trials = size(EEG_merged.data, 3); % Number of trials
EEG_merged.srate = sampling_rate; %Sample rate
EEG_merged.times = time; %Time 
% Set the label
EEG_merged(1).labels = 'Merged Voluntary and Involuntary Raw Data';
% Define the channel labels
chan_labels = {'Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1', 'O2'};
% Assigning these channel labels to the data
for i = 1:length(chan_labels)
    EEG_merged.chanlocs(i).labels = chan_labels{i};
end
% Loading the standard channel location file
EEG_merged = pop_chanedit(EEG_merged, 'lookup','standard-10-5-cap385.elp');

% Saving the cleaned data as a .set file
outputFilenameSet = 'Neuro_merged_raw.set';
pop_saveset(EEG_merged, 'filename', outputFilenameSet, 'filepath', '');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % merged_data is the input array
original_size = size(merged_data);

%Applying the bandpass filter to the merged data
warning('off', 'all');
% Reshape the array
reshaped_array_merged = reshape(merged_data, [original_size(1), prod(original_size(2:end))]);
bandpass_merged = zeros(size(reshaped_array_merged, 1), size(reshaped_array_merged, 2));
for k = 1:size(reshaped_array_merged, 1)
    bandpass_merged(k,:) = bandpass(reshaped_array_merged(k,:),[1 50], 256, ImpulseResponse="iir", Steepness=0.95);
end
% Reshape back to the original size
merged_data = reshape(bandpass_merged, original_size);

warning('on', 'all');


%Processing the merged data to obtain the .set to use in the EEGLAB:
samples=size(merged_data, 3);
sampling_rate=256; %Sample rate of the Neuro dataset
% Calculate time duration
timeDuration = (samples - 1) / sampling_rate; % Subtracting 1 to start from 0
% Generate time variable
time = linspace(0, timeDuration, samples);

% Initialize EEGLAB structure for the merged data
EEG_merged = eeg_emptyset;

% Adding the merged data
EEG_merged.data = merged_data;

% Update dimensions
EEG_merged.nbchan = size(EEG_merged.data, 1); % Number of channels
EEG_merged.pnts = size(EEG_merged.data, 2); % Number of points
EEG_merged.trials = size(EEG_merged.data, 3); % Number of trials
EEG_merged.srate = sampling_rate; %Sample rate
EEG_merged.times = time; %Time 
% Set the label
EEG_merged(1).labels = 'Merged Voluntary and Involuntary Bandpass Filtered Data';
% Define the channel labels
chan_labels = {'Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1', 'O2'};
% Assigning these channel labels to the data
for i = 1:length(chan_labels)
    EEG_merged.chanlocs(i).labels = chan_labels{i};
end
% Loading the standard channel location file
EEG_merged = pop_chanedit(EEG_merged, 'lookup','standard-10-5-cap385.elp');

% Saving the cleaned data as a .set file
outputFilenameSet = 'Neuro_merged_bandpass.set';
pop_saveset(EEG_merged, 'filename', outputFilenameSet, 'filepath', '');


