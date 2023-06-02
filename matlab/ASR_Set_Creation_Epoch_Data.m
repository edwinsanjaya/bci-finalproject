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

% % merged_data is the input array
original_size = size(merged_data);

warning('off', 'all');
% Reshape the array
reshaped_array_merged = reshape(merged_data, [original_size(1), prod(original_size(2:end))]);
bandpass_merged = zeros(size(reshaped_array_merged, 1), size(reshaped_array_merged, 2));
for k = 1:size(reshaped_array_merged, 1)
    bandpass_merged(k,:) = bandpass(reshaped_array_merged(k,:),[1 50], 256, ImpulseResponse="iir", Steepness=0.95);
end
% Reshape back to the original size
bandpass_merged = reshape(bandpass_merged, original_size);

warning('on', 'all');


% warning('off', 'all');
% % Preallocate bandpass_voluntary for speed
% bandpass_voluntary = zeros(size(blinks_data.Voluntary, 1), size(blinks_data.Voluntary, 2), size(blinks_data.Voluntary, 3));
% 
% % Preallocate bandpass_involuntary for speed
% bandpass_involuntary = zeros(size(blinks_data.Involuntary, 1), size(blinks_data.Involuntary, 2), size(blinks_data.Involuntary, 3));
% 
% for k = 1:size(blinks_data.Voluntary, 3)
%     bandpass_voluntary(:,:,k) = bandpass(blinks_data.Voluntary(:,:,k),[1 50], 256, ImpulseResponse="iir", Steepness=0.95);
%     %bandpass_voluntary(:,:,k) = bandpass(blinks_data.Voluntary(:,:,k),[1 50], 256); 
% end
% 
% for k = 1:size(blinks_data.Involuntary, 3)
%     bandpass_involuntary(:,:,k) = bandpass(blinks_data.Involuntary(:,:,k),[1 50], 256, ImpulseResponse="iir", Steepness=0.95);
% end
% 
% warning('on', 'all');

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
EEG_merged(1).labels = 'Voluntary Data';
% Define the channel labels
chan_labels = {'Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1', 'O2'};
% Assign these channel labels to your data
for i = 1:length(chan_labels)
    EEG_merged.chanlocs(i).labels = chan_labels{i};
end
% Load the standard channel location file
EEG_merged = pop_chanedit(EEG_merged, 'lookup','standard-10-5-cap385.elp');

% Save the cleaned data as a .set file
outputFilenameSet = 'jap_merged_data.set';
pop_saveset(EEG_merged, 'filename', outputFilenameSet, 'filepath', '');


