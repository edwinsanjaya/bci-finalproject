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

% Assume blinks_data.Voluntary is your input array
original_size = size(blinks_data.Voluntary);

warning('off', 'all');
% Reshape the array
reshaped_array_vol = reshape(blinks_data.Voluntary, [original_size(1), prod(original_size(2:end))]);
bandpass_voluntary = zeros(size(reshaped_array_vol, 1), size(reshaped_array_vol, 2));
for k = 1:size(reshaped_array_vol, 1)
    bandpass_voluntary(k,:) = bandpass(reshaped_array_vol(k,:),[1 50], 256, ImpulseResponse="iir", Steepness=0.95);
end
% Reshape back to the original size
bandpass_voluntary = reshape(bandpass_voluntary, original_size);

% Assume blinks_data.Voluntary is your input array
original_size2 = size(blinks_data.Involuntary);

% Reshape the array
reshaped_array_invol = reshape(blinks_data.Involuntary, [original_size2(1), prod(original_size2(2:end))]);
bandpass_involuntary = zeros(size(reshaped_array_invol, 1), size(reshaped_array_invol, 2));
for k = 1:size(reshaped_array_invol, 1)
    bandpass_involuntary(k,:) = bandpass(reshaped_array_invol(k,:),[1 50], 256, ImpulseResponse="iir", Steepness=0.95);
end
% Reshape back to the original size
bandpass_involuntary = reshape(bandpass_involuntary, original_size2);

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

samples=size(bandpass_voluntary, 3);
sampling_rate=256; %Sample rate of the dataset of voluntary blinking
% Calculate time duration
timeDuration = (samples - 1) / sampling_rate; % Subtracting 1 to start from 0
% Generate time variable
time = linspace(0, timeDuration, samples);

% Assuming 'newEpochs' is your modified data

% Initialize EEGLAB structure for 'Voluntary'
EEG_voluntary = eeg_emptyset;

% Add your data
EEG_voluntary.data = bandpass_voluntary;

% Update dimensions
EEG_voluntary.nbchan = size(EEG_voluntary.data, 1); % Number of channels
EEG_voluntary.pnts = size(EEG_voluntary.data, 2); % Number of points
EEG_voluntary.trials = size(EEG_voluntary.data, 3); % Number of trials
EEG_voluntary.srate = sampling_rate;
EEG_voluntary.times = time; %Time 
% Set the label
EEG_voluntary(1).labels = 'Voluntary Data';
% Define the channel labels
chan_labels = {'Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1', 'O2'};
% Assign these channel labels to your data
for i = 1:length(chan_labels)
    EEG_voluntary.chanlocs(i).labels = chan_labels{i};
end
% Load the standard channel location file
EEG_voluntary = pop_chanedit(EEG_voluntary, 'lookup','standard-10-5-cap385.elp');

% Save the cleaned data as a .set file
outputFilenameSet = 'jap_voluntary_data.set';
pop_saveset(EEG_voluntary, 'filename', outputFilenameSet, 'filepath', '');


% Repeat the same steps for 'Involuntary'
EEG_involuntary = eeg_emptyset;
EEG_involuntary.data = bandpass_involuntary;

EEG_involuntary.nbchan = size(EEG_involuntary.data, 1);
EEG_involuntary.pnts = size(EEG_involuntary.data, 2);
EEG_involuntary.trials = size(EEG_involuntary.data, 3);
EEG_involuntary.srate = sampling_rate;
EEG_involuntary.times = time; %Time 
% Set the label
EEG_involuntary(1).labels = 'Involuntary Data';
% Define the channel labels
chan_labels = {'Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1', 'O2'};
% Assign these channel labels to your data
for i = 1:length(chan_labels)
    EEG_involuntary.chanlocs(i).labels = chan_labels{i};
end
% Load the standard channel location file
EEG_involuntary = pop_chanedit(EEG_involuntary, 'lookup','standard-10-5-cap385.elp');

% Save the cleaned data as a .set file
outputFilenameSet = 'jap_involuntary_data.set';
pop_saveset(EEG_involuntary, 'filename', outputFilenameSet, 'filepath', '');

