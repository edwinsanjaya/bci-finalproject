%Processing the GNAN Bandpass Filtered Data with the ASR Algorithm

% Store the original epoch structure
original_epoch_structure = EEG.epoch;

% Reshaping the EEG data to continuous form
original_size = size(epochs_blinks_4_seconds_merged);
reshaped_merged_data=reshape(epochs_blinks_4_seconds_merged, [original_size(1), prod(original_size(2:end))]);
EEG.data = reshaped_merged_data;
EEG.trials = 1;
EEG.pnts = size(epochs_blinks_4_seconds_merged, 2);
EEG.epoch = [];

disp(size(EEG.data));
% Parameters for the ASR method
arg_flatline = 'off';  % Flatline correction: 'off'
arg_highpass = 'off';  % High pass filter: 'off'
arg_channel = -1;  % Channel criterion: -1, do not remove bad channels
arg_noisy = 'off';  % Noisy criterion: 'off'
arg_burst = 5;  % Burst criterion: 5 standard deviations
arg_window = 'off';  % Window criterion: 'off'

% Apply artifact subspace reconstruction method
EEG = clean_artifacts(EEG, 'FlatlineCriterion', arg_flatline, 'Highpass', arg_highpass, 'ChannelCriterion', arg_channel, 'NoisyCriterion', arg_noisy, 'BurstCriterion', arg_burst, 'WindowCriterion', arg_window);

% Reshaping the data back to its original epoch form after cleaning
num_timepoints_per_epoch = 1000; %4 seconds of data (250 Hz frequency)

% Reshaping the data back to its original 3D form
EEG.data = reshape(EEG.data, EEG.nbchan, num_timepoints_per_epoch, []);
EEG.pnts = num_timepoints_per_epoch;
EEG.trials = size(EEG.data, 3);
EEG.epoch = original_epoch_structure;

%Obtaining the ASR Data in a .mat variable to use in the Machine Learning Model
n_vol = size(epochs_blinks_4_seconds_vol,3); 
% Separate epochs_blinks_4_seconds_vol
epochs_blinks_ASR_vol = EEG.data(:,:,1:n_vol);
% Separate epochs_blinks_4_seconds_invol
epochs_blinks_ASR_invol = EEG.data(:,:,(n_vol+1):end);

samples=size(EEG.data, 3);
sampling_rate=250; %Sample rate of the GNAN dataset
% Calculate time duration
timeDuration = (samples - 1) / sampling_rate; % Subtracting 1 to start from 0
% Generating the time variable
time = linspace(0, timeDuration, samples);

% Initializing EEGLAB structure for the merged data
EEG_ASR_Data = eeg_emptyset;

% Adding the data
EEG_ASR_Data.data = EEG.data;

% Updating dimensions
EEG_ASR_Data.nbchan = size(EEG_ASR_Data.data, 1); % Number of channels
EEG_ASR_Data.pnts = size(EEG_ASR_Data.data, 2); % Number of points
EEG_ASR_Data.trials = size(EEG_ASR_Data.data, 3); % Number of trials
EEG_ASR_Data.srate = sampling_rate;
EEG_ASR_Data.times = time; %Time 
% Set the label
EEG_ASR_Data(1).labels = 'GNAN Dataset ASR Processed Data';
% Defining the channel labels
chan_labels = {'Fp1', 'Fp2'};
% Assigning these channel labels to your data
for i = 1:length(chan_labels)
    EEG_ASR_Data.chanlocs(i).labels = chan_labels{i};
end
% Loading the standard channel location file
EEG_ASR_Data = pop_chanedit(EEG_ASR_Data, 'lookup','standard-10-5-cap385.elp');

% Saving the cleaned data as a .set file
outputFilenameSet = 'GNAN_merged_ASR.set';
pop_saveset(EEG_ASR_Data, 'filename', outputFilenameSet, 'filepath', '');
