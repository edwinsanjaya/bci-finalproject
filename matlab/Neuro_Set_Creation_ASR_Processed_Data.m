%Processing the bandpass filtered data with an additional ASR Algorithm
% Store the original epoch structure
original_epoch_structure = EEG.epoch;

% Reshaping the EEG data to continuous form
original_size = size(merged_data);
reshaped_merged_data=reshape(merged_data, [original_size(1), prod(original_size(2:end))]);
EEG.data = reshaped_merged_data;
EEG.trials = 1;
EEG.pnts = size(merged_data, 2);
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
num_timepoints_per_epoch = 1024; % 4 seconds of data (256 Hz frequency)
EEG.data = reshape(EEG.data, EEG.nbchan, num_timepoints_per_epoch, []);
EEG.pnts = num_timepoints_per_epoch;
EEG.trials = size(EEG.data, 3);
EEG.epoch = original_epoch_structure;


samples=size(EEG.data, 3);
sampling_rate=256; %Sample rate of the Neuro dataset 
% Calculating time duration
timeDuration = (samples - 1) / sampling_rate; % Subtracting 1 to start from 0
% Generating the time variable
time = linspace(0, timeDuration, samples);

% Initializing EEGLAB structure for the merged data
EEG_ASR_Data = eeg_emptyset;

% Adding the data
EEG_ASR_Data.data = EEG.data;

% Update dimensions
EEG_ASR_Data.nbchan = size(EEG_ASR_Data.data, 1); % Number of channels
EEG_ASR_Data.pnts = size(EEG_ASR_Data.data, 2); % Number of points
EEG_ASR_Data.trials = size(EEG_ASR_Data.data, 3); % Number of trials
EEG_ASR_Data.srate = sampling_rate;
EEG_ASR_Data.times = time; %Time 
% Set the label
EEG_ASR_Data(1).labels = 'ASR Processed Data';
% Define the channel labels
chan_labels = {'Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1', 'O2'};
% Assigning these channel labels to the data
for i = 1:length(chan_labels)
    EEG_ASR_Data.chanlocs(i).labels = chan_labels{i};
end
% Loading the standard channel location file
EEG_ASR_Data = pop_chanedit(EEG_ASR_Data, 'lookup','standard-10-5-cap385.elp');

% Saving the cleaned data as a .set file
outputFilenameSet = 'Neuro_merged_ASR.set';
pop_saveset(EEG_ASR_Data, 'filename', outputFilenameSet, 'filepath', '');
