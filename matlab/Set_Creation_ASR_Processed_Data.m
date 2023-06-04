samples=size(EEG.data, 3);
sampling_rate=256; %Sample rate of the dataset of voluntary blinking
% Calculate time duration
timeDuration = (samples - 1) / sampling_rate; % Subtracting 1 to start from 0
% Generate time variable
time = linspace(0, timeDuration, samples);

% Assuming 'newEpochs' is your modified data

% Initialize EEGLAB structure for the merged data
EEG_ASR_Data = eeg_emptyset;

% Add your data
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
% Assign these channel labels to your data
for i = 1:length(chan_labels)
    EEG_ASR_Data.chanlocs(i).labels = chan_labels{i};
end
% Load the standard channel location file
EEG_ASR_Data = pop_chanedit(EEG_ASR_Data, 'lookup','standard-10-5-cap385.elp');

% Save the cleaned data as a .set file
outputFilenameSet = 'jap_ASR_processed_data.set';
pop_saveset(EEG_ASR_Data, 'filename', outputFilenameSet, 'filepath', '');
