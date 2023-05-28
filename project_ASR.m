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

samples=size(blinks_data.Voluntary, 3);
sampling_rate=256; %Sample rate of the dataset of voluntary blinking
% Calculate time duration
timeDuration = (samples - 1) / sampling_rate; % Subtracting 1 to start from 0
% Generate time variable
time = linspace(0, timeDuration, samples);

% Assuming 'newEpochs' is your modified data

% Initialize EEGLAB structure for 'Voluntary'
EEG_voluntary = eeg_emptyset;

% Add your data
EEG_voluntary.data = blinks_data.Voluntary;

% Update dimensions
EEG_voluntary.nbchan = size(EEG_voluntary.data, 1); % Number of channels
EEG_voluntary.pnts = size(EEG_voluntary.data, 2); % Number of points
EEG_voluntary.trials = size(EEG_voluntary.data, 3); % Number of trials
EEG_voluntary.srate = sampling_rate;
EEG_voluntary.times = time; %Time 
% Set the channel labels
EEG_voluntary(1).labels = 'Voluntary Data';
% Save the cleaned data as a .set file
outputFilenameSet = 'jap_voluntary_data.set';
pop_saveset(EEG_voluntary, 'filename', outputFilenameSet, 'filepath', '');


% Repeat the same steps for 'Involuntary'
EEG_involuntary = eeg_emptyset;
EEG_involuntary.data = blinks_data.Involuntary;

EEG_involuntary.nbchan = size(EEG_involuntary.data, 1);
EEG_involuntary.pnts = size(EEG_involuntary.data, 2);
EEG_involuntary.trials = size(EEG_involuntary.data, 3);
EEG_involuntary.srate = sampling_rate;
EEG_involuntary.times = time; %Time 
% Set the channel labels
EEG_involuntary(1).labels = 'Involuntary Data';

% Save the cleaned data as a .set file
outputFilenameSet = 'jap_involuntary_data.set';
pop_saveset(EEG_involuntary, 'filename', outputFilenameSet, 'filepath', '');

%examp = EEG.icawinv*EEG.data;
%EEG.icaact = (EEG.icaweights*EEG.icasphere)*EEG.data;