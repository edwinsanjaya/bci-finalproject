% Preallocate EEG.icaact for speed
EEG.icaact = zeros(size(EEG.icaweights, 1), size(EEG.data, 2), size(EEG.data, 3));

% Iterate over the third dimension
for k = 1:size(EEG.data, 3)
    EEG.icaact(:,:,k) = (EEG.icaweights * EEG.icasphere) * EEG.data(:,:,k);
end


% Assuming 'EEG' is your EEG dataset that has been epoched

% Store the original epoch structure
original_epoch_structure = EEG.epoch;

% Reshape the EEG data to continuous form
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
arg_burst = 10;  % Burst criterion: 5 standard deviations
arg_window = 'off';  % Window criterion: 'off'

% Apply artifact subspace reconstruction method
EEG = clean_artifacts(EEG, 'FlatlineCriterion', arg_flatline, 'Highpass', arg_highpass, 'ChannelCriterion', arg_channel, 'NoisyCriterion', arg_noisy, 'BurstCriterion', arg_burst, 'WindowCriterion', arg_window);

% After cleaning, you can reshape the data back to its original epoch form
% However, you need to know the number of time points per epoch
num_timepoints_per_epoch = 1024; % fill this in

% Reshape the data back to its original 3D form
EEG.data = reshape(EEG.data, EEG.nbchan, num_timepoints_per_epoch, []);
EEG.pnts = num_timepoints_per_epoch;
EEG.trials = size(EEG.data, 3);
EEG.epoch = original_epoch_structure;