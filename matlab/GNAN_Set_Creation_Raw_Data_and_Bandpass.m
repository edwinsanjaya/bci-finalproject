% Options to read: 'EEG-IO', 'EEG-VV', 'EEG-VR' for GNAN dataset
data_folder = 'EEG-IO'; 

% For EEG-IO
%Obtaining the raw data, blinks in time, corrupt data interval, blink
%intervals and minimum blink interval 
[master_data_sig_io, master_blinks_io, master_interval_corrupt_io, blink_intervals_io, min_blink_interval_io] = process_data('EEG-IO');
%Processing the blinks that are not in the corrupt interval and are not
%soft blinks, looking for intervals ranging from 2 to 5 seconds
[epochs_blinks_2_seconds_vol, epochs_blinks_3_seconds_vol, epochs_blinks_4_seconds_vol, epochs_blinks_5_seconds_vol] = process_blinks('EEG-IO', master_blinks_io, blink_intervals_io, master_interval_corrupt_io, master_data_sig_io);
% For EEG-VV
%Obtaining the raw data, blinks in time, corrupt data interval, blink
%intervals and minimum blink interval 
[master_data_sig_vv, master_blinks_vv, master_interval_corrupt_vv, blink_intervals_vv, min_blink_interval_vv] = process_data('EEG-VV');
%Processing the blinks that are not in the corrupt interval and are not
%soft blinks, looking for intervals ranging from 2 to 5 seconds
[epochs_blinks_2_seconds_vv, epochs_blinks_3_seconds_vv, epochs_blinks_4_seconds_vv, epochs_blinks_5_seconds_vv] = process_blinks('EEG-VV', master_blinks_vv, blink_intervals_vv, master_interval_corrupt_vv, master_data_sig_vv);
% For EEG-VR
%Obtaining the raw data, blinks in time, corrupt data interval, blink
%intervals and minimum blink interval 
[master_data_sig_vr, master_blinks_vr, master_interval_corrupt_vr, blink_intervals_vr, min_blink_interval_vr] = process_data('EEG-VR');
%Processing the blinks that are not in the corrupt interval and are not
%soft blinks, looking for intervals ranging from 2 to 5 seconds
[epochs_blinks_2_seconds_vr, epochs_blinks_3_seconds_vr, epochs_blinks_4_seconds_vr, epochs_blinks_5_seconds_vr] = process_blinks('EEG-VR', master_blinks_vr, blink_intervals_vr, master_interval_corrupt_vr, master_data_sig_vr);

%Appending the involuntary blinks
epochs_blinks_2_seconds_invol = cat(3, epochs_blinks_2_seconds_vv, epochs_blinks_2_seconds_vr);
epochs_blinks_3_seconds_invol = cat(3, epochs_blinks_3_seconds_vv, epochs_blinks_3_seconds_vr);
epochs_blinks_4_seconds_invol = cat(3, epochs_blinks_4_seconds_vv, epochs_blinks_4_seconds_vr);
epochs_blinks_5_seconds_invol = cat(3, epochs_blinks_5_seconds_vv, epochs_blinks_5_seconds_vr);

%Plotting the blinks in different intervals of time
plot_blink_data(epochs_blinks_2_seconds_invol, 'epochs_blinks_2_seconds_invol');
plot_blink_data(epochs_blinks_3_seconds_invol, 'epochs_blinks_3_seconds_invol');
plot_blink_data(epochs_blinks_4_seconds_invol, 'epochs_blinks_4_seconds_invol');
plot_blink_data(epochs_blinks_5_seconds_invol, 'epochs_blinks_5_seconds_invol');
plot_blink_data(epochs_blinks_2_seconds_vol, 'epochs_blinks_2_seconds_vol');
plot_blink_data(epochs_blinks_3_seconds_vol, 'epochs_blinks_3_seconds_vol');
plot_blink_data(epochs_blinks_4_seconds_vol, 'epochs_blinks_4_seconds_vol');
plot_blink_data(epochs_blinks_5_seconds_vol, 'epochs_blinks_5_seconds_vol');

%Appending the voluntary and involuntary blinks for the selected interval
%of 4 seconds, following standards in papers
epochs_blinks_4_seconds_merged = cat(3, epochs_blinks_4_seconds_vol, epochs_blinks_4_seconds_invol);


%Obtaining the .set file to use the data in EEGLAB
samples=size(epochs_blinks_4_seconds_merged, 3);
sampling_rate=250; %Sample rate of the entire GNAN dataset
% Calculating time duration
timeDuration = (samples - 1) / sampling_rate; % Subtracting 1 to start from 0
% Generating the time variable
time = linspace(0, timeDuration, samples);


% Initializing EEGLAB structure for the merged data
EEG_merged = eeg_emptyset;

% Adding the data
EEG_merged.data = epochs_blinks_4_seconds_merged;

% Update dimensions
EEG_merged.nbchan = size(EEG_merged.data, 1); % Number of channels
EEG_merged.pnts = size(EEG_merged.data, 2); % Number of points
EEG_merged.trials = size(EEG_merged.data, 3); % Number of trials
EEG_merged.srate = sampling_rate;
EEG_merged.times = time; %Time 
% Set the label
EEG_merged(1).labels = 'Merged Voluntary and Involuntary Blinking Raw Data';
% Define the channel labels
chan_labels = {'Fp1', 'Fp2'};
% Assign these channel labels to the data
for i = 1:length(chan_labels)
    EEG_merged.chanlocs(i).labels = chan_labels{i};
end
% Loading the standard channel location file
EEG_merged = pop_chanedit(EEG_merged, 'lookup','standard-10-5-cap385.elp');

% Saving the cleaned data as a .set file
outputFilenameSet = 'GNAN_merged_raw.set';
pop_saveset(EEG_merged, 'filename', outputFilenameSet, 'filepath', '');


%Original size of the merged data
original_size = size(epochs_blinks_4_seconds_merged);

%Applying the bandpass filter
warning('off', 'all');
% Reshaping the array
reshaped_epochs_blinks_4_seconds_merged = reshape(epochs_blinks_4_seconds_merged, [original_size(1), prod(original_size(2:end))]);
bandpass_epochs_blinks_4_seconds_merged = zeros(size(reshaped_epochs_blinks_4_seconds_merged, 1), size(reshaped_epochs_blinks_4_seconds_merged, 2));
for k = 1:size(reshaped_epochs_blinks_4_seconds_merged, 1)
    bandpass_epochs_blinks_4_seconds_merged(k,:) = bandpass(reshaped_epochs_blinks_4_seconds_merged(k,:),[1 50], 256, ImpulseResponse="iir", Steepness=0.95);
end
% Reshaping back to the original size with the bandpass filtered data
epochs_blinks_4_seconds_merged = reshape(bandpass_epochs_blinks_4_seconds_merged, original_size);

warning('on', 'all');

%Variables to export .mat file and use elsewhere
n_vol = size(epochs_blinks_4_seconds_vol,3); 
% Separate epochs_blinks_4_seconds_vol
epochs_blinks_bandpass_vol = epochs_blinks_4_seconds_merged(:,:,1:n_vol);
% Separate epochs_blinks_4_seconds_invol
epochs_blinks_bandpass_invol = epochs_blinks_4_seconds_merged(:,:,(n_vol+1):end);


samples=size(epochs_blinks_4_seconds_merged, 3);
sampling_rate=250; %Sample rate of the entire GNAN dataset
% Calculating time duration
timeDuration = (samples - 1) / sampling_rate; % Subtracting 1 to start from 0
% Generating the time variable
time = linspace(0, timeDuration, samples);


% Initializing EEGLAB structure for the merged data
EEG_merged = eeg_emptyset;

% Adding the data
EEG_merged.data = epochs_blinks_4_seconds_merged;

% Update dimensions
EEG_merged.nbchan = size(EEG_merged.data, 1); % Number of channels
EEG_merged.pnts = size(EEG_merged.data, 2); % Number of points
EEG_merged.trials = size(EEG_merged.data, 3); % Number of trials
EEG_merged.srate = sampling_rate;
EEG_merged.times = time; %Time 
% Set the label
EEG_merged(1).labels = 'Merged Voluntary and Involuntary Blinking Bandpass Filtered Data';
% Defining the channel labels
chan_labels = {'Fp1', 'Fp2'};
% Assigning these channel labels to your data
for i = 1:length(chan_labels)
    EEG_merged.chanlocs(i).labels = chan_labels{i};
end
% Loading the standard channel location file
EEG_merged = pop_chanedit(EEG_merged, 'lookup','standard-10-5-cap385.elp');

% Saving the cleaned data as a .set file
outputFilenameSet = 'GNAN_merged_bandpass.set';
pop_saveset(EEG_merged, 'filename', outputFilenameSet, 'filepath', '');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Function to plot one epoch of the filtered blinks in the data
function plot_blink_data(blink_data, variable_name)
    % Validate input
    if ndims(blink_data) ~= 3
        error('Input must be a 3D matrix.');
    end
    
    % Extract the first element in the third dimension
    first_blink_data = blink_data(:,:,10);
    
    % Create time vector
    sample_freq = 250; % Sample frequency in Hz of this dataset
    num_samples = size(first_blink_data, 2);
    time = (0:num_samples-1) / sample_freq; % Time in seconds
    
    % Create a new figure
    figure;
    
    % Plot the data from the first and second row (channel)
    plot(time, first_blink_data(1,:));
    hold on;
    plot(time, first_blink_data(2,:));
    hold off;
    % Determine the type of blink based on the variable name
    blink_type = '';
    if endsWith(variable_name, '_seconds_vol')
        blink_type = 'Voluntary';
        num_seconds = str2double(variable_name(15:end-12)); 
        disp(variable_name(15:end-12));
    elseif endsWith(variable_name, '_seconds_invol')
        blink_type = 'Involuntary';
        num_seconds = str2double(variable_name(15:end-14));
        disp(variable_name(15:end-14));
    end
    
    % Set the title and labels
    title(['Tenth Epoch of ' num2str(num_seconds) ' Seconds ' blink_type ' Blinking Data']);
    xlabel('Time (s)');
    ylabel('Amplitude (uV)');
    legend('Channel FP1', 'Channel FP2');
end

%Processing the blinks out of the corrupt intervals and soft blinks in the
%GNAN dataset
function [epochs_blinks_2_seconds, epochs_blinks_3_seconds, epochs_blinks_4_seconds, epochs_blinks_5_seconds] = process_blinks(data_folder, master_blinks, blink_intervals, master_interval_corrupt, master_data_sig)
    % Helper function to find the nearest value
    function idx = find_nearest(array, value)
        [~, idx] = min(abs(array - value));
    end

    % Initializing counters and arrays
    count1 = 0; count2 = 0; count3 = 0; count4 = 0;
    epochs_blinks_2_seconds = [];
    epochs_blinks_3_seconds = [];
    epochs_blinks_4_seconds = [];
    epochs_blinks_5_seconds = [];

    % Iterating through array, starting from second element
    for i = 2:length(blink_intervals)
        % Get the timestamp of the current blink
        current_blink_time = master_blinks(i,1);

        % Initialize flag to indicate whether the blink is within a corrupted interval
        blink_in_corrupt_interval = false;

        % Iterate through all corrupted intervals
        for j = 1:size(master_interval_corrupt,1)
            % Check if the blink is within the current corrupted interval
            if current_blink_time >= master_interval_corrupt(j,1) && current_blink_time <= master_interval_corrupt(j,2)
                blink_in_corrupt_interval = true;
                break;
            end
        end

        % Skip this iteration if the blink is within a corrupted interval
        if blink_in_corrupt_interval
            continue;
        end

        % Set the condition for the blink validity based on data_folder
        valid_blink_condition = false;
        if (strcmp(data_folder, 'EEG-VV') || strcmp(data_folder, 'EEG-VR')) && master_blinks(i,2) == 0
            valid_blink_condition = true;
        elseif strcmp(data_folder, 'EEG-IO') && master_blinks(i,2) == 1
            valid_blink_condition = true;
        end

        % Count valid blinks
        if valid_blink_condition
            % Find the index of the nearest timestamp in master_data_sig
            idx = find_nearest(master_data_sig(:,1), current_blink_time);
            % Extract previous 249 samples and current sample (1 second
            % before the event)
            samples = master_data_sig(idx-249:idx,2:3);

            if blink_intervals(i) >= 1 && blink_intervals(i-1) >= 1
                % Append next 250 samples (1 second)
                epochs_blinks_2_seconds(:,:,count1+1) = [samples; master_data_sig(idx+1:idx+250,2:3)];
                count1 = count1 + 1;
            end
            if blink_intervals(i) >= 2 && blink_intervals(i-1) >= 1
                % Append next 500 samples (2 seconds)
                epochs_blinks_3_seconds(:,:,count2+1) = [samples; master_data_sig(idx+1:idx+500,2:3)];
                count2 = count2 + 1;
            end
            if blink_intervals(i) >= 3 && blink_intervals(i-1) >= 1
                % Append next 750 samples (3 seconds)
                epochs_blinks_4_seconds(:,:,count3+1) = [samples; master_data_sig(idx+1:idx+750,2:3)];
                count3 = count3 + 1;
            end
            if blink_intervals(i) >= 4 && blink_intervals(i-1) >= 1
                % Append next 1000 samples (4 seconds)
                epochs_blinks_5_seconds(:,:,count4+1) = [samples; master_data_sig(idx+1:idx+1000,2:3)];
                count4 = count4 + 1;
            end
        end
    end

    % Adjust the dimension order of the output arrays (channels, signal,
    % epochs)
    epochs_blinks_2_seconds = permute(epochs_blinks_2_seconds, [2 1 3]);
    epochs_blinks_3_seconds = permute(epochs_blinks_3_seconds, [2 1 3]);
    epochs_blinks_4_seconds = permute(epochs_blinks_4_seconds, [2 1 3]);
    epochs_blinks_5_seconds = permute(epochs_blinks_5_seconds, [2 1 3]);
end



%Function to extract the data out of each folder and from each subject in the dataset
function [master_data_sig, master_blinks, master_interval_corrupt, blink_intervals, min_blink_interval] = process_data(data_folder)
    % Sampling frequency for the GNAN Dataset
    fs = 250.0;

    % Reading data files
    all_files = dir(fullfile(data_folder,'*_data*'));
    list_of_files = {all_files.name};

    master_data_sig = [];
    master_blinks = [];
    master_interval_corrupt = [];
    previous_end_time = 0;
    %Iterating through files in the folder
    for file_idx = 1:numel(list_of_files)
        file_sig = list_of_files{file_idx};

        file_stim = replace(file_sig, '_data', '_labels');

        % Obtaining blinks and corrupt data interval 
        interval_corrupt = [];
        blinks = [];
        n_corrupt = 0;
        file_stim_full = fullfile(data_folder, file_stim);
        fileID = fopen(file_stim_full,'r');
        C = textscan(fileID,'%s%f','Delimiter',',');
        fclose(fileID);
        idx = 1;
        while idx <= numel(C{1})
            if strcmp(C{1}{idx},"corrupt")
                n_corrupt = C{2}(idx);
            elseif n_corrupt > 0
                if C{2}(idx) == -1
                    t_end = data_sig(end,1);
                else
                    t_end = C{2}(idx);
                end
                interval_corrupt = [interval_corrupt; [str2double(C{1}(idx))+previous_end_time, t_end+previous_end_time]];
                n_corrupt = n_corrupt - 1;
            elseif strcmp(C{1}{idx},"blinks")
                %check that n_corrupt is 0
                if not(n_corrupt==0)
                    disp("Error in parsing")
                end
            else
                blinks = [blinks; str2double(C{1}(idx))+previous_end_time, C{2}(idx)];
            end
            idx = idx + 1;
        end
        % Loading data for each channel
        if strcmp(data_folder, 'EEG-IO')
            data_sig = dlmread(fullfile(data_folder, file_sig),';',1,0);
            data_sig(:,1) = data_sig(:,1) + previous_end_time;
        elseif strcmp(data_folder, 'EEG-VR') || strcmp(data_folder, 'EEG-VV')
            data_sig = dlmread(fullfile(data_folder, file_sig),',',5,0);
            data_sig = data_sig(1:(int32(200*fs)+1),:);
            data_sig = data_sig(:,1:3);
            data_sig(:,1) = ((0:length(data_sig)-1)'/fs) + previous_end_time;
        end
        %The data is appended for one general variable, this is done with each subject 
        master_data_sig = [master_data_sig; data_sig];
        master_blinks = [master_blinks; blinks];
        master_interval_corrupt = [master_interval_corrupt; interval_corrupt];

        previous_end_time = master_data_sig(end, 1);
    end

    % Calculate difference between consecutive blinks
    blink_intervals = diff(master_blinks(:,1));

    % Calculate minimum blink interval
    min_blink_interval = min(blink_intervals);

    % Visualizing data FP1 and FP2 and ground truth
    figure;
    plot(master_data_sig(:,1), master_data_sig(:,2)); % FP1
    hold on;
    plot(master_data_sig(:,1), master_data_sig(:,3)); % FP2
    
    %Visualizing the corrupt interval in red
    for i = 1:size(master_interval_corrupt,1)
        x = [master_interval_corrupt(i,1) master_interval_corrupt(i,2)];
        y = get(gca, 'YLim');
        patch([x fliplr(x)], [y(1) y(1) y(2) y(2)], 'red', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    end

    %Visualizing the blinks (induced and normal) in green and soft blinks in black
    for d = 1:size(master_blinks,1)
        if master_blinks(d,2) < 2
            xline(master_blinks(d,1),'Color','green');
        elseif master_blinks(d,2) == 2
            xline(master_blinks(d,1),'Color','black');
        end
    end

    xlabel('Time (s)'); % x-axis label
    ylabel('Amplitude (uV)'); % y-axis label
    title([data_folder,' Dataset Analysis: Blink Interval Patterns of All Subjects']); % Title
    legend('Channel FP1', 'Channel FP2');
    % Set x-axis limits
    xlim([0 max(master_data_sig(:,1))]);
    hold off;

end
