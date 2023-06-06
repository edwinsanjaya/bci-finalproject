% Options to read: 'EEG-IO', 'EEG-VV', 'EEG-VR'
data_folder = 'EEG-IO'; 

% For EEG-IO
%[master_data_sig_io, master_blinks_io, master_interval_corrupt_io, blink_intervals_io, min_blink_interval_io] = process_data('EEG-IO');
[epochs_blinks_2_seconds_vol, epochs_blinks_3_seconds_vol, epochs_blinks_4_seconds_vol, epochs_blinks_5_seconds_vol] = process_blinks('EEG-IO', master_blinks_io, blink_intervals_io, master_interval_corrupt_io, master_data_sig_io);
% For EEG-VV
%[master_data_sig_vv, master_blinks_vv, master_interval_corrupt_vv, blink_intervals_vv, min_blink_interval_vv] = process_data('EEG-VV');
[epochs_blinks_2_seconds_vv, epochs_blinks_3_seconds_vv, epochs_blinks_4_seconds_vv, epochs_blinks_5_seconds_vv] = process_blinks('EEG-VV', master_blinks_vv, blink_intervals_vv, master_interval_corrupt_vv, master_data_sig_vv);
% For EEG-VR
%[master_data_sig_vr, master_blinks_vr, master_interval_corrupt_vr, blink_intervals_vr, min_blink_interval_vr] = process_data('EEG-VR');
[epochs_blinks_2_seconds_vr, epochs_blinks_3_seconds_vr, epochs_blinks_4_seconds_vr, epochs_blinks_5_seconds_vr] = process_blinks('EEG-VR', master_blinks_vr, blink_intervals_vr, master_interval_corrupt_vr, master_data_sig_vr);

epochs_blinks_2_seconds_invol = cat(3, epochs_blinks_2_seconds_vv, epochs_blinks_2_seconds_vr);
epochs_blinks_3_seconds_invol = cat(3, epochs_blinks_3_seconds_vv, epochs_blinks_3_seconds_vr);
epochs_blinks_4_seconds_invol = cat(3, epochs_blinks_4_seconds_vv, epochs_blinks_4_seconds_vr);
epochs_blinks_5_seconds_invol = cat(3, epochs_blinks_5_seconds_vv, epochs_blinks_5_seconds_vr);

function [epochs_blinks_2_seconds, epochs_blinks_3_seconds, epochs_blinks_4_seconds, epochs_blinks_5_seconds] = process_blinks(data_folder, master_blinks, blink_intervals, master_interval_corrupt, master_data_sig)

    % Helper function to find the nearest value
    function idx = find_nearest(array, value)
        [~, idx] = min(abs(array - value));
    end

    % Initialize counters and arrays
    count1 = 0; count2 = 0; count3 = 0; count4 = 0;
    epochs_blinks_2_seconds = [];
    epochs_blinks_3_seconds = [];
    epochs_blinks_4_seconds = [];
    epochs_blinks_5_seconds = [];

    % Iterate through array, starting from second element
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

            % Extract previous 249 samples and current sample
            samples = master_data_sig(idx-249:idx,2:3);

            if blink_intervals(i) >= 1 && blink_intervals(i-1) >= 1
                % Append next 250 samples
                epochs_blinks_2_seconds(:,:,count1+1) = [samples; master_data_sig(idx+1:idx+250,2:3)];
                count1 = count1 + 1;
            end
            if blink_intervals(i) >= 2 && blink_intervals(i-1) >= 1
                % Append next 500 samples
                epochs_blinks_3_seconds(:,:,count2+1) = [samples; master_data_sig(idx+1:idx+500,2:3)];
                count2 = count2 + 1;
            end
            if blink_intervals(i) >= 3 && blink_intervals(i-1) >= 1
                % Append next 750 samples
                epochs_blinks_4_seconds(:,:,count3+1) = [samples; master_data_sig(idx+1:idx+750,2:3)];
                count3 = count3 + 1;
            end
            if blink_intervals(i) >= 4 && blink_intervals(i-1) >= 1
                % Append next 1000 samples
                epochs_blinks_5_seconds(:,:,count4+1) = [samples; master_data_sig(idx+1:idx+1000,2:3)];
                count4 = count4 + 1;
            end
        end
    end

    % Adjust the dimension order of the output arrays
    epochs_blinks_2_seconds = permute(epochs_blinks_2_seconds, [2 1 3]);
    epochs_blinks_3_seconds = permute(epochs_blinks_3_seconds, [2 1 3]);
    epochs_blinks_4_seconds = permute(epochs_blinks_4_seconds, [2 1 3]);
    epochs_blinks_5_seconds = permute(epochs_blinks_5_seconds, [2 1 3]);
end




function [master_data_sig, master_blinks, master_interval_corrupt, blink_intervals, min_blink_interval] = process_data(data_folder)

    % Parameters
    fs = 250.0;

    % Reading data files
    all_files = dir(fullfile(data_folder,'*_data*'));
    list_of_files = {all_files.name};

    master_data_sig = [];
    master_blinks = [];
    master_interval_corrupt = [];
    previous_end_time = 0;

    for file_idx = 1:numel(list_of_files)
        file_sig = list_of_files{file_idx};

        % Determine file_stim based on file_sig
        file_stim = replace(file_sig, '_data', '_labels');

        % Function to read stimulations
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
                % after the "blinks" line, each row with two numbers should be added to `blinks`
                blinks = [blinks; str2double(C{1}(idx))+previous_end_time, C{2}(idx)];
            end
            idx = idx + 1;  % Increment index at each loop iteration
        end
        % Loading data
        if strcmp(data_folder, 'EEG-IO')
            data_sig = dlmread(fullfile(data_folder, file_sig),';',1,0);
            data_sig(:,1) = data_sig(:,1) + previous_end_time;
        elseif strcmp(data_folder, 'EEG-VR') || strcmp(data_folder, 'EEG-VV')
            data_sig = dlmread(fullfile(data_folder, file_sig),',',5,0);
            data_sig = data_sig(1:(int32(200*fs)+1),:);
            data_sig = data_sig(:,1:3);
            data_sig(:,1) = ((0:length(data_sig)-1)'/fs) + previous_end_time;
        end

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

    for i = 1:size(master_interval_corrupt,1)
        x = [master_interval_corrupt(i,1) master_interval_corrupt(i,2)];
        y = get(gca, 'YLim');
        patch([x fliplr(x)], [y(1) y(1) y(2) y(2)], 'red', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    end
    for d = 1:size(master_blinks,1)
        if master_blinks(d,2) < 2
            xline(master_blinks(d,1),'Color','green');
        elseif master_blinks(d,2) == 2
            xline(master_blinks(d,1),'Color','black');
        end
    end

    xlabel('Time (s)'); % x-axis label
    ylabel('Amplitude (uV)'); % y-axis label
    title(['Blinks in ', data_folder, ' dataset']); % Title

    hold off;

end
