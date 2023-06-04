% Add necessary paths for MATLAB to access the needed functions
addpath('/path/to/your/functions');

% Options to read: 'EEG-IO', 'EEG-VV', 'EEG-VR', 'EEG-MB'
data_folder = 'EEG-VV'; 

% Parameters
fs = 250.0;

% Reading data files
file_idx = 1;
all_files = dir(fullfile(data_folder,'*_data*'));
list_of_files = {all_files.name};
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
for idx = 1:numel(C{1})
    if strcmp(C{1}{idx},"corrupt")
        n_corrupt = C{2}(idx);
    elseif n_corrupt > 0
        if C{2}(idx) == -1
            t_end = data_sig(end,1);
        else
            t_end = C{2}(idx);
        end
        interval_corrupt = [interval_corrupt; [C{2}(idx), t_end]];
        n_corrupt = n_corrupt - 1;
    elseif strcmp(C{1}{idx},"blinks")
        %check that n_corrupt is 0
        if not(n_corrupt==0)
            disp("Error in parsing")
        else
            blinks = [blinks; C{2}(idx), C{2}(idx+1)];
            idx = idx + 1;  % Skip next index as it's already appended
        end
    end
end

% Loading data
if strcmp(data_folder, 'EEG-IO') || strcmp(data_folder, 'EEG-MB')
    data_sig = dlmread(fullfile(data_folder, file_sig),';',1,0);
elseif strcmp(data_folder, 'EEG-VR') || strcmp(data_folder, 'EEG-VV')
    data_sig = dlmread(fullfile(data_folder, file_sig),',',5,0);
    data_sig = data_sig(1:(int32(200*fs)+1),:);
    data_sig = data_sig(:,1:3);
    data_sig(:,1) = (0:length(data_sig)-1)'/fs;
end

% Visualizing data and ground truth
chan_id = 2;
figure;
plot(data_sig(:,1), data_sig(:,chan_id));
hold on;
for i = 1:size(interval_corrupt,1)
    x = [interval_corrupt(i,1) interval_corrupt(i,2)];
    y = get(gca, 'YLim');
    patch([x fliplr(x)], [y(1) y(1) y(2) y(2)], 'red', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
end
for d = 1:size(blinks,1)
    if blinks(d,2) < 2
        xline(blinks(d,1),'Color','green');
    elseif blinks(d,2) == 2
        xline(blinks(d,1),'Color','black');
    end
end
hold off;
