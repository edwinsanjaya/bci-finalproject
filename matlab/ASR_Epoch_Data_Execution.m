% Preallocate EEG.icaact for speed
EEG.icaact = zeros(size(EEG.icaweights, 1), size(EEG.data, 2), size(EEG.data, 3));

% Iterate over the third dimension
for k = 1:size(EEG.data, 3)
    EEG.icaact(:,:,k) = (EEG.icaweights * EEG.icasphere) * EEG.data(:,:,k);
end


% ASREEG = clean_rawdata(EEG, 5, [-1], 0.85, 4, 20, 0.25);