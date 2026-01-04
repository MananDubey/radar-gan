%% Radar Simulation & Dataset Generation Code
%% AUTHOR: Manan Dubey



clc; clear; close all;


fs = 1e6;
fc = 10e9;
c = physconst('LightSpeed');
prf = 20e3;
pulseWidth = 1e-6;
numPulses = 128;
numSamples = 3000;

imgSize = [128, 128];

datasetRoot = fullfile(pwd, 'Research_Radar_Dataset');
imgFolder = fullfile(datasetRoot, 'images_debug');
rawFolder = fullfile(datasetRoot, 'raw_matrix');
if ~exist(datasetRoot, 'dir'), mkdir(datasetRoot); end
if ~exist(imgFolder, 'dir'), mkdir(imgFolder); end
if ~exist(rawFolder, 'dir'), mkdir(rawFolder); end

fprintf('Generating dataset...\n');

txPulse = phased.RectangularWaveform( ...
    'SampleRate', fs, ...
    'PulseWidth', pulseWidth, ...
    'PRF', prf, ...
    'OutputFormat', 'Pulses', ...
    'NumPulses', 1);

fullPulse = txPulse();
activeSamps = round(pulseWidth * fs);
basePulse = fullPulse(1:activeSamps);

maxRange = 2500;
maxLag = round((2 * maxRange / c) * fs);
windowLen = maxLag + activeSamps;
maxSamplesPerPRI = floor(fs / prf);
if windowLen > maxSamplesPerPRI, windowLen = maxSamplesPerPRI; end

rdProc = phased.RangeDopplerResponse( ...
    'SampleRate', fs, ...
    'PropagationSpeed', c, ...
    'DopplerFFTLengthSource', 'Property', ...
    'DopplerFFTLength', 256, ...
    'RangeMethod', 'Matched filter', ...
    'PRFSource', 'Property', ...
    'PRF', prf);

aircrafts = {
    'F-16',  300, 5.0, 1, 1, 400, 1.5, 0;
    'F-15',  280, 8.0, 2, 1, 350, 1.8, 0;
    'Su-57', 320, 2.0, 3, 1, 450, 1.2, 0;
    'C-130', 150, 25.0, 4, 2, 80,  5.0, 4;
    'MQ-9',  180, 3.0, 5, 2, 100, 3.5, 3;
    'Drone', 50,  0.5, 6, 2, 200, 6.0, 2;
};

calibRCS = 25.0;
calibPulse = sqrt(calibRCS) * basePulse;
calibIQ = calibPulse * ones(1, numPulses);
if size(calibIQ,1) < windowLen
    calibIQ(windowLen, numPulses) = 0;
end
[calibResp, ~, ~] = rdProc(calibIQ, basePulse);
systemRefAmp = max(abs(calibResp(:)));

fprintf('Calibration done.\n');

for i = 1:numSamples
    idx = randi(size(aircrafts, 1));
    name = aircrafts{idx, 1};
    velocity = aircrafts{idx, 2};
    rcs = aircrafts{idx, 3};
    classID = aircrafts{idx, 4};
    mdType = aircrafts{idx, 5};
    modFreq = aircrafts{idx, 6};
    modIdx = aircrafts{idx, 7};
    numBlades = aircrafts{idx, 8};

    theta = deg2rad(60 * rand);
    v_radial = velocity * cos(theta);

    t_slow = (0:numPulses-1) / prf;

    dopplerFreq = 2 * v_radial * fc / c;
    phase_bulk = 2 * pi * dopplerFreq * t_slow;

    phase_md = zeros(size(t_slow));
    amp_md = ones(size(t_slow));

    if mdType == 1
        phase_md = modIdx * sin(2 * pi * modFreq * t_slow);
        amp_md = 1 + 0.05 * sin(2 * pi * (modFreq/2) * t_slow);
    elseif mdType == 2
        for b = 1:numBlades
            phase_offset = (b-1) * (2*pi/numBlades);
            phase_md = phase_md + (modIdx/numBlades) * ...
                       sin(2 * pi * modFreq * t_slow + phase_offset);
        end
        raw_am = 1 + 0.1 * abs(sin(2 * pi * numBlades * modFreq * t_slow));
        amp_md = raw_am / mean(raw_am);
    end

    total_phase = exp(1j * (phase_bulk + phase_md));

    targetRange = 1000 + 500*rand;
    tau = 2 * targetRange / c;
    delaySamps = round(tau * fs);

    amp_scale = sqrt(rcs);
    rxPulse = amp_scale * [zeros(delaySamps, 1); basePulse];

    if length(rxPulse) < windowLen
        rxPulse(windowLen) = 0;
    else
        rxPulse = rxPulse(1:windowLen);
    end

    iqMatrix = rxPulse * (amp_md .* total_phase);

    clutter = 0.05 * (randn(size(iqMatrix)) + 1j*randn(size(iqMatrix)));
    b_filt = ones(5,1)/5;
    clutter = filter(b_filt, 1, clutter, [], 2);

    noiseFloor = 0.01 * (randn(size(iqMatrix)) + 1j*randn(size(iqMatrix)));
    iqSig = iqMatrix + clutter + noiseFloor;

    [resp, ~, ~] = rdProc(iqSig, basePulse);
    magSpec = abs(resp);

    logSpec = 20 * log10(magSpec + 1e-9);

    p_low  = prctile(logSpec(:), 5);
    p_high = prctile(logSpec(:), 95);
    if p_high <= p_low
        p_high = p_low + 1e-6;
    end

    imgNorm = (logSpec - p_low) / (p_high - p_low);
    imgNorm = min(max(imgNorm, 0), 1);

    imgResized = imresize(imgNorm, imgSize, 'box');

    imgUint8 = uint8(imgResized * 255);
    imwrite(imgUint8, fullfile(imgFolder, ...
        sprintf('spec_%04d_c%d.png', i, classID)));

    gan_matrix = (imgResized * 2) - 1;
    save(fullfile(rawFolder, sprintf('data_%04d.mat', i)), ...
         'gan_matrix', 'iqSig', 'classID', 'name', '-v7');

    if mod(i, 500) == 0
        fprintf('%d / %d\n', i, numSamples);
    end
end

fprintf('Done.\n');
