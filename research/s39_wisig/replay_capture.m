function replay_capture(root)
% Replay one screened capture without losing equalization failures or identity.
set(groot,'defaultFigureVisible','off');
cd(fullfile(root,'detect'));
addpath(fullfile(root,'upstream','matlab_detection_screening'));
addpath(fullfile(root,'adapted'),'-begin');
split_signal_fixed('node23-5/');

statistics = load('statistics/node23-5/20-15.mat','endpoints');
screened = load('packets/node23-5/packets_20-15.mat','packet_log');
x = read_complex_binary('data/node23-5/capture/capture.dat');
assert(numel(x)==12800000);
starts = real(statistics.endpoints);
stops = imag(statistics.endpoints);
assert(all(starts>=1 & stops<=numel(x) & starts<=stops));

n = numel(screened.packet_log);
assert(n>0,'No packets passed the upstream screen.');
raw = complex(zeros(n,256,'single'));
eq = complex(nan(n,256,'single'));
offsets = zeros(n,2);
matches = zeros(n,1);
eq_lengths = zeros(n,1);
eq_status = strings(n,1);
for row = 1:n
    pkt = screened.packet_log{row};
    assert(numel(pkt)>=256 && all(isfinite(pkt)));
    candidates = find(stops-starts+1==numel(pkt) & reshape(x(starts),size(starts))==pkt(1));
    exact = candidates(arrayfun(@(k) isequal(x(starts(k):stops(k)),pkt),candidates));
    matches(row) = numel(exact);
    if numel(exact)~=1
        eq_status(row) = "unresolved_identity";
        continue;
    end

    offsets(row,:) = [starts(exact),stops(exact)];
    raw(row,:) = pkt(1:256).';
    equalized = equalize_channel(pkt);
    eq_lengths(row) = numel(equalized);
    eq_status(row) = "short_or_empty";
    if ~all(isfinite(equalized))
        eq_status(row) = "nonfinite";
    elseif numel(equalized)>=256
        eq(row,:) = equalized(1:256).';
        eq_status(row) = "ok";
    end
end

release = version('-release');
detected = numel(starts);
status = table(offsets(:,1),offsets(:,2),matches,eq_lengths,eq_status, ...
    'VariableNames',{'start','stop','matches','eq_length','eq_status'});
writetable(status,fullfile(root,'screened.csv'));
save(fullfile(root,'VIEWS.mat'),'raw','eq','release','detected');
fprintf('Detected %d, screened %d, unique provenance %d, EQ success %d\n', ...
    detected,n,nnz(matches==1),nnz(eq_status=="ok"));
end
