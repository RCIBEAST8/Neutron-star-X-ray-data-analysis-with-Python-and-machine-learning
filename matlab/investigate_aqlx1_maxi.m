%% investigate_aqlx1_maxi.m
% Aql X-1, what actually are the day to day "fluctuations" in the MAXI light curve?
%
% code breakdown
% splits the variability into
%   1. counting noise           - quiet-time scatter vs the quoted error bars
%   2. faint outbursts          - multi-day low level activity between the big outbursts
%   3. single-orbit spikes      - one scan way above both neighbours
% and uses the spectral shape (2-4 / 4-10 / 10-20 keV) to separate real source activity
% (soft, like Aql X-1) from hard, background-like artefacts
%
% known events used as a check
%   ATel #14079 - MAXI superburst, 2020-10-08 16:59 UT
%   ATel #17657 - MAXI type-I burst, 2026-02-08 23:56 UT
%
% any setting can be overridden by defining the variable before running

%%                      settings                      %%
if ~exist('dataDir','var'),     dataDir  = fullfile(fileparts(mfilename('fullpath')), 'data'); end
if ~exist('maxiFile','var'),    maxiFile = fullfile(dataDir, 'maxi 1 orbit Light curve.txt'); end
if ~exist('ztfDir','var'),      ztfDir   = fullfile(dataDir, 'ZTF AQL-X1 DATA'); end
if ~exist('saveOutputs','var'), saveOutputs = false; end
if ~exist('outDir','var'),      outDir   = fullfile(fileparts(mfilename('fullpath')), 'output'); end

actRate   = 0.05;   % 5 day mean 2-20 keV rate that counts as active (ph/cm^2/s)
actSig    = 5;      % and it has to be this many sigma above zero
majorPeak = 0.5;    % 5 day peak above this = major outburst
minFaintD = 5;      % faint episodes shorter than this aren't trusted
spikeSig  = 6;      % a spike has to sit this many sigma above its neighbours
hardFrac  = 0.25;   % 10-20 keV share of 2-20 keV above this = hard / background-like
srcRA = 287.8167; srcDec = 0.585;                    % Aql X-1 position (deg)
knownT = [59130.7076, 61079.9972];                    % ATel 14079, ATel 17657
knownL = {'superburst (ATel 14079)', 'burst (ATel 17657)'};
bandName = {'2-20','2-4','4-10','10-20'};

%%                     load MAXI                      %%
% MJD | 2-20 keV rate, err | 2-4 | 4-10 | 10-20   (ph/cm^2/s)
M = readmatrix(maxiFile, 'FileType', 'text', 'CommentStyle', '#');
M = M(all(isfinite(M(:,1:9)), 2), :);
badErr = any(M(:,[3 5 7 9]) <= 0, 2);                 % a few rows have 0 error in a sub-band
M = M(~badErr, :);
t = M(:,1); R = M(:,2:2:8); E = M(:,3:2:9);
el = sunElong(t, srcRA, srcDec);                      % Sun to Aql X-1 angle for every orbit
fprintf('\nMAXI: %d orbits kept (%d dropped for zero errors)\n', numel(t), nnz(badErr));

%%          find active periods (5 day mean)          %%
d = floor(t); d0 = min(d); idx = d - d0 + 1; nD = max(idx);
days = (d0:d0+nD-1)';
w  = 1 ./ E(:,1).^2;
sw = accumarray(idx, w, [nD 1]);
sr = accumarray(idx, R(:,1).*w, [nD 1]);
SW = conv(sw, ones(5,1), 'same');
SR = conv(sr, ones(5,1), 'same');
m5 = SR ./ SW;  e5 = 1 ./ sqrt(SW);
activeDay = SW > 0 & m5 > actRate & m5 ./ e5 > actSig;
act = conv(double(activeDay), ones(11,1), 'same') > 0;   % pads each active day by +-5 d
quiet = ~act(idx);

% group active days into episodes (gaps of <= 2 days bridged)
ia = find(activeDay);
grp = cumsum([true; diff(ia) > 3]);
nE = max([grp; 0]);
ep = table('Size', [nE 9], ...
    'VariableTypes', {'double','double','double','double','double','double','double','double','string'}, ...
    'VariableNames', {'mjdStart','mjdEnd','days','peak5d','HR_4_10_over_2_4','frac_10_20','sunElong','ztfDetInside','class'});
for k = 1:nE
    dd = ia(grp == k);
    a = days(dd(1)); b = days(dd(end));
    in = t >= a & t < b + 1;
    ep.mjdStart(k) = a; ep.mjdEnd(k) = b; ep.days(k) = b - a + 1;
    ep.peak5d(k) = max(m5(dd));
    ep.HR_4_10_over_2_4(k) = sum(R(in,3)) / sum(R(in,2));
    ep.frac_10_20(k) = sum(R(in,4)) / sum(R(in,1));
    ep.sunElong(k) = mean(el(in));
    % sorts each episode into major / short / faint source-like / faint hard
    if ep.peak5d(k) >= majorPeak
        ep.class(k) = "major outburst";
    elseif ep.days(k) < minFaintD
        ep.class(k) = "short (unconfirmed)";
    elseif ep.frac_10_20(k) < hardFrac
        ep.class(k) = "faint, source-like";
    else
        ep.class(k) = "faint, hard (suspect)";
    end
end

%%                 1. counting noise                  %%
fprintf('\n===== 1. Quiet-time noise (active periods +-5 d masked, %d orbits) =====\n', nnz(quiet));
fprintf('  z = rate/error. Pure counting noise -> std(z) = 1.\n');
for b = 1:4
    z = R(quiet,b) ./ E(quiet,b);
    wb = 1 ./ E(quiet,b).^2;
    s1 = accumarray(idx(quiet), wb, [nD 1]);
    s2 = accumarray(idx(quiet), R(quiet,b).*wb, [nD 1]);
    kk = s1 > 0;
    zd = (s2(kk) ./ s1(kk)) .* sqrt(s1(kk));
    fprintf('  %-6s keV  orbit std(z) %.2f (robust %.2f) | daily std(z) %.2f (robust %.2f) | %2.0f%% of orbits negative\n', ...
        bandName{b}, std(z), 1.4826*median(abs(z - median(z))), std(zd), 1.4826*median(abs(zd - median(zd))), ...
        100*mean(R(quiet,b) < 0));
end

%%                    2. episodes                     %%
fprintf('\n===== 2. Activity episodes (5-d mean > %.2f and > %g sigma) =====\n', actRate, actSig);
cls = ["major outburst","faint, source-like","faint, hard (suspect)","short (unconfirmed)"];
for c = cls, fprintf('  %-22s : %d\n', c, nnz(ep.class == c)); end

% ZTF: detections vs non-detections inside each faint episode (alert stream only)
ztfDet = readZTFmjd(fullfile(ztfDir, 'detections.csv'));
ztfFP  = readZTFmjd(fullfile(ztfDir, 'forced_photometry.csv'));
ztfND  = readZTFmjd(fullfile(ztfDir, 'non_detections.csv'));
for k = 1:nE
    ep.ztfDetInside(k) = nnz(ztfDet.mjd >= ep.mjdStart(k) & ztfDet.mjd <= ep.mjdEnd(k));
end
fprintf('\n  Faint episodes >= %d d:\n', minFaintD);
fprintf('  %-13s %5s %6s %6s %7s %6s   %-22s %s\n', 'MJD', 'days', 'peak', 'HR', '10-20%', 'Sun', 'class', 'ZTF det/nondet inside | before');
fe = find(contains(ep.class, "faint"));
for k = fe'
    a = ep.mjdStart(k); b = ep.mjdEnd(k); L = b - a + 1;
    nIn = nnz(ztfND.mjd >= a & ztfND.mjd <= b);
    dBe = nnz(ztfDet.mjd >= a-L-10 & ztfDet.mjd <= a-10);
    nBe = nnz(ztfND.mjd >= a-L-10 & ztfND.mjd <= a-10);
    fprintf('  %5d-%5d  %5d %6.2f %6.2f %6.0f%% %5.0f deg %-22s %d/%d | %d/%d\n', a, b, L, ep.peak5d(k), ...
        ep.HR_4_10_over_2_4(k), 100*ep.frac_10_20(k), ep.sunElong(k), ep.class(k), ep.ztfDetInside(k), nIn, dBe, nBe);
end

%%               3. single-orbit spikes               %%
i  = (2:numel(t)-1)';
ok = abs(t(i-1) - t(i)) < 0.2 & abs(t(i+1) - t(i)) < 0.2;
base = (R(i-1,1) + R(i+1,1)) / 2;
se   = sqrt(E(i,1).^2 + ((E(i-1,1) + E(i+1,1))/2).^2);
% a spike has to be way above the average of both neighbours, more than double it,
% and above each neighbour on its own
c1 = (R(i,1) - base) ./ se > spikeSig;
c2 = R(i,1) > 2*max(base, 0.05);
c3 = (R(i,1) - R(i-1,1)) ./ hypot(E(i,1), E(i-1,1)) > 5 & (R(i,1) - R(i+1,1)) ./ hypot(E(i,1), E(i+1,1)) > 5;
s  = i(ok & c1 & c2 & c3);
sp = table(t(s), R(s,1), E(s,1), R(s-1,1), R(s+1,1), R(s,2), R(s,3), R(s,4), ...
    R(s,3)./R(s,2), R(s,4)./R(s,1), el(s), act(idx(s)), ...
    'VariableNames', {'mjd','rate','err','prevOrbit','nextOrbit','r2_4','r4_10','r10_20', ...
    'HR_4_10_over_2_4','frac_10_20','sunElong','duringActivity'});
sp.class = repmat("source-like (burst candidate)", height(sp), 1);
sp.class(sp.frac_10_20 >= hardFrac) = "hard (likely background)";
sp.utc = string(datetime(sp.mjd, 'ConvertFrom', 'mjd', 'Format', 'yyyy-MM-dd HH:mm'));

fprintf('\n===== 3. Single-orbit spikes (> %g sigma above both neighbours, > 2x) : %d =====\n', spikeSig, height(sp));
fprintf('  %-17s %-11s %5s %5s %6s %7s %s\n', 'UTC', 'MJD', 'rate', 'HR', '10-20%', 'active', 'class');
for k = 1:height(sp)
    fprintf('  %-17s %11.4f %5.2f %5.2f %5.0f%% %7d %s\n', sp.utc(k), sp.mjd(k), sp.rate(k), ...
        sp.HR_4_10_over_2_4(k), 100*sp.frac_10_20(k), sp.duringActivity(k), sp.class(k));
end
srcLike = sp.class == "source-like (burst candidate)";
fprintf('  source-like: %d (%d during activity) | hard: %d (%d during activity)\n', nnz(srcLike), ...
    nnz(srcLike & sp.duringActivity), nnz(~srcLike), nnz(~srcLike & sp.duringActivity));
for k = 1:numel(knownT)
    [dt, j] = min(abs(t - knownT(k)));
    hit = any(abs(sp.mjd - knownT(k)) < 0.05);
    fprintf('  check %-24s: nearest orbit %.4f (|dt| %.3f d), rate %.2f, flagged as spike: %d\n', ...
        knownL{k}, t(j), dt, R(j,1), hit);
end

%%                      figures                       %%
cEp  = struct('major', [0.85 0.85 0.85], 'src', [0.30 0.70 0.35], 'hard', [0.85 0.45 0.10], 'short', [0.75 0.75 0.95]);
cSrc = [0.15 0.45 0.80]; cHard = [0.85 0.30 0.10];

%% fig 1: overview timeline
f1 = figure('Name', 'MAXI variability overview', 'Color', 'w', 'Position', [60 60 1300 520]);
try, theme(f1, 'light'); catch, end
ax = axes(f1); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
kk = sw > 0;
dm = sr(kk) ./ sw(kk);
yl = [-0.3 4.6];
for k = 1:nE
    switch ep.class(k)
        case "major outburst",        cc = cEp.major;
        case "faint, source-like",    cc = cEp.src;
        case "faint, hard (suspect)", cc = cEp.hard;
        otherwise,                    continue          % short episodes not shaded (too many, unconfirmed)
    end
    patch(ax, [ep.mjdStart(k) ep.mjdEnd(k)+1 ep.mjdEnd(k)+1 ep.mjdStart(k)], yl([1 1 2 2]), cc, ...
        'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');
end
plot(ax, days(kk), dm, '.', 'Color', [0.2 0.2 0.2], 'MarkerSize', 4, 'DisplayName', 'MAXI 2-20 keV daily');
plot(ax, sp.mjd(srcLike), sp.rate(srcLike), 'v', 'Color', cSrc, 'MarkerFaceColor', cSrc, 'DisplayName', 'spike: source-like');
plot(ax, sp.mjd(~srcLike), sp.rate(~srcLike), 'v', 'Color', cHard, 'MarkerFaceColor', cHard, 'DisplayName', 'spike: hard');
for k = 1:numel(knownT), xline(ax, knownT(k), '--', knownL{k}, 'Color', [0.4 0 0.6], 'HandleVisibility', 'off', 'FontSize', 8); end
patch(ax, NaN, NaN, cEp.src,  'EdgeColor', 'none', 'DisplayName', 'faint, source-like');
patch(ax, NaN, NaN, cEp.hard, 'EdgeColor', 'none', 'DisplayName', 'faint, hard (suspect)');
patch(ax, NaN, NaN, cEp.major,'EdgeColor', 'none', 'DisplayName', 'major outburst');
ylim(ax, yl); xlim(ax, [min(t)-20 max(t)+20]);
xlabel(ax, 'MJD'); ylabel(ax, '2-20 keV  [ph cm^{-2} s^{-1}]');
title(ax, 'Aql X-1 MAXI: classified activity episodes and single-orbit spikes');
legend(ax, 'Location', 'northoutside', 'NumColumns', 6);

%% fig 2: diagnostics
f2 = figure('Name', 'MAXI variability diagnostics', 'Color', 'w', 'Position', [80 80 1250 900]);
try, theme(f2, 'light'); catch, end
tl = tiledlayout(f2, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
edges = -6:0.25:6; xc = edges(1:end-1) + 0.125;
for b = [1 4]
    z = R(quiet,b) ./ E(quiet,b);
    histogram(ax, z, edges, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.4, ...
        'DisplayName', sprintf('%s keV (std %.2f)', bandName{b}, std(z)));
end
plot(ax, xc, exp(-xc.^2/2)/sqrt(2*pi), 'k--', 'DisplayName', 'pure noise N(0,1)');
set(ax, 'YScale', 'log'); ylim(ax, [1e-5 1]);
xlabel(ax, 'quiet-time orbit rate / error'); ylabel(ax, 'pdf');
title(ax, '(a) Quiet-time scatter is mostly counting noise');
legend(ax, 'Location', 'south');

ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
nOrb = accumarray(idx(quiet), 1, [nD 1]);
s1 = accumarray(idx(quiet), 1 ./ E(quiet,4).^2, [nD 1]);
s2 = accumarray(idx(quiet), R(quiet,4) ./ E(quiet,4).^2, [nD 1]);
kk = s1 > 0;
scatter(ax, nOrb(kk) + 0.3*(rand(nnz(kk),1)-0.5), s2(kk)./s1(kk), 6, 1./sqrt(s1(kk)), 'filled');
cb = colorbar(ax); cb.Label.String = 'daily error';
xlabel(ax, 'orbits (scans) in the day'); ylabel(ax, 'daily 10-20 keV rate');
title(ax, '(b) 10-20 keV outliers come from low-coverage days');

ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
plot(ax, sp.HR_4_10_over_2_4(srcLike & sp.duringActivity), sp.frac_10_20(srcLike & sp.duringActivity), 'o', ...
    'Color', cSrc, 'MarkerFaceColor', cSrc, 'DisplayName', 'source-like, during activity');
plot(ax, sp.HR_4_10_over_2_4(srcLike & ~sp.duringActivity), sp.frac_10_20(srcLike & ~sp.duringActivity), 'o', ...
    'Color', cSrc, 'DisplayName', 'source-like, quiescence');
plot(ax, sp.HR_4_10_over_2_4(~srcLike & sp.duringActivity), sp.frac_10_20(~srcLike & sp.duringActivity), 's', ...
    'Color', cHard, 'MarkerFaceColor', cHard, 'DisplayName', 'hard, during activity');
plot(ax, sp.HR_4_10_over_2_4(~srcLike & ~sp.duringActivity), sp.frac_10_20(~srcLike & ~sp.duringActivity), 's', ...
    'Color', cHard, 'DisplayName', 'hard, quiescence');
inb = m5(idx) > 1;
xline(ax, median(R(inb,3)./R(inb,2)), ':', 'bright-outburst HR', 'HandleVisibility', 'off', 'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'middle');
yline(ax, hardFrac, '--', 'hard cut', 'HandleVisibility', 'off');
xlim(ax, [0.4 2.2]);
xlabel(ax, 'HR = (4-10 keV)/(2-4 keV)'); ylabel(ax, '10-20 keV fraction of 2-20 keV');
title(ax, sprintf('(c) %d single-orbit spikes: two populations', height(sp)));
legend(ax, 'Location', 'northwest');

ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
patch(ax, [0 45 45 0], [-0.1 -0.1 0.8 0.8], [1 0.93 0.75], 'EdgeColor', 'none', 'DisplayName', 'within 45° of Sun (Dec-Feb)');
fs = ep.class == "faint, source-like"; fh = ep.class == "faint, hard (suspect)";
scatter(ax, ep.sunElong(fs), ep.frac_10_20(fs), 8*ep.days(fs), cEp.src, 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'faint, source-like');
scatter(ax, ep.sunElong(fh), ep.frac_10_20(fh), 8*ep.days(fh), cEp.hard, 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'faint, hard (suspect)');
yline(ax, hardFrac, '--', 'HandleVisibility', 'off');
xlim(ax, [0 180]); ylim(ax, [-0.1 0.8]);
xlabel(ax, 'mean Sun-source angle [deg]'); ylabel(ax, '10-20 keV fraction');
title(ax, '(d) Faint episodes >= 5 d (marker size = duration)');
legend(ax, 'Location', 'northeast');

%% fig 3: zoom gallery
f3 = figure('Name', 'MAXI zooms', 'Color', 'w', 'Position', [100 100 1250 900]);
try, theme(f3, 'light'); catch, end
tl = tiledlayout(f3, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
bc = [0.85 0.15 0.15; 0.15 0.6 0.2; 0.2 0.3 0.85];

ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
k = t > knownT(1) - 1 & t < knownT(1) + 1.2;
errorbar(ax, t(k), R(k,1), E(k,1), 'o-', 'Color', [0.2 0.2 0.2], 'MarkerFaceColor', [0.2 0.2 0.2], 'MarkerSize', 4, 'CapSize', 0);
xline(ax, knownT(1), '--', 'ATel 14079 onset', 'Color', [0.4 0 0.6]);
xlabel(ax, 'MJD'); ylabel(ax, '2-20 keV (1 orbit)');
title(ax, '(a) 2020-10-08 superburst recovered in the MAXI data');

ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
win = [59240 59340];
kk = days >= win(1) & days <= win(2) & sw > 0;
yyaxis(ax, 'left');
errorbar(ax, days(kk), sr(kk)./sw(kk), 1./sqrt(sw(kk)), '.', 'CapSize', 0);
ylabel(ax, 'MAXI 2-20 keV daily');
yyaxis(ax, 'right');
zz = ztfDet.mjd >= win(1) & ztfDet.mjd <= win(2);
plot(ax, ztfDet.mjd(zz), ztfDet.mag(zz), 'o', 'MarkerFaceColor', [0.85 0.15 0.15]);
zn = ztfND.mjd >= win(1) & ztfND.mjd <= win(2);
plot(ax, ztfND.mjd(zn), ztfND.lim(zn), 'v', 'Color', [0.6 0.6 0.6]);
set(ax, 'YDir', 'reverse'); ylabel(ax, 'ZTF mag (o) / diff-limit (v)');
xlim(ax, win); xlabel(ax, 'MJD');
title(ax, '(b) 2021 faint outburst: ZTF switches from limits to detections');

ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
[~, j] = max(sp.frac_10_20 .* (sp.rate > 0.4) .* ~sp.duringActivity);
tc = sp.mjd(j);
k = t > tc - 0.5 & t < tc + 0.5;
for b = 2:4
    errorbar(ax, t(k), R(k,b), E(k,b), 'o-', 'Color', bc(b-1,:), 'MarkerFaceColor', bc(b-1,:), 'MarkerSize', 3, ...
        'CapSize', 0, 'DisplayName', [bandName{b} ' keV']);
end
xlabel(ax, 'MJD'); ylabel(ax, 'rate (1 orbit)'); legend(ax, 'Location', 'northwest');
title(ax, sprintf('(c) Hard quiet-time spike %s: 10-20 keV dominated', sp.utc(j)));

ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
fhK = find(fh); fsK = find(fs);
a2 = fsK(ep.mjdStart(fsK) == 57062);                  % 2015 faint outburst (soft)
a1 = fhK(ep.mjdStart(fhK) == 60666);                  % 2024-25 winter episode (hard)
if isempty(a2), [~, a2] = max(ep.days(fsK)); a2 = fsK(a2); end
if isempty(a1), [~, a1] = max(ep.days(fhK)); a1 = fhK(a1); end
pick = [a2 a1]; ls = {'-', '--'}; tag = {'source-like', 'hard / suspect'};
for q = 1:2
    a = ep.mjdStart(pick(q)); b = ep.mjdEnd(pick(q));
    for bb = [2 4]
        s1 = accumarray(idx, 1./E(:,bb).^2, [nD 1]); s2 = accumarray(idx, R(:,bb)./E(:,bb).^2, [nD 1]);
        kk = days >= a - 10 & days <= a + 35 & s1 > 0;
        plot(ax, days(kk) - a, s2(kk)./s1(kk), ls{q}, 'Color', bc((bb>2)*2 + 1,:), 'LineWidth', 1.3, ...
            'DisplayName', sprintf('%s keV, MJD %d (%s)', bandName{bb}, a, tag{q}));
    end
end
yline(ax, 0, ':', 'HandleVisibility', 'off');
xlabel(ax, 'days since episode start'); ylabel(ax, 'daily rate');
legend(ax, 'Location', 'northeast', 'FontSize', 8);
title(ax, '(d) Real faint outburst (soft) vs suspect winter episode (hard)');

%%                        save                        %%
if saveOutputs
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    exportgraphics(f1, fullfile(outDir, 'maxi_variability_overview.png'), 'Resolution', 150);
    exportgraphics(f2, fullfile(outDir, 'maxi_variability_diagnostics.png'), 'Resolution', 150);
    exportgraphics(f3, fullfile(outDir, 'maxi_variability_zooms.png'), 'Resolution', 150);
    writetable(sp, fullfile(outDir, 'maxi_single_orbit_spikes.csv'));
    writetable(ep, fullfile(outDir, 'maxi_activity_episodes.csv'));
    fprintf('\nSaved figures and tables to %s\n', outDir);
end

%%                  local functions                   %%
% reads a ZTF csv, works for both detections (mag_corr) and non-detections (diffmaglim)
function T = readZTFmjd(f)
    opts = detectImportOptions(f);
    v = opts.VariableNames;
    if any(strcmp(v, 'mag_corr'))
        opts.SelectedVariableNames = {'mjd','fid','mag_corr'};
        T = readtable(f, opts); T.Properties.VariableNames = {'mjd','fid','mag'};
    else
        opts.SelectedVariableNames = {'mjd','fid','diffmaglim'};
        T = readtable(f, opts); T.Properties.VariableNames = {'mjd','fid','lim'};
    end
    T = T(isfinite(T.mjd), :);
end

function el = sunElong(mjd, raDeg, decDeg)
    % low-precision solar position (good to ~0.1 deg), angle Sun-source in degrees
    n   = mjd - 51544.5;
    L   = deg2rad(mod(280.460 + 0.9856474*n, 360));
    g   = deg2rad(mod(357.528 + 0.9856003*n, 360));
    lam = L + deg2rad(1.915)*sin(g) + deg2rad(0.020)*sin(2*g);
    ob  = deg2rad(23.439);
    sra = atan2(cos(ob).*sin(lam), cos(lam));
    sde = asin(sin(ob).*sin(lam));
    ra = deg2rad(raDeg); de = deg2rad(decDeg);
    el = rad2deg(acos(sin(de).*sin(sde) + cos(de).*cos(sde).*cos(ra - sra)));
end
