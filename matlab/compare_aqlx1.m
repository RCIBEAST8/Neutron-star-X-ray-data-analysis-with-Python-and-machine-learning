%% compare_aqlx1.m
% Aql X-1, optical vs X-ray comparison
%
% code breakdown
% loads three light curves
%   ZTF (ALeRCE export, object ZTF18accedau) - g/r/i optical magnitudes
%   MAXI/GSC 1-orbit light curve           - 2-20 keV (plus 2-4/4-10/10-20)
%   Swift/BAT 1-orbit light curve           - 15-50 keV
% then makes
%   fig 1 - stacked light curves on one shared (linked) MJD axis
%   fig 2 - ZTF flux vs the X-ray rate at the same time, per band, with correlations
% and prints a summary of time coverage, overlaps and match counts in the command window
%
% any setting below can be overridden by defining the variable before running,
% e.g.  binDays = 0; run('compare_aqlx1.m')

%%                      settings                      %%
if ~exist('dataDir','var'),       dataDir = fullfile(fileparts(mfilename('fullpath')), 'data'); end
if ~exist('ztfDir','var'),        ztfDir  = fullfile(dataDir, 'ZTF AQL-X1 DATA'); end
if ~exist('batFile','var'),       batFile  = fullfile(dataDir, 'swift-bat 1 orbit Light curve.txt'); end
if ~exist('maxiFile','var'),      maxiFile = fullfile(dataDir, 'maxi 1 orbit Light curve.txt'); end

if ~exist('binDays','var'),       binDays = 1;      end   % X-ray bin width in days, 0 = raw 1-orbit points
if ~exist('maxSepDays','var'),    maxSepDays = 0.5; end   % max time gap (days) for a ZTF <-> X-ray match
if ~exist('maxiBand','var'),      maxiBand = 1;     end   % 1=2-20, 2=2-4, 3=4-10, 4=10-20 keV
if ~exist('showNonDet','var'),    showNonDet = false; end % also plot the ZTF non-detection limits
if ~exist('zoomToZTF','var'),     zoomToZTF = true; end   % start the x axis where the ZTF data starts
if ~exist('saveOutputs','var'),   saveOutputs = false; end % true = save figures and match tables to outDir
if ~exist('outDir','var'),        outDir = fullfile(fileparts(mfilename('fullpath')), 'output'); end

bandName  = {'g','r','i'};                                 % ZTF filter id 1/2/3
bandCol   = [0.10 0.60 0.20; 0.85 0.15 0.15; 0.45 0.25 0.10];
maxiLabel = {'2-20 keV','2-4 keV','4-10 keV','10-20 keV'};

%%                        ZTF                         %%
% uses mag_corr, the reference-corrected (total flux) mag. the raw 'mag' column is the
% difference-image mag and is NOT the star's actual brightness, so it isn't used
ztfDet = readZTF(fullfile(ztfDir, 'detections.csv'));
ztfFP  = readZTF(fullfile(ztfDir, 'forced_photometry.csv'));
ztfDet.src = repmat("det", height(ztfDet), 1);
ztfFP.src  = repmat("fp",  height(ztfFP),  1);
ztf = sortrows([ztfDet; ztfFP], 'mjd');
ztf.flux_mJy  = 3631e3 * 10.^(-0.4*ztf.mag);              % AB mag -> mJy (no dereddening)
ztf.eflux_mJy = 0.4*log(10) * ztf.flux_mJy .* ztf.emag;

optsND = detectImportOptions(fullfile(ztfDir, 'non_detections.csv'));
optsND.SelectedVariableNames = {'mjd','fid','diffmaglim'};
ztfND = readtable(fullfile(ztfDir, 'non_detections.csv'), optsND);

%%                        MAXI                        %%
% MAXI 1-orbit columns (photons/cm^2/s):
%   MJD | 2-20 keV rate, err | 2-4 keV rate, err | 4-10 keV rate, err | 10-20 keV rate, err
M = readmatrix(maxiFile, 'FileType', 'text', 'CommentStyle', '#');
M = M(all(isfinite(M(:,1:9)), 2), :);
maxi.mjd  = M(:,1);
maxi.rate = M(:, 2*maxiBand);
maxi.err  = M(:, 2*maxiBand + 1);

%%                     Swift/BAT                      %%
% BAT columns: TIME(s, Swift MET) RATE ERROR YEAR DAY MJD TIMEDEL STAT_ERR SYS_ERR PCODEFR DATA_FLAG DITHER_FLAG
% the MJD column is whole days only, so time gets rebuilt from MET instead (epoch 2001-01-01 = MJD 51910)
B = readmatrix(batFile, 'FileType', 'text', 'CommentStyle', '#');
B = B(all(isfinite(B(:,1:11)), 2), :);                    % drops cut off / partial rows
B = B(B(:,11) == 0, :);                                   % DATA_FLAG == 0 (good data only)
bat.mjd  = B(:,1)/86400 + 51910;
bat.rate = B(:,2);                                        % counts/cm^2/s (15-50 keV)
bat.err  = B(:,3);

%%               optional time binning                %%
if binDays > 0
    maxiB = binLC(maxi, binDays);
    batB  = binLC(bat,  binDays);
    binTxt = sprintf('%g-d bins', binDays);
else
    maxiB = maxi;  batB = bat;
    binTxt = '1-orbit';
end

%%                  coverage summary                  %%
fprintf('\n===== Aql X-1 data coverage =====\n');
covLine('ZTF (det+forced)', ztf.mjd);
for b = 1:3
    covLine(sprintf('  ZTF %s', bandName{b}), ztf.mjd(ztf.fid == b));
end
covLine('ZTF non-detections', ztfND.mjd);
covLine(['MAXI ' maxiLabel{maxiBand}], maxi.mjd);
covLine('Swift/BAT 15-50 keV', bat.mjd);
ovLine('ZTF  x MAXI', ztf.mjd, maxi.mjd);
ovLine('ZTF  x BAT ', ztf.mjd, bat.mjd);
ovLine('MAXI x BAT ', maxi.mjd, bat.mjd);

%%            fig 1: stacked light curves             %%
f1 = figure('Name', 'Aql X-1 light curves', 'Color', 'w', 'Position', [80 80 1200 850]);
try, theme(f1, 'light'); catch, end              % forces the light theme so it's readable if MATLAB is in dark mode
tl = tiledlayout(f1, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'Aql X-1: ZTF optical vs MAXI / Swift-BAT X-ray');

ax1 = nexttile(tl); hold(ax1, 'on'); box(ax1, 'on');
hLeg = gobjects(0); legTxt = {};
for b = 1:3
    k = ztf.fid == b & ztf.src == "det";
    if any(k)
        h = errorbar(ax1, ztf.mjd(k), ztf.mag(k), ztf.emag(k), 'o', 'Color', bandCol(b,:), ...
            'MarkerFaceColor', bandCol(b,:), 'MarkerSize', 4, 'CapSize', 0);
        hLeg(end+1) = h; legTxt{end+1} = ['ZTF ' bandName{b}]; %#ok<SAGROW>
    end
    k = ztf.fid == b & ztf.src == "fp";
    if any(k)
        errorbar(ax1, ztf.mjd(k), ztf.mag(k), ztf.emag(k), 's', 'Color', bandCol(b,:), ...
            'MarkerSize', 5, 'CapSize', 0);
    end
    if showNonDet
        k = ztfND.fid == b;
        plot(ax1, ztfND.mjd(k), ztfND.diffmaglim(k), 'v', 'Color', [bandCol(b,:) 0.3], 'MarkerSize', 3);
    end
end
set(ax1, 'YDir', 'reverse');
ylabel(ax1, 'mag (AB, mag\_corr)');
legend(ax1, hLeg, legTxt, 'Location', 'best');
text(ax1, 0.01, 0.05, 'filled = alert detections, open squares = forced photometry', ...
    'Units', 'normalized', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);

ax2 = nexttile(tl); hold(ax2, 'on'); box(ax2, 'on');
errorbar(ax2, maxiB.mjd, maxiB.rate, maxiB.err, '.', 'Color', [0.1 0.3 0.8], 'CapSize', 0);
yline(ax2, 0, ':');
ylabel(ax2, 'ph cm^{-2} s^{-1}');
title(ax2, sprintf('MAXI %s (%s)', maxiLabel{maxiBand}, binTxt));

ax3 = nexttile(tl); hold(ax3, 'on'); box(ax3, 'on');
if ~isempty(batB.mjd)
    errorbar(ax3, batB.mjd, batB.rate, batB.err, '.', 'Color', [0.5 0.1 0.6], 'CapSize', 0);
end
yline(ax3, 0, ':');
ylabel(ax3, 'counts cm^{-2} s^{-1}');
title(ax3, sprintf('Swift/BAT 15-50 keV (%s)', binTxt));
xlabel(ax3, 'MJD');

linkaxes([ax1 ax2 ax3], 'x');
allT = [ztf.mjd; maxi.mjd; bat.mjd];
if zoomToZTF && ~isempty(ztf.mjd)
    xlim(ax1, [min(ztf.mjd)-30, max([ztf.mjd; maxi.mjd])+30]);
else
    xlim(ax1, [min(allT)-30, max(allT)+30]);
end
for ax = [ax1 ax2 ax3], grid(ax, 'on'); end

%%              cross-match ZTF to X-ray              %%
mMAXI = matchLC(ztf, maxiB, maxSepDays);
mBAT  = matchLC(ztf, batB,  maxSepDays);

fprintf('\n===== ZTF <-> X-ray matches (|dt| <= %.2f d, X-ray %s) =====\n', maxSepDays, binTxt);
xr = {mMAXI, mBAT};  xrName = {['MAXI ' maxiLabel{maxiBand}], 'BAT 15-50 keV'};
for j = 1:2
    for b = 1:3
        k = xr{j}.fid == b;
        [rP, rS] = corrPair(xr{j}.flux_mJy(k), xr{j}.xrate(k));
        fprintf('  %-16s  ZTF %s : %4d matches   Pearson r = %6.3f   Spearman rho = %6.3f\n', ...
            xrName{j}, bandName{b}, nnz(k), rP, rS);
    end
end

%%         fig 2: optical flux vs X-ray rate          %%
f2 = figure('Name', 'Aql X-1 optical vs X-ray', 'Color', 'w', 'Position', [120 120 1100 450]);
try, theme(f2, 'light'); catch, end
tl2 = tiledlayout(f2, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl2, sprintf('ZTF flux vs simultaneous X-ray rate (|\\Deltat| \\leq %.2g d, X-ray %s)', maxSepDays, binTxt));
for j = 1:2
    ax = nexttile(tl2); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
    hh = gobjects(0); lt = {};
    for b = 1:3
        k = xr{j}.fid == b;
        if any(k)
            hh(end+1) = errorbar(ax, xr{j}.xrate(k), xr{j}.flux_mJy(k), ...
                xr{j}.eflux_mJy(k), xr{j}.eflux_mJy(k), xr{j}.xerr(k), xr{j}.xerr(k), ...
                'o', 'Color', bandCol(b,:), 'MarkerFaceColor', bandCol(b,:), 'MarkerSize', 4, 'CapSize', 0); %#ok<SAGROW>
            lt{end+1} = sprintf('ZTF %s (n=%d)', bandName{b}, nnz(k)); %#ok<SAGROW>
        end
    end
    xlabel(ax, [xrName{j} ' rate']);
    ylabel(ax, 'ZTF flux density [mJy, not dereddened]');
    if isempty(hh)
        text(ax, 0.5, 0.5, sprintf('No overlapping epochs\n(check %s time coverage)', xrName{j}), ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', 'Color', [0.6 0 0]);
    else
        legend(ax, hh, lt, 'Location', 'best');
    end
end

%%                  optional saving                   %%
if saveOutputs
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    exportgraphics(f1, fullfile(outDir, 'aqlx1_lightcurves.png'), 'Resolution', 150);
    xl = xlim(ax1);                                       % also saves the full 2005-2026 view
    xlim(ax1, [min(allT)-30, max(allT)+30]);
    exportgraphics(f1, fullfile(outDir, 'aqlx1_lightcurves_full_history.png'), 'Resolution', 150);
    xlim(ax1, xl);
    exportgraphics(f2, fullfile(outDir, 'aqlx1_optical_vs_xray.png'), 'Resolution', 150);
    writetable(mMAXI, fullfile(outDir, 'aqlx1_ztf_maxi_matches.csv'));
    writetable(mBAT,  fullfile(outDir, 'aqlx1_ztf_bat_matches.csv'));
    fprintf('\nSaved figures and match tables to %s\n', outDir);
end

%%                  local functions                   %%
% reads one ZTF csv, keeps mjd, filter and the corrected mag + error
function T = readZTF(f)
    opts = detectImportOptions(f);
    opts.SelectedVariableNames = {'mjd','fid','mag_corr','e_mag_corr'};
    T = readtable(f, opts);
    T.Properties.VariableNames = {'mjd','fid','mag','emag'};
    T = T(isfinite(T.mjd) & isfinite(T.mag) & isfinite(T.emag), :);
end

function S = binLC(L, w)
    % inverse-variance weighted mean in fixed width time bins
    ok = isfinite(L.rate) & isfinite(L.err) & L.err > 0;
    t = L.mjd(ok); r = L.rate(ok); e = L.err(ok);
    if isempty(t), S = struct('mjd', [], 'rate', [], 'err', []); return; end
    idx = floor((t - floor(min(t))) / w) + 1;
    wt  = 1 ./ e.^2;
    sw  = accumarray(idx, wt);
    has = sw > 0;
    S.mjd  = accumarray(idx, t .* wt) ./ sw;
    S.rate = accumarray(idx, r .* wt) ./ sw;
    S.err  = 1 ./ sqrt(sw);
    S.mjd = S.mjd(has); S.rate = S.rate(has); S.err = S.err(has);
end

function M = matchLC(ztf, X, maxSep)
    % nearest X-ray point to each ZTF epoch, kept if within maxSep days
    vars = {'mjd','fid','mag','emag','flux_mJy','eflux_mJy','xmjd','dt','xrate','xerr'};
    if isempty(X.mjd) || isempty(ztf.mjd)
        M = cell2table(cell(0, numel(vars)), 'VariableNames', vars); return;
    end
    [tx, o] = sort(X.mjd(:));
    rx = X.rate(o); ex = X.err(o);
    [tx, u] = unique(tx); rx = rx(u); ex = ex(u);
    if numel(tx) == 1
        j = ones(height(ztf), 1);
    else
        j = interp1(tx, (1:numel(tx))', ztf.mjd, 'nearest', 'extrap');
    end
    dt = ztf.mjd - tx(j);
    k = abs(dt) <= maxSep;
    M = ztf(k, {'mjd','fid','mag','emag','flux_mJy','eflux_mJy'});
    M.xmjd = tx(j(k)); M.dt = dt(k); M.xrate = rx(j(k)); M.xerr = ex(j(k));
end

function [rP, rS] = corrPair(x, y)
    % Pearson and Spearman without needing the Statistics Toolbox
    rP = NaN; rS = NaN;
    if numel(x) < 3, return; end
    c = corrcoef(x, y); rP = c(1,2);
    c = corrcoef(rankAvg(x), rankAvg(y)); rS = c(1,2);
end

% ranks with ties averaged (for the Spearman correlation)
function r = rankAvg(v)
    v = v(:); [s, o] = sort(v); n = numel(v);
    r = zeros(n, 1); i = 1;
    while i <= n
        j = i;
        while j < n && s(j+1) == s(i), j = j + 1; end
        r(o(i:j)) = (i + j) / 2;
        i = j + 1;
    end
end

% prints the time range a dataset covers
function covLine(name, t)
    if isempty(t)
        fprintf('  %-22s : no data\n', name);
    else
        fprintf('  %-22s : %6d pts   MJD %.1f - %.1f   (%s to %s)\n', name, numel(t), min(t), max(t), ...
            string(datetime(min(t), 'ConvertFrom', 'mjd'), 'yyyy-MM-dd'), ...
            string(datetime(max(t), 'ConvertFrom', 'mjd'), 'yyyy-MM-dd'));
    end
end

% prints where two datasets overlap in time
function ovLine(name, a, b)
    if isempty(a) || isempty(b)
        fprintf('  overlap %s : none (one set empty)\n', name); return;
    end
    lo = max(min(a), min(b)); hi = min(max(a), max(b));
    if lo < hi
        fprintf('  overlap %s : MJD %.1f - %.1f (%.0f d)\n', name, lo, hi, hi - lo);
    else
        fprintf('  overlap %s : NONE - time ranges do not overlap\n', name);
    end
end
