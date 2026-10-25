function ex_5_2_7()
% 5.2.7 3-Dimensional Examples
%   Fig. 15: Algorithm 3_2，計算域 [-pi,pi]^3, 128^3 網格, p = 2。
%   初始猜測: 取 d(x) 的 sigma-等值面（sigma 適度大）所包圍的區域。
%   點雲: 雜訊 torus、兩個互扣的 torus、凹凸 torus，以及（若有檔案）bunny、pig。
% 註: 論文未給前三個點雲的確切參數，以下參數依 Fig. 15 外觀自行設定。
%     bunny / pig 需自備資料：將 bunny.xyz / bunny.txt / bunny.ply（或 pig.*）
%     放在本檔同一資料夾，檔案為每列 x y z（ASCII）；找不到則略過。
close all
p = 2;
maxit = 2000;
rng(1);
n = 128;
[x, K2, h] = setup_grid(n, 3);
taus3 = [0.02 0.01 0.005 0.0025 0.00125];

names = {};
clouds = {};
sigmas = [];
tau1 = [];      % 各例起始 tau：細管或細部需較小的 tau，否則曲面會整個消失（Remark 3）

% (1) 雜訊 torus
N = 6000;
uu = 2*pi*rand(N, 1); vv = 2*pi*rand(N, 1);
P = [(1.2 + 0.5*cos(uu)).*cos(vv), (1.2 + 0.5*cos(uu)).*sin(vv), 0.5*sin(uu)];
names{end+1} = 'noisy torus';
clouds{end+1} = P + 0.02*randn(size(P));
sigmas(end+1) = 0.3;
tau1(end+1) = 0.02;

% (2) 兩個互扣的 torus：一個在 xy 平面、圓心 (-0.6,0,0)；一個在 xz 平面、圓心 (0.6,0,0)
N = 5000; R = 1.2; a = 0.3;
uu = 2*pi*rand(N, 1); vv = 2*pi*rand(N, 1);
PA = [-0.6 + (R + a*cos(uu)).*cos(vv), (R + a*cos(uu)).*sin(vv), a*sin(uu)];
uu = 2*pi*rand(N, 1); vv = 2*pi*rand(N, 1);
PB = [0.6 + (R + a*cos(uu)).*cos(vv), a*sin(uu), (R + a*cos(uu)).*sin(vv)];
names{end+1} = 'two tori';
clouds{end+1} = [PA; PB];
sigmas(end+1) = 0.2;
tau1(end+1) = 0.005;

% (3) 凹凸 torus：管半徑隨角度起伏
N = 15000;
uu = 2*pi*rand(N, 1); vv = 2*pi*rand(N, 1);
rho = 0.5 + 0.15*cos(12*vv) .* (0.6 + 0.4*cos(4*uu));
P = [(1.6 + rho.*cos(uu)).*cos(vv), (1.6 + rho.*cos(uu)).*sin(vv), rho.*sin(uu)];
names{end+1} = 'bumpy torus';
clouds{end+1} = P;
sigmas(end+1) = 0.25;
tau1(end+1) = 0.01;

% (4)(5) bunny / pig（可選）
here = fileparts(mfilename('fullpath'));
for nm = {'bunny', 'pig'}
    P = load_cloud(fullfile(here, nm{1}));
    if isempty(P)
        fprintf('找不到 %s 的點雲檔案，略過。\n', nm{1});
        continue
    end
    P = P - (max(P) + min(P))/2;                % 置中
    P = P * (3.6 / max(max(P) - min(P)));       % 縮放到最大邊長 3.6
    names{end+1} = nm{1}; %#ok<AGROW>
    clouds{end+1} = P; %#ok<AGROW>
    sigmas(end+1) = 0.15; %#ok<AGROW>
    tau1(end+1) = 0.005; %#ok<AGROW>
end

nc = numel(clouds);
figure('Name', 'Fig.15: 3D examples, Algorithm 3_2', 'Position', [50 30 1200 270*nc]);
views = {[30 40], [0 90], [90 0], [0 0]};
vnames = {'', 'xy-view', 'yz-view', 'xz-view'};
for c = 1:nc
    P = clouds{c};
    band = 2*sigmas(c) + 0.2;                   % 距離函數只需在點雲附近精確
    d = distance_to_cloud(P, x, 3, band);
    u0 = fill_enclosed(d < sigmas(c));          % sigma-等值面所包圍的區域
    psi = d.^(p/2);
    t0 = tic;
    [~, phi, info] = algorithm3_2(psi, u0, taus3(taus3 <= tau1(c)), K2, maxit, h, 3, false);
    t = toc(t0);
    fprintf('%-12s: %6d points, %.2fs, %d iterations\n', names{c}, size(P, 1), t, info.iters);
    for v = 1:4
        subplot(nc, 4, (c - 1)*4 + v);
        if v == 1
            ttl = sprintf('%s, %.2fs', names{c}, t);
        else
            ttl = vnames{v};
        end
        plot3d(x, P, phi, ttl, views{v});
    end
    drawnow
end
end

function u = fill_enclosed(mask)
% 回傳 mask 本身加上被 mask 包圍的區域（類似 imfill(mask,'holes')），以 double 0/1 表示。
if exist('imfill') == 2
    u = double(imfill(mask, 'holes'));
    return
end
% 從計算域邊界開始，在 ~mask 中以 6-鄰域往內擴張，得到「外部」區域
ext = false(size(mask));
ext([1 end], :, :) = true; ext(:, [1 end], :) = true; ext(:, :, [1 end]) = true;
ext = ext & ~mask;
while true
    grow = ext;
    grow(2:end, :, :)   = grow(2:end, :, :)   | ext(1:end-1, :, :);
    grow(1:end-1, :, :) = grow(1:end-1, :, :) | ext(2:end, :, :);
    grow(:, 2:end, :)   = grow(:, 2:end, :)   | ext(:, 1:end-1, :);
    grow(:, 1:end-1, :) = grow(:, 1:end-1, :) | ext(:, 2:end, :);
    grow(:, :, 2:end)   = grow(:, :, 2:end)   | ext(:, :, 1:end-1);
    grow(:, :, 1:end-1) = grow(:, :, 1:end-1) | ext(:, :, 2:end);
    grow = grow & ~mask;
    if isequal(grow, ext), break; end
    ext = grow;
end
u = double(~ext);
end

function P = load_cloud(base)
% 讀取 base.xyz / base.txt（每列 x y z ...）或 ASCII base.ply；找不到回傳 []
P = [];
if exist([base '.xyz'], 'file')
    A = load([base '.xyz']); P = A(:, 1:3);
elseif exist([base '.txt'], 'file')
    A = load([base '.txt']); P = A(:, 1:3);
elseif exist([base '.ply'], 'file')
    if exist('pcread') == 2
        pc = pcread([base '.ply']); P = double(pc.Location);
    else
        fid = fopen([base '.ply'], 'r');
        nv = 0;
        while true
            ln = fgetl(fid);
            if ~ischar(ln), break; end
            tk = sscanf(ln, 'element vertex %d');
            if ~isempty(tk), nv = tk; end
            if strncmp(ln, 'end_header', 10), break; end
        end
        P = zeros(nv, 3);
        for i = 1:nv
            v = sscanf(fgetl(fid), '%f');
            P(i, :) = v(1:3).';
        end
        fclose(fid);
    end
end
end

%% ======================================================================
%%                         以下為共用子程式
%% ======================================================================

function [x, K2, h] = setup_grid(n, dim)
% 週期計算域 [-pi,pi]^dim 的均勻網格與 Fourier 波數平方 |k|^2
% 週期為 2*pi，故波數為整數；G_tau 的 Fourier 乘子為 exp(-tau*|k|^2)
x = -pi + 2*pi*(0:n-1)/n;
h = 2*pi/n;
k = [0:n/2-1, -n/2:-1];
if dim == 2
    [k1, k2] = ndgrid(k, k);
    K2 = k1.^2 + k2.^2;
else
    [k1, k2, k3] = ndgrid(k, k, k);
    K2 = k1.^2 + k2.^2 + k3.^2;
end
end

function d = distance_to_cloud(P, x, dim, band)
% 計算網格點到點雲 P (N x dim) 的距離 d(x) = min_{y in C} |x - y|，
% 即 (20) 式 |grad d| = 1, d = 0 on C 的解。
% band = inf 時算精確距離；band 有限時只在點附近 band 範圍內精確，
% 其餘處截斷為 band（用於 3D 以節省時間；需 band 大於演算法實際用到的區域）。
n = numel(x);
h = x(2) - x(1);
N = size(P, 1);
if exist('knnsearch') == 2
    % 有 Statistics Toolbox 時用 KD-tree 精確計算
    if dim == 2
        [X1, X2] = ndgrid(x, x);
        Q = [X1(:), X2(:)];
    else
        [X1, X2, X3] = ndgrid(x, x, x);
        Q = [X1(:), X2(:), X3(:)];
    end
    [~, dd] = knnsearch(P, Q);
    d = reshape(dd, n*ones(1, max(dim, 2)));
    if isfinite(band), d = min(d, band); end
    return
end
if dim == 2 || ~isfinite(band)
    % 分塊暴力法
    if dim == 2
        [X1, X2] = ndgrid(x, x);
        Q = [X1(:), X2(:)];
    else
        [X1, X2, X3] = ndgrid(x, x, x);
        Q = [X1(:), X2(:), X3(:)];
    end
    M = size(Q, 1);
    d2 = inf(M, 1);
    chunk = max(1, floor(2e7 / N));
    for s = 1:chunk:M
        idx = s:min(M, s + chunk - 1);
        D = zeros(numel(idx), N);
        for c = 1:dim
            D = D + (Q(idx, c) - P(:, c).').^2;
        end
        d2(idx) = min(D, [], 2);
    end
    d = reshape(sqrt(d2), n*ones(1, max(dim, 2)));
    if isfinite(band), d = min(d, band); end
    return
end
% 3D 窄帶蓋印法：每個點只更新其周圍 band 內的網格
d2 = band^2 * ones(n, n, n);
m = ceil(band / h);
off = -m:m;
for i = 1:N
    c = round((P(i, :) + pi) / h) + 1;
    ix = c(1) + off; ix = ix(ix >= 1 & ix <= n);
    iy = c(2) + off; iy = iy(iy >= 1 & iy <= n);
    iz = c(3) + off; iz = iz(iz >= 1 & iz <= n);
    D = (x(ix).' - P(i, 1)).^2 + (x(iy) - P(i, 2)).^2 ...
        + reshape((x(iz) - P(i, 3)).^2, 1, 1, []);
    d2(ix, iy, iz) = min(d2(ix, iy, iz), D);
end
d = sqrt(d2);
end

function f = heat_conv(g, G)
% G_tau * g，以 FFT 計算（週期邊界）
f = real(ifftn(fftn(g) .* G));
end

function E = energy6(psi, u, G, tau, h, dim)
% (6) 式: E^tau(u) = sqrt(pi/tau) * int psi*u * G_tau*(psi*(1-u)) dx, psi = |d|^(p/2)
E = sqrt(pi / tau) * sum(sum(sum(psi .* u .* heat_conv(psi .* (1 - u), G)))) * h^dim;
end

function [u, phi, nit, E] = algorithm2(psi, u, tau, K2, maxit, h, dim, needE)
% Algorithm 2:
%   1. phi^k = G_tau * ( |d|^(p/2) (1 - 2u^k) )
%   2. u^{k+1} = 1 if phi^k <= 0, 0 otherwise
% 收斂準則 (Remark 1): 所有網格點上 u 都不再改變。
G = exp(-tau * K2);
E = [];
for nit = 1:maxit
    phi = heat_conv(psi .* (1 - 2*u), G);
    if needE
        E(end+1) = energy6(psi, u, G, tau, h, dim); %#ok<AGROW>
    end
    unew = double(phi <= 0);
    if isequal(unew, u)
        break
    end
    u = unew;
end
end

function [u, phi, info] = algorithm3_2(psi, u, taus, K2, maxit, h, dim, needE)
% Algorithm 3_2: 以遞減的 tau_1 > tau_2 > ... 依序執行 Algorithm 2，
% 前一個 tau 的穩態解作為下一個 tau 的初始猜測；
% 若相鄰兩個 tau 得到完全相同的解則停止。
info.E = [];
info.switchIt = [];
info.u = {};
info.phi = {};
info.tau = [];
info.iters = 0;
uprev = [];
for s = 1:numel(taus)
    [u, phi, nit, E] = algorithm2(psi, u, taus(s), K2, maxit, h, dim, needE);
    info.iters = info.iters + nit;
    info.E = [info.E, E];
    info.switchIt(end+1) = numel(info.E);
    info.u{end+1} = u;
    info.phi{end+1} = phi;
    info.tau(end+1) = taus(s);
    if ~isempty(uprev) && isequal(u, uprev)
        break
    end
    uprev = u;
end
end

function plot2d(x, P, u0, phi, ttl)
% 藍色虛線: 初始猜測；黑色: 重建曲線 (Step 1 中 phi 的零等值線)；紅點: 點雲
hold on
if ~isempty(u0)
    contour(x, x, u0.', [0.5 0.5], 'b--');
end
plot(P(:, 1), P(:, 2), 'r.', 'MarkerSize', 4);
if ~isempty(phi)
    contour(x, x, phi.', [0 0], 'k', 'LineWidth', 1.2);
end
axis equal
axis([-2.6 2.6 -2.6 2.6])
set(gca, 'XTick', [], 'YTick', [])
box on
title(ttl)
hold off
end

function plot3d(x, P, phi, ttl, vw)
% 重建曲面 (phi 的零等值面) 與點雲
hold on
if any(phi(:) <= 0)
    fv = isosurface(x, x, x, permute(phi, [2 1 3]), 0);
    patch(fv, 'FaceColor', [0.45 0.85 0.85], 'EdgeColor', 'none');
else
    ttl = [ttl ' (surface vanished)'];   % 曲面已消失（tau 太大，見 Remark 3）
end
plot3(P(:, 1), P(:, 2), P(:, 3), 'r.', 'MarkerSize', 2);
axis equal
axis([-pi pi -pi pi -pi pi] * 0.75)
axis off
view(vw)
try
    camlight; lighting gouraud;
catch
end
title(ttl)
hold off
end

function suptitle_safe(str)
try
    sgtitle(str);
catch
    annotation('textbox', [0 0.95 1 0.05], 'String', str, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end
end
