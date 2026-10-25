function ex_5_2_1()
% 5.2.1 Comparisons Among Algorithms（只用 Algorithm 2 與 Algorithm 3_2）
%   Fig. 5 第二列: 2D 五瓣點雲, Algorithm 2, tau = 0.02, 0.015, 0.01, 0.005, 0.0025
%   Fig. 6 右    : 2D 五瓣點雲, Algorithm 3_2, tau_i = 0.02, 0.01, 0.005, 0.0025, 0.00125
%   Fig. 7 第二列: 3D torus,  Algorithm 2, tau = 0.02, 0.018, 0.016, 0.01, 0.005
%   Fig. 8 右    : 3D torus,  Algorithm 3_2, tau_i = 0.02, 0.01, 0.005, 0.0025
% 計算域 [-pi,pi]^n, p = 2；CPU 時間不含 d(x) 的計算。
close all
p = 2;
maxit = 2000;
rng(1);

%% ---------------- 2D 五瓣點雲 ----------------
n = 128;
[x, K2, h] = setup_grid(n, 2);
N = 200;
theta = 2*pi*(0:N-1)'/N;
r = 1 + 0.5*cos(5*(theta - pi/2));
P2 = [r.*cos(theta), r.*sin(theta)];
psi = distance_to_cloud(P2, x, 2, inf).^(p/2);
[X, Y] = ndgrid(x, x);
u0 = double(X.^2 + Y.^2 < 1.8^2);       % 初始猜測: 包住整個點雲的圓

taus = [0.02 0.015 0.01 0.005 0.0025];
figure('Name', 'Fig.5 (Algorithm 2), 2D', 'Position', [50 400 1300 300]);
for j = 1:numel(taus)
    t0 = tic;
    [~, phi, nit] = algorithm2(psi, u0, taus(j), K2, maxit, h, 2, false);
    t = toc(t0);
    subplot(1, numel(taus), j);
    plot2d(x, P2, u0, phi, sprintf('\\tau = %g\n%.3fs, %d iters', taus(j), t, nit));
end
suptitle_safe('Algorithm 2, five-fold point cloud');

taus3 = [0.02 0.01 0.005 0.0025 0.00125];
t0 = tic;
[~, phi, info] = algorithm3_2(psi, u0, taus3, K2, maxit, h, 2, false);
t = toc(t0);
figure('Name', 'Fig.6 right (Algorithm 3_2), 2D', 'Position', [50 50 450 450]);
plot2d(x, P2, u0, phi, sprintf('Algorithm 3\\_2: %.3fs (stop at \\tau = %g)', t, info.tau(end)));

%% ---------------- 3D torus ----------------
n = 128;
[x, K2, h] = setup_grid(n, 3);
N = 2000;
uu = 2*pi*rand(N, 1);
vv = 2*pi*rand(N, 1);
P3 = [(1 + 0.5*cos(uu)).*cos(vv), (1 + 0.5*cos(uu)).*sin(vv), 0.5*sin(uu)];
psi = distance_to_cloud(P3, x, 3, 1.2).^(p/2);
[X, Y, Z] = ndgrid(x, x, x);
u0 = double(abs(X) < 1.6 & abs(Y) < 1.6 & abs(Z) < 0.6);   % 初始猜測: 長方體
clear X Y Z

taus = [0.02 0.018 0.016 0.01 0.005];
figure('Name', 'Fig.7 (Algorithm 2), 3D torus', 'Position', [50 400 1500 320]);
for j = 1:numel(taus)
    t0 = tic;
    [~, phi, nit] = algorithm2(psi, u0, taus(j), K2, maxit, h, 3, false);
    t = toc(t0);
    subplot(1, numel(taus), j);
    plot3d(x, P3, phi, sprintf('\\tau = %g\n%.2fs, %d iters', taus(j), t, nit), [30 40]);
    drawnow
end
suptitle_safe('Algorithm 2, torus point cloud');

taus3 = [0.02 0.01 0.005 0.0025];
t0 = tic;
[~, phi, info] = algorithm3_2(psi, u0, taus3, K2, maxit, h, 3, false);
t = toc(t0);
figure('Name', 'Fig.8 right (Algorithm 3_2), 3D torus', 'Position', [600 50 500 450]);
plot3d(x, P3, phi, sprintf('Algorithm 3\\_2: %.2fs (stop at \\tau = %g)', t, info.tau(end)), [30 40]);
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
