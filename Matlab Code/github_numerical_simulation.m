% Guide

% Part I: calculate optimized TRI for each sample size
% Part II: repeat data generation and fitting to get summary table of P_e and ISE
% Part III: plot estimated zero region and MCR
% Part IV: Calculate Probability Coverage
% Part V: Performance of different TRI (Optimized, Denser, Sparser, Uniform). We pick sample size n=2000 as example

%% Part I: calculate optimized TRI for each sample size

% Define Varying Coefficient Functions
f = @(x,y) sqrt((x-0.5).^2 + (y-0.5).^2); fd = @(p) ddiff(drectangle(p, 0, 2, 0, 2), dcircle(p, 1, 1, 0.5)); fh=@(p) 1+4*dcircle(p,1,1,0.5);
f1 = @(x,y) 1.* sin(2.*pi./(sqrt(2)-0.5) .* (sqrt((x-1).^2+(y-1).^2)-0.5))+1; f2 = @(x,y) 2.* (exp((y-1).* (y>=1)) - 1) ; f3 = @(x,y) 0 .* y;

% Part 0: Generate optimized TRI for all sample sizes

% generate TRI for each h value. h is a hyperparameter used in the function
% distmesh2d()

h_opt = 0.13:0.01:0.21;

%% Attention:
% The code chunk below is used to generate optimized "h" value for each sample size, in which "h" is a
% parameter used to generate triangulation. We find calling the function "distmesh2d()" with
% specific "h" values might lead to infinite running on some PC. So, we comment out this process to
% make sure you can directly run this entire script directly. The code below could be run on our PC

% for j = 1:length(h_opt)
%     rng(1000); % Set seed to make sure the generated triangulations are identical
%     h_now = h_opt(j); 
%     disp(h_now)
%     [p,TRI] = distmesh2d(fd, fh, h_now, [0,0;2,2], [0,0;0,2;2,0;2,2]);
%     % You need to output some Triangulations manually for later repeated use
%     if abs(h_now - 0.17) < 0.001
%         p17 = p; TRI17 = TRI;
%     end   
%     if abs(h_now - 0.18) < 0.001
%         p18 = p; TRI18 = TRI;
%     end 
%     if abs(h_now - 0.19) < 0.001
%         p19 = p; TRI19 = TRI;
%     end 
% end

% In total, we have 4 different sample sizes
n_choice = [500, 1000, 2000, 5000]; 
h_choice = 0.14:0.01:0.22; 
m = 3;
bic_records = zeros(length(n_choice), length(h_choice));

%% Attention:
% The below comment section is used to get optimized TRI for each sample size, which can be run in R2018a. 
% But, if you use another version of Matlab, it might cause some issues when you run this code line: 
% -----------------------------------------------------------------------
% `[p,TRI] = distmesh2d(fd, fh, 0.17, [0,0;2,2], [0,0;0,2;2,0;2,2]);`
% -----------------------------------------------------------------------
% For this case when h=0.17, we manually provide triangulation information and stored as "p17" and "TRI17"

% sample size: 500
% rng(100);
% for i=1
%     n = n_choice(i);
%     temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); % Generate points first 
%     while(temp_no <= n)
%         temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); 
%         % Use beta distribution to generate locations of data points
%         temp_theta = 2*pi*rand(1);
%         c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
%         if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
%             X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
%         end
%     end
%     % generate response values via defined varying coefficient functions
%     beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
%     X_1 = randn(n,1); X_2 = randn(n,1); X_3 = randn(n,1); epsilon=randn(n, 1);
%     Z = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3+epsilon;
%     ori_Z = Z;
%     
%     for j=1:length(h_choice)
%         % generate Triangulations
%         disp(['The current n value:', num2str(n), ' the current h value: ', num2str(h_choice(j))])
%         if abs(h_choice(j) - 0.17) < 0.0001
%             p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
%         else
%             rng(1000);
%             [p,TRI] = distmesh2d(fd, fh, h_choice(j),[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
%         end 
%               
%         [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
%            vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
%         nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;
%         
%         % Call "CZ_SPL_est" to generate design matrix of Bernstein basis
%         % polynomials
%         [B, valid_id] = CZ_SPL_est(X, Y, vx, vy, TRI,v1,v2,v3,nt,nc,nv,d);
%         
%         % To get the final design matrix, we need to multiply covariant
%         % values to Bernstein basis polynomials
%         mat_Z = zeros(n, m*nc);
%         for k = 1:n
%             temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
%             mat_Z(k,:) = [temp1, temp2, temp3];
%         end
%         full_mat_Z = mat_Z;
%         mat_Z = full_mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
%         
%         b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z;
%         
%         % Calculate BIC value
%         bic_records(i, j) = log(mean((Z - mat_Z * b_hat).^2)) + log(length(valid_id)) * sum(b_hat ~=0) / length(valid_id);
%     end
%     disp(['The experiment of n:', num2str(n), ' is finished'])
%     [~, argmin] = min(bic_records(i, :), [], 2); h_choice(argmin)
%     disp(['The best h value is ', num2str(h_choice(argmin))])
% end
% [~, argmin] = min(bic_records(1, :), [], 2); h_choice(argmin) % best h choice is 0.21
% 
% % sample size: 1000. Similar to the case sample size=500
% rng(200);
% for i=2
%     n = n_choice(i);
%     temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); % Generate points first 
%     while(temp_no <= n)
%         temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); 
%         temp_theta = 2*pi*rand(1);
%         c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
%         if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
%             X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
%         end
%     end
%     beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
%     X_1 = randn(n,1); X_2 = randn(n,1); X_3 = randn(n,1); epsilon=randn(n, 1);
%     Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3 + epsilon;
%     ori_Z = Z;
%     
%     for j=1:length(h_choice)
%         disp(['The current n value:', num2str(n), ' the current h value: ', num2str(h_choice(j))])
%         if h_choice(j) == 0.17
%             p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
%         else
%             rng(1000);
%             [p,TRI] = distmesh2d(fd, fh, h_choice(j),[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
%         end 
%               
%         [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
%            vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
%         nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;
%         [B, valid_id] = CZ_SPL_est(X, Y, vx, vy, TRI,v1,v2,v3,nt,nc,nv,d);
%         
%         mat_Z = zeros(n, m*nc);
%         for k = 1:n
%             temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
%             mat_Z(k,:) = [temp1, temp2, temp3];
%         end
%         full_mat_Z = mat_Z;
%         mat_Z = full_mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
%         
%         b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z;
%         bic_records(i, j) = log(mean((Z - mat_Z * b_hat).^2)) + log(length(valid_id)) * sum(b_hat ~=0) / length(valid_id);
%     end
%     disp(['The experiment of n:', num2str(n), ' is finished'])
%     [~, argmin] = min(bic_records(i, :), [], 2); h_choice(argmin)
%     disp(['The best h value is ', num2str(h_choice(argmin))])
% end
% [~, argmin] = min(bic_records(1:2, :), [], 2); h_choice(argmin) % best h choice is 0.21
% 
% % sample size: 2000. Similar to the case sample size=500
% rng(200);
% for i=3
%     n = n_choice(i);
%     temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); % Generate points first 
%     while(temp_no <= n)
%         temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); 
%         temp_theta = 2*pi*rand(1);
%         c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
%         if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
%             X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
%         end
%     end
%     beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
%     X_1 = randn(n,1); X_2 = randn(n,1); X_3 = randn(n,1); epsilon=randn(n, 1);
%     Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3 + epsilon;
%     ori_Z = Z;
%     
%     for j=1:length(h_choice)
%         disp(['The current n value:', num2str(n), ' the current h value: ', num2str(h_choice(j))])
%         if h_choice(j) == 0.17
%             p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
%         else
%             rng(1000);
%             [p,TRI] = distmesh2d(fd, fh, h_choice(j),[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
%         end 
%               
%         [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
%            vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
%         nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;
%         [B, valid_id] = CZ_SPL_est(X, Y, vx, vy, TRI,v1,v2,v3,nt,nc,nv,d);
%         
%         mat_Z = zeros(n, m*nc);
%         for k = 1:n
%             temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
%             mat_Z(k,:) = [temp1, temp2, temp3];
%         end
%         full_mat_Z = mat_Z;
%         mat_Z = full_mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
%         
%         b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z;
%         bic_records(i, j) = log(mean((Z - mat_Z * b_hat).^2)) + log(length(valid_id)) * sum(b_hat ~=0) / length(valid_id);
%     end
%     disp(['The experiment of n:', num2str(n), ' is finished'])
%     [~, argmin] = min(bic_records(i, :), [], 2); h_choice(argmin)
%     disp(['The best h value is ', num2str(h_choice(argmin))])
% end
% [~, argmin] = min(bic_records(1:3, :), [], 2); h_choice(argmin) % best h choice is 0.19
% 
% % sample size: 5000. Similar to the case sample size=500
% rng(100);
% for i=4
%     n = n_choice(i);
%     temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); % Generate points first 
%     while(temp_no <= n)
%         temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); 
%         temp_theta = 2*pi*rand(1);
%         c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
%         if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
%             X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
%         end
%     end
%     beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
%     X_1 = randn(n,1); X_2 = randn(n,1); X_3 = randn(n,1); epsilon=randn(n, 1);
%     Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3 + epsilon;
%     ori_Z = Z;
%     
%     for j=1:length(h_choice)
%         disp(['The current n value:', num2str(n), ' the current h value: ', num2str(h_choice(j))])
%         if h_choice(j) == 0.17
%             p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
%         else
%             rng(1000);
%             [p,TRI] = distmesh2d(fd, fh, h_choice(j),[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
%         end 
%               
%         [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
%            vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
%         nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;
%         [B, valid_id] = CZ_SPL_est(X, Y, vx, vy, TRI,v1,v2,v3,nt,nc,nv,d);
%         
%         mat_Z = zeros(n, m*nc);
%         for k = 1:n
%             temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
%             mat_Z(k,:) = [temp1, temp2, temp3];
%         end
%         full_mat_Z = mat_Z;
%         mat_Z = full_mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
%         
%         b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z;
%         bic_records(i, j) = log(mean((Z - mat_Z * b_hat).^2)) + log(length(valid_id)) * sum(b_hat ~=0) / length(valid_id);
%     end
%     disp(['The experiment of n:', num2str(n), ' is finished'])
%     [~, argmin] = min(bic_records(i, :), [], 2); h_choice(argmin)
%     disp(['The best h value is ', num2str(h_choice(argmin))])
% end
% [~, argmin] = min(bic_records(1:4, :), [], 2); h_choice(argmin) % best h choice is 0.17

% Now we get the best h choices are [0.21, 0.21, 0.19, 0.17] for sample sizes [500, 1000, 2000, 5000] respectively
% In case of some PC might not be able to generate the TRI when h=0.17, we manually copy and paste
% it below

p17 = [
    -0.0000   0.9990;0         0;...
         0    2.0000;0.0000    0.5916;0.0000    1.4104;0.4366    1.1947;0.4447    0.8102;...
    0.5000    1.0019;0.5119    0.4630;0.5185    1.5535;0.5988    1.2984;0.6093    0.6880;...
    0.6958    0.0000;0.7363    2.0000;0.7432    1.4290;0.7709    0.5556;0.8834    0.3485;...
    0.8846    1.4865;0.9876    0.5002;1.0386    1.6114;1.1737    0.4504;1.1915    1.4619;...
    1.2594    0.0000;1.2883    2.0000;1.2957    0.5968;1.3353    1.3709;1.4202    0.7290;...
    1.4469    1.2242;1.4963    0.9395;1.5199    1.5250;1.5275    0.4341;1.6200    1.1016;...
    1.6202    0.7766;2.0000    0.7297;2.0000    1.2924;2.0000         0;2.0000    2.0000;...
    ];

TRI17 = [
    36    34    31; 31    23    36; 34    35    32; 33    31    34; 33    32    29; 34    32    33;...
    23    31    21; 24    14    20; 20    14    18; 10    14     3;  3     5    10; 10    18    14;...
    15    18    10; 31    33    27; 27    33    29;  2    13     9; 30    32    35; 30    35    37;...
    37    24    30; 25    21    31; 31    27    25;  2     9     4;  6     5     1;  6    10     5;...
    32    30    28; 30    26    28; 29    32    28; 20    18    22; 22    26    30; 22    24    20;...
    22    30    24; 17     9    13; 17    13    23; 23    21    17; 21    25    19; 19    17    21;...
     1     4     7;  7     4     9;  9    12     7;  8     7    12;  8     6     1;  1     7     8;...
    17    19    16; 16    12     9;  9    17    16;  6     8    11; 15    10    11; 10     6    11;... 
];

%% Part II: repeat data generation and fitting to get summary table of P_e and ISE
kLoopTime=100;
n_choice = [500, 1000, 2000, 5000]; best_h_values = [0.21, 0.21, 0.19, 0.17]; % These 4 are best h values (i.e., best triangulation)
record_table = zeros(kLoopTime, length(n_choice), 9);  diag_lamb_vec = zeros(kLoopTime, 3); 

% generate grid points
% Generate grids. They fall in [0, 2]x[0, 2] but not falling into the central circle
grid_len = 200;
grid_s = linspace(0, 2, grid_len + 2); grid_t = linspace(0, 2, grid_len + 2); grid_s = grid_s(2:(grid_len+1)); grid_t = grid_t(2:(grid_len+1));
[grid_S, grid_T] = meshgrid(grid_s, grid_t); grid_S = reshape(grid_S, [grid_len^2, 1]); grid_T = reshape(grid_T, [grid_len^2, 1]);
grid_idx = (grid_S - 1).^2 + (grid_T - 1).^2 > 0.25; 

% Only valid grid points are kept
grid_S = grid_S(grid_idx); grid_T = grid_T(grid_idx); 
grid_f1 = f1(grid_S, grid_T); grid_f2 = f2(grid_S, grid_T); grid_f3 = f3(grid_S, grid_T); 
grid_f2_zeroidx = find(grid_f2 == 0); grid_f3_zeroidx = find(grid_f3 == 0);

% Repeat data generation and fitting to get summary results of Mean and Standard Error of simulations
disp("Start to generate summary table of P_e and ISE")
tic; trial_seed=92 ; % 
% seed is set in the nested for loop
for i=1:length(n_choice)
    n = n_choice(i); rng(1000);h_now = best_h_values(i);  
    if (h_now == 0.17) 
        p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
    else
        [p,TRI] = distmesh2d(fd,fh,h_now,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
    end
    [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
        vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
    nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt; [grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
    
    rng(trial_seed);
    for j=1:kLoopTime
        temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); % Generate points first 
        while(temp_no <= n)
            temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); % Use beta distribution to generate observations
            temp_theta = 2*pi*rand(1);
            c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
            if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
                X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
            end
        end
        
        beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); X_1 = randn(n,1);X_2 = randn(n,1);X_3 = randn(n,1);epsilon=randn(n, 1);
        Z = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3+epsilon;
        Z_no_epi = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3;Z_1 = X_1.*beta_1; Z_2 = X_2.*beta_2; Z_3 = X_3.*beta_3;
        [B, valid_id] = CZ_SPL_est(X,Y,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

        mat_Z = zeros(n, m*nc);            
        for k = 1:n
            temp1 = (B(k, :).* X_1(k,1));temp2 = (B(k, :).* X_2(k,1));temp3 = (B(k, :).* X_3(k,1));
            mat_Z(k,:) = [temp1, temp2, temp3];
        end
        full_mat_Z = mat_Z; full_Z = Z;
        mat_Z = mat_Z(valid_id, :); Z = Z(valid_id, :);
        
        b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z; % add a small diagonal mat to avoid singularity
        ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat(1:nc)).^2); 
        ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((nc+1):(2*nc))).^2); 
        ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((2*nc+1):(3*nc))).^2);
        record_table(j, i, 1) = ISE_1; record_table(j, i, 2) = ISE_2; record_table(j, i, 3) = ISE_3; 
        %ISE_UNPEN part is done.
        
        nlam = 40; a = 3.7;threshold = 10 ^ (-3); 
        lam_vec = linspace(0.01, 0.4, nlam);
        bic = zeros(nlam, 1); converged_or_not = zeros(nlam, 1);
        for q = 1:nlam
            [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id), 2, 1);
            converged_or_not(q) = dist_logical;
            bic(q) = log(mean((Z - mat_Z * p_b_hat).^2)) + log(length(valid_id)) * sum(p_b_hat ~=0) / length(valid_id);
        end
        
        [temp_min, temp_index] = min(bic); diag_lamb_vec(j,i) = lam_vec(temp_index);
        [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id), 2, 1);
        
        ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat(1:nc)).^2); 
        ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((nc+1):(2*nc))).^2); 
        ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((2*nc+1):(3*nc))).^2);       
        record_table(j, i, 4) = ISE_1; record_table(j, i, 5) = ISE_2; record_table(j, i, 6) = ISE_3;
        %ISE_SCAD part is done.
        
        record_table(j, i, 7) = sum(grid_B(grid_valid_id,:) * p_b_hat(1:nc) == 0) / length(grid_valid_id);
        grid_f2_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+nc:2*nc) == 0);
        record_table(j, i, 8) = (length(intersect(grid_f2_zeroidx, grid_valid_id)) + length(grid_f2_zeroidx_pred) - 2*length(intersect(grid_f2_zeroidx, grid_f2_zeroidx_pred))) / length(grid_valid_id);       
        grid_f3_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+2*nc:3*nc) == 0);
        record_table(j, i, 9) = (length(grid_f3_zeroidx) + length(grid_f3_zeroidx_pred) - 2*length(intersect(grid_f3_zeroidx, grid_f3_zeroidx_pred))) / length(grid_valid_id);
        % P_e part is done
    end
end
toc;
% running time: 200s

% Display the mean and std
str = zeros(length(n_choice), 9); str = string(str);
for i = 1:length(n_choice)
    for j = 1:9
         str(i, j) = sprintf('%.4f (%.4f)', round(mean(record_table(:, i, j)),4), round(std(record_table(:, i, j)), 4)) ;
    end
end
summary_T = [string('n') string('Beta') string('P_e') string('ISE_SCAD') string('ISE_UNPEN'); ...1
    '500' 'F1' str(1,7) str(1,4) str(1,1); '1000' 'F1' str(2, 7) str(2,4) str(2,1);...
    '2000' 'F1' str(3,7) str(3,4) str(3,1); '5000' 'F1' str(4,7) str(4,4) str(4,1);...
    '500' 'F2' str(1,8) str(1,5) str(1,2); '1000' 'F2' str(2, 8) str(2,5) str(2,2);...
    '2000' 'F2' str(3,8) str(3,5) str(3,2); '5000' 'F2' str(4,8) str(4,5) str(4,2);...
    '500' 'F3' str(1,9) str(1,6) str(1,3); '1000' 'F3' str(2, 9) str(2,6) str(2,3);...
    '2000' 'F3' str(3,9) str(3,6) str(3,3); '5000' 'F3' str(4,9) str(4,6) str(4,3);];
summary_T
% rng(92)
%     "n"       "Beta"    "P_e"                "ISE_SCAD"           "ISE_UNPEN"      
%     "500"     "F1"      "0.0000 (0.0000)"    "0.3095 (0.1129)"    "0.3153 (0.1123)"
%     "1000"    "F1"      "0.0000 (0.0000)"    "0.2097 (0.0423)"    "0.2149 (0.0463)"
%     "2000"    "F1"      "0.0000 (0.0000)"    "0.1799 (0.0315)"    "0.1818 (0.0320)"
%     "5000"    "F1"      "0.0000 (0.0000)"    "0.1322 (0.0223)"    "0.1331 (0.0230)"
%     "500"     "F2"      "0.1290 (0.0807)"    "0.1146 (0.0908)"    "0.1576 (0.0838)"
%     "1000"    "F2"      "0.0875 (0.0543)"    "0.0492 (0.0289)"    "0.0701 (0.0382)"
%     "2000"    "F2"      "0.0695 (0.0422)"    "0.0259 (0.0129)"    "0.0297 (0.0114)"
%     "5000"    "F2"      "0.0432 (0.0439)"    "0.0118 (0.0057)"    "0.0152 (0.0055)"
%     "500"     "F3"      "0.0686 (0.0901)"    "0.0467 (0.0798)"    "0.1526 (0.0892)"
%     "1000"    "F3"      "0.0501 (0.0780)"    "0.0167 (0.0330)"    "0.0651 (0.0382)"
%     "2000"    "F3"      "0.0435 (0.0713)"    "0.0061 (0.0105)"    "0.0324 (0.0141)"
%     "5000"    "F3"      "0.0390 (0.0587)"    "0.0024 (0.0040)"    "0.0143 (0.0053)"

%% Part III: plot estimated zero region and MCR
% Prepare for all plots and MCR diagrams %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate one-time the width over total number of triangles (WOT)
n_choice = [500, 1000, 2000, 5000]; best_h_choices = [0.21, 0.21, 0.19, 0.17];
all_b_hat = cell(length(n_choice), 1); all_p_b_hat = cell(length(n_choice), 1); 
all_grid_B = cell(length(n_choice), 1); all_grid_valid_id = cell(length(n_choice), 1); 
all_TRI = cell(length(n_choice), 1); all_vx = cell(length(n_choice), 1); all_vy = cell(length(n_choice), 1);
all_LBR = cell(length(n_choice), 3); all_UBR = cell(length(n_choice), 3);
all_wot = zeros(length(n_choice), 3);
tic;
for i=1:length(n_choice)  % This for loop is for fitting
    n = n_choice(i); h_now = best_h_choices(i);
    if (h_now == 0.17) 
        p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
    else
        rng(1000);
        [p,TRI] = distmesh2d(fd, fh, h_now,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
    end

    all_TRI{i,1} = TRI; all_vx{i,1} = vx; all_vy{i,1} = vy;
    [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
    nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt; [grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
    all_grid_B{i, 1} = grid_B; all_grid_valid_id{i, 1} = grid_valid_id;
    
    rng(5); % set seed before generating data points
    temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); % Generate points here
    while(temp_no <= n)
        temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); % Use beta distribution to generate observations
        temp_theta = 2*pi*rand(1);
        c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
        if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
            X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
        end
    end
    % Generate covariates and response
    beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); X_1 = randn(n,1); X_2 = randn(n,1); X_3 = randn(n,1); epsilon=randn(n, 1);
    Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3 + epsilon;
    Z_no_epi = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3;Z_1 = X_1.*beta_1; Z_2 = X_2.*beta_2; Z_3 = X_3.*beta_3;
    [B, valid_id] = CZ_SPL_est(X,Y,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
    
    mat_Z = zeros(n, m*nc);            
    for k = 1:n
        temp1 = (B(k, :).* X_1(k,1));temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
        mat_Z(k,:) = [temp1, temp2, temp3];
    end
    full_mat_Z = mat_Z; full_Z = Z;
    mat_Z = mat_Z(valid_id, :); Z = Z(valid_id, :);
    b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z; % UNPEN estimator
    all_b_hat{i,1} = b_hat;
    
    nlam = 40; a = 3.7;threshold = 10 ^ (-3); % SCAD estimator
    lam_vec = linspace(0.01, 0.4, nlam);
    bic = zeros(nlam, 1); converged_or_not = zeros(nlam, 1);
    for q = 1:nlam
        [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id), 2, 1);
        converged_or_not(q) = dist_logical;
        bic(q) = log(mean((Z - mat_Z * p_b_hat).^2)) + log(length(valid_id)) * sum(p_b_hat ~=0) / length(valid_id);
    end
    [temp_min, temp_index] = min(bic); diag_lamb_vec(j,i) = lam_vec(temp_index);
    [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id), 2, 1);
    all_p_b_hat{i,1} = p_b_hat;
    
    records =  CZ_bootstrap(2, TRI, mat_Z, Z, n, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, 100, 1, 1, 3.7, 0);
    MCB_records = cell(size(records,1)+1, 2*m); CR_records = zeros(size(records,2)/m+1, m);
    for j = 1:m
       TRI_no = 1+(j-1)*nt:j*nt; [out, pi_idx] = sort(sum(records(:, TRI_no), 1), 'descend');
       for w = 0:1:nt
          cr = 0; LBM = []; UBM = [];
          for k = 0:1:nt-w
             temp_LBM = pi_idx(1:k); temp_UBM = sort(pi_idx(1:k+w));
             temp_cr = CZ_CoverageRate(records(:, TRI_no), temp_LBM, temp_UBM);
             if temp_cr > cr
                cr = temp_cr; LBM = temp_LBM; UBM = temp_UBM; 
             end
          end
          MCB_records{w+1, 2*j-1} = sort(LBM); MCB_records{w+1, 2*j} = sort(UBM); CR_records(w+1, j) = cr;
       end
    end
    all_LBR{i, 1} = sort(MCB_records{find(CR_records(:,1) > 0.95,1), 2*1-1}) ; all_UBR{i, 1} = sort(MCB_records{find(CR_records(:,1) > 0.95,1), 2*1});
    all_LBR{i, 2} = sort(MCB_records{find(CR_records(:,2) > 0.95,1), 2*2-1}) ; all_UBR{i, 2} = sort(MCB_records{find(CR_records(:,2) > 0.95,1), 2*2});
    all_LBR{i, 3} = sort(MCB_records{find(CR_records(:,3) > 0.95,1), 2*3-1}) ; all_UBR{i, 3} = sort(MCB_records{find(CR_records(:,3) > 0.95,1), 2*3});
    all_wot(i, 1:3) = [find(CR_records(:,1) > 0.95, 1)-1 find(CR_records(:,2) > 0.95, 1)-1 find(CR_records(:,3) > 0.95, 1)-1];
    all_wot(i, 1:3) = all_wot(i, 1:3) ./ nt;
end
toc; 
all_wot

% rng(5), it took ~150 seconds to finish
%          0    0.5161    0.7097
%          0    0.3226    0.4194
%          0    0.0938    0.3438
%          0    0.0833    0.1250

% Estimated Zero Regions
subplot(1+length(n_choice), m, 1); hold on; line([0 2 2 0 0], [0 0 2 2 0], 'color','black');
th = 0:pi/50:2*pi; x_circle = 0.5 * cos(th) + 1; y_circle = 0.5 * sin(th) + 1; circles = plot(x_circle, y_circle);
fill(x_circle, y_circle, 'white', 'LineStyle', 'none');fill(x_circle, y_circle, 'black', 'LineStyle', 'none');
title('$$\beta_1$$ zero region','interpreter','latex'); hold off;
subplot(1+length(n_choice), m, 2); hold on; line([0 2 2 0 0], [0 0 2 2 0], 'color','black')
fill([0 2 2 0], [0 0 1 1], 'red'); fill(x_circle, y_circle, 'white', 'LineStyle', 'none'); fill(x_circle, y_circle, 'black', 'LineStyle', 'none');
title('$$\beta_2$$ zero region','interpreter','latex'); hold off;
subplot(1+length(n_choice), m, 3); hold on; line([0 2 2 0 0], [0 0 2 2 0], 'color','black'); fill([0 2 2 0], [0 0 2 2], 'red')
fill(x_circle, y_circle, 'white', 'LineStyle', 'none'); fill(x_circle, y_circle, 'black', 'LineStyle', 'none');
title('$$\beta_3$$ zero region','interpreter','latex'); hold off;
for i=1:length(n_choice)   
    temp_TRI = all_TRI{i, 1}; temp_vx = all_vx{i, 1}; temp_vy = all_vy{i, 1};
    for j=1:m
        order = i*m+j; subplot(1+length(n_choice), m, order); hold on; line([0 2 2 0 0], [0 0 2 2 0], 'color','black');      
        triplot(temp_TRI, temp_vx, temp_vy); xlim([0 2]); ylim([0 2]);
        temp_p_b_hat = all_p_b_hat{i, 1}; nc = length(temp_p_b_hat)/3; temp_p_b_hat = temp_p_b_hat(1+(j-1)*nc:j*nc); % j-th VCF
        for l=1:size(temp_TRI, 1) % l=1:N_t
            if temp_p_b_hat(temp_TRI(l, 1)) == 0 && temp_p_b_hat(temp_TRI(l, 2)) == 0 && temp_p_b_hat(temp_TRI(l, 3)) == 0
                fill(temp_vx([temp_TRI(l, 1) temp_TRI(l, 2) temp_TRI(l, 3)]), temp_vy([temp_TRI(l, 1) temp_TRI(l, 2) temp_TRI(l, 3)]), 'red');
            end
        end
        if j== 1
            ylabel(sprintf('n = %d', n_choice(i)));
        end 
        if j==2
            line([0 2], [1 1], 'LineStyle', '--');
        end
        hold off;
    end
end
% This generates: Fig - Estimated Zero Regions of Continuous Simulation

% generate MCR plot
for i=1:length(n_choice)
    temp_TRI = all_TRI{i, 1}; temp_vx = all_vx{i, 1}; temp_vy = all_vy{i, 1};
    for j=1:m
        order = (i-1)*m+j; subplot(length(n_choice), m, order); hold on;
        line([0 2 2 0 0], [0 0 2 2 0]);
        triplot(temp_TRI, temp_vx, temp_vy); xlim([0,2]); ylim([0,2]); 
        for l=all_UBR{i, j}
           fill(temp_vx([temp_TRI(l, 1) temp_TRI(l, 2) temp_TRI(l, 3)]), temp_vy([temp_TRI(l, 1) temp_TRI(l, 2) temp_TRI(l, 3)]), 'red') ;         
        end
        for l=all_LBR{i, j}
           fill(temp_vx([temp_TRI(l, 1) temp_TRI(l, 2) temp_TRI(l, 3)]), temp_vy([temp_TRI(l, 1) temp_TRI(l, 2) temp_TRI(l, 3)]), 'yellow') ;         
        end
        if i==1 && j==1
            title('$$\beta_1$$' ,'interpreter','latex');
        end
        if i==1 && j==2
            title('$$\beta_2$$' ,'interpreter','latex');
        end
        if i==1 && j==3
            title('$$\beta_3$$' ,'interpreter','latex');
        end
        if j== 1
            ylabel(sprintf('n = %d', n_choice(i)));
        end
        hold off;
    end
    
end
% This generates: Fig - MCR of Continuous Simulation

%% Part IV: Calculate Coverage Probability
disp("Start to generate coverage probability for each sample size")
% Calculate Coverage Prob. for each sample size
%% Use old code - sample size 500 (confirmed)
rng(1000);
[p,TRI] = distmesh2d(fd,fh,0.21,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2); % optimized h value is 0.21 (3rd argument in distmesh2d()
[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
nv = length(vx); d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt; 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S, grid_T, vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

% Plot triangulation and we manually enter "m_star_low" and "m_star_high" below
hold on;
trimesh(TRI, vx, vy);
for i = 1:nt
    text((vx(v1(i))+vx(v2(i))+vx(v3(i)))./3, (vy(v1(i))+vy(v2(i))+vy(v3(i)))./3, string(i));
end
plot([0 2], [1 1]);
hold off;

n=500;
% lower bound region would be those triangles that are entirely contained in true zero region
m_star_low = [1,2,3,4,5,6,12,13,14,15,16,17,29,31];
% upper bound region would be those triangles that are entirely or partially contained in true zero region. 
% If a triangle has interaction with the boundary line, it would be only contained in the variable "m_start_high"
m_star_high = [1,2,3,4,5,6,12,13,14,15,16,17,29,31, 7,30,22,23];
m_star_sep = cell(3,1); m_star_sep{1,1} = []; m_star_sep{2,1} = []; m_star_sep{3,1} = 1:nt; 
m_star_s1 = m_star_sep; m_star_s1{2, 1} = m_star_high;
m_star_s2 = m_star_sep; m_star_s2{2, 1} = m_star_low;

tic; outloopTimes=100;
[sep_count_TRI1, cover_count_TRI1, width_records_TRI1, LBM_outloop_TRI1] = CZ_BootstrapCR(2, m_star_sep, m_star_s1, m_star_s2, TRI, n, nc, vx, vy, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, 100, 1, 1, 2, 3.7, 0, outloopTimes) ;
toc; % it took 1467 seconds
sum(sep_count_TRI1) 
% Coverage Probability:
% rng(2) -> [100 96 99] ./ 100
cover_count_TRI1 / outloopTimes % 0.95 
[mean(width_records_TRI1)./nt; std(width_records_TRI1)]

%% Sample size - 1000 (confirmed)

rng(1000);
[p,TRI] = distmesh2d(fd,fh,0.21,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
nv = length(vx); d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt; 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S, grid_T, vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

% Plot triangulation and we manually enter "m_star_low" and "m_star_high" below
hold on;
trimesh(TRI, vx, vy);
for i = 1:nt
    text((vx(v1(i))+vx(v2(i))+vx(v3(i)))./3, (vy(v1(i))+vy(v2(i))+vy(v3(i)))./3, string(i));
end
plot([0 2], [1 1]);
hold off;

n=1000; % assign sample size
m_star_low = [1,2,3,4,5,6,12,13,14,15,16,17,29,31];
m_star_high = [1,2,3,4,5,6,12,13,14,15,16,17,29,31, 7,30,22,23];
m_star_sep = cell(3,1); m_star_sep{1,1} = []; m_star_sep{2,1} = []; m_star_sep{3,1} = 1:nt; 
m_star_s1 = m_star_sep; m_star_s1{2, 1} = m_star_high;
m_star_s2 = m_star_sep; m_star_s2{2, 1} = m_star_low;

tic; outloopTimes=100;
[sep_count_TRI2, cover_count_TRI2, width_records_TRI2, LBM_outloop_TRI2] = CZ_BootstrapCR(100, m_star_sep, m_star_s1, m_star_s2, TRI, n, nc, vx, vy, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, 100, 1, 1, 2, 3.7, 0, outloopTimes) ;
toc; % it took 1910 seconds
sum(sep_count_TRI2) %
% Coverage Probability:
% rng(100) -> [100 97 100] ./ 100
cover_count_TRI2 / outloopTimes % rng(100) -> 0.97
[mean(width_records_TRI2)./nt; std(width_records_TRI2)]


%% Sample size - 2000 (confirmed)
rng(1000);
[p,TRI] = distmesh2d(fd,fh,0.19,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
nv = length(vx); d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt; 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S, grid_T, vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

% Plot triangulation and we manually enter "m_star_low" and "m_star_high" below
hold on;
trimesh(TRI, vx, vy);
for i = 1:nt
    text((vx(v1(i))+vx(v2(i))+vx(v3(i)))./3, (vy(v1(i))+vy(v2(i))+vy(v3(i)))./3, string(i));
end
plot([0 2], [1 1]);
hold off;

n=2000;
m_star_low = [23,25,10,11,17,16,5,21,22,9,8,20,15,14];
m_star_high = [23,25,10,11,17,16,5,21,22,9,8,20,15,14, 1,7,18,24];
m_star_sep = cell(3,1); m_star_sep{1,1} = []; m_star_sep{2,1} = []; m_star_sep{3,1} = 1:nt; 
m_star_s1 = m_star_sep; m_star_s1{2, 1} = m_star_high;
m_star_s2 = m_star_sep; m_star_s2{2, 1} = m_star_low;

tic;
outloopTimes=100;
[sep_count_TRI3, cover_count_TRI3, width_records_TRI3, LBM_outloop_TRI3] = CZ_BootstrapCR(2, m_star_sep, m_star_s1, m_star_s2, TRI, n, nc, vx, vy, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, 100, 1, 1, 2, 3.7, 0, outloopTimes) ;
toc; % it took 2764 seconds
sum(sep_count_TRI3) 
% Coverage Probability:
% rng(2) -> [100 99 100] ./ 100
cover_count_TRI3 / outloopTimes % rng(2) -> 0.99
[mean(width_records_TRI3)./nt; std(width_records_TRI3)]


%% Sample size - 5000

p = p17; TRI = TRI17;
vx = p(:,1); vy = p(:,2);
[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
nv = length(vx); d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt; 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S, grid_T, vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

% Plot triangulation and we manually enter "m_star_low" and "m_star_high" below
hold on;
trimesh(TRI, vx, vy);
for i = 1:nt
    text((vx(v1(i))+vx(v2(i))+vx(v3(i)))./3, (vy(v1(i))+vy(v2(i))+vy(v3(i)))./3, string(i));
end
plot([0 2], [1 1]);
hold off;

n=5000;
% Triangles 23, 41, 5, 6, 3, 27 have overlaps with the horizontal line y=1
m_star_low = [37,38,39,44,22,16,32,45,43,36,35,33,34,20,7,2,1,21,14,4,15];
m_star_high = [37,38,39,44,22,16,32,45,43,36,35,33,34,20,7,2,1,21,14,4,15,  41,42,40,27,5,6,3,23];
m_star_sep = cell(3,1); m_star_sep{1,1} = []; m_star_sep{2,1} = []; m_star_sep{3,1} = 1:nt; 
m_star_s1 = m_star_sep; m_star_s1{2, 1} = m_star_high;
m_star_s2 = m_star_sep; m_star_s2{2, 1} = m_star_low;

tic;
outloopTimes=100;
[sep_count_TRI4, cover_count_TRI4, width_records_TRI4, LBM_outloop_TRI4] = CZ_BootstrapCR(12, m_star_sep, m_star_s1, m_star_s2, TRI, n, nc, vx, vy, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, 100, 1, 1, 2, 3.7, 0, outloopTimes) ;
toc; % it took 10587 seconds
sum(sep_count_TRI4) 
% Coverage Probability:
% seed 100: [100 99 100] ./ 100
% seed 12: [100 99 100] ./ 100
cover_count_TRI4 / outloopTimes
[mean(width_records_TRI4)./nt; std(width_records_TRI4)]


summary_coverage_prob = [
    sum(sep_count_TRI1) ./ 100; ...
    sum(sep_count_TRI2) ./ 100; ...
    sum(sep_count_TRI3) ./ 100; ...
    sum(sep_count_TRI4) ./ 100; ...
];
summary_coverage_prob
% 1.00 0.96 0.99
% 1.00 0.97 1.00
% 1.00 0.99 1.00
% 1.00 0.99 1.00

%% Part V: Performance of different TRI (Optimized, Denser, Sparser, Uniform). We pick sample size n=2000 as example
tic;
n = 2000;
record_table_ise_unpen = zeros(kLoopTime, 1*12);  
record_table_ise_scad = zeros(kLoopTime, 1*12);  
record_table_pe = zeros(kLoopTime, 1*12);  
diag_lamb_vec = zeros(kLoopTime, 1); 

% Triangulation 1: optimized via BIC: h=0.19, nt=32, nc=28
rng(1000); % set seed!

% calling distmesh2d() would also plot triangulation: TRI-BIC.jpg
[p,TRI] = distmesh2d(fd,fh,0.19,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
nv = length(vx); d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt; 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S, grid_T, vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

rng(111);
for i=1:kLoopTime
    temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); 
    while(temp_no <= n) 
        temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); % Use beta distribution to generate observations
        temp_theta = 2*pi*rand(1);
        c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
        if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
            X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
        end
    end
    beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
    X_1 = randn(n,1);X_2 = randn(n,1);X_3 = randn(n,1); epsilon=randn(n, 1);
    Z = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3+epsilon;
    Z_no_epi = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3;
    Z_1 = X_1.*beta_1; Z_2 = X_2.*beta_2; Z_3 = X_3.*beta_3;
    [B, valid_id] = CZ_SPL_est(X,Y,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
   
    mat_Z = zeros(n, m*nc);
    for k = 1:n
        temp1 = (B(k, :).* X_1(k,1));temp2 = (B(k, :).* X_2(k,1));temp3 = (B(k, :).* X_3(k,1));
        mat_Z(k,:) = [temp1, temp2, temp3];
    end
    full_mat_Z = mat_Z; full_Z = Z;
    mat_Z = mat_Z(valid_id, :); Z = Z(valid_id, :);

    b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z; 
    ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat(1:nc)).^2); 
    ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((nc+1):(2*nc))).^2); 
    ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((2*nc+1):(3*nc))).^2);
    record_table_ise_unpen(i, 1) = ISE_1; record_table_ise_unpen(i, 2) = ISE_2; record_table_ise_unpen(i, 3) = ISE_3; %ISE_UNPEN part is done.
    
    nlam = 40; a = 3.7;threshold = 10 ^ (-3); 
    lam_vec = linspace(0.01, 0.4, nlam);
    bic = zeros(nlam, 1); converged_or_not = zeros(nlam, 1);
    for q = 1:nlam
        [p_b_hat, dist_logical, ~] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id),1,0);      
        converged_or_not(q) = dist_logical;
        bic(q) = log(mean((Z - mat_Z * p_b_hat).^2)) + log(length(valid_id)) * sum(p_b_hat ~=0) / length(valid_id);
    end

    [temp_min, temp_index] = min(bic);
    [p_b_hat, dist_logical, tri_index] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id),1,0);      

    ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat(1:nc)).^2); 
    ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((nc+1):(2*nc))).^2); 
    ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((2*nc+1):(3*nc))).^2);
    record_table_ise_scad(i, 1) = ISE_1; record_table_ise_scad(i, 2) = ISE_2; record_table_ise_scad(i, 3) = ISE_3; %ISE_SCAD part is done.
    
    color_p_beta = ones(length(p_b_hat), 1); color_p_beta(p_b_hat == 0) = 0;
    
    % f1 P_e part
    record_table_pe(i, 1) = sum(grid_B(grid_valid_id,:) * p_b_hat(1:nc) == 0) / length(grid_valid_id);
    grid_f2_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+nc:2*nc) == 0);
    record_table_pe(i, 2) = (length(intersect(grid_f2_zeroidx, grid_valid_id)) + length(grid_f2_zeroidx_pred) - 2*length(intersect(grid_f2_zeroidx, grid_f2_zeroidx_pred))) / length(grid_valid_id);
    grid_f3_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+2*nc:3*nc) == 0);
    record_table_pe(i, 3) = (length(grid_f3_zeroidx) + length(grid_f3_zeroidx_pred) - 2*length(intersect(grid_f3_zeroidx, grid_f3_zeroidx_pred))) / length(grid_valid_id); % P_e part is done
    
end

% Triangulation 2: more triangles, h=0.14, nt=81, nc=59
rng(1000);

% calling distmesh2d() would also plot triangulation: TRI-DEN.jpg
[p,TRI] = distmesh2d(fd,fh,0.14,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
nv = length(vx); d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt; 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S, grid_T, vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

rng(111);
for i=1:kLoopTime
    %% Generate points first
    temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); 
    while(temp_no <= n) 
        temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); % Use beta distribution to generate observations
        temp_theta = 2*pi*rand(1);
        c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
        if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
            X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
        end
    end
    beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
    X_1 = randn(n,1);X_2 = randn(n,1);X_3 = randn(n,1); epsilon=randn(n, 1);
    Z = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3+epsilon;
    Z_no_epi = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3;
    Z_1 = X_1.*beta_1; Z_2 = X_2.*beta_2; Z_3 = X_3.*beta_3;
    [B, valid_id] = CZ_SPL_est(X,Y,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
   
    mat_Z = zeros(n, m*nc);
    for k = 1:n
        temp1 = (B(k, :).* X_1(k,1));temp2 = (B(k, :).* X_2(k,1));temp3 = (B(k, :).* X_3(k,1));
        mat_Z(k,:) = [temp1, temp2, temp3];
    end
    full_mat_Z = mat_Z; full_Z = Z;
    mat_Z = mat_Z(valid_id, :); Z = Z(valid_id, :);
        
    b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z; 
    ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat(1:nc)).^2); 
    ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((nc+1):(2*nc))).^2); 
    ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((2*nc+1):(3*nc))).^2);
    record_table_ise_unpen(i, 4) = ISE_1; record_table_ise_unpen(i, 5) = ISE_2; record_table_ise_unpen(i, 6) = ISE_3; %ISE_UNPEN part is done.
    
    nlam = 40; a = 3.7;threshold = 10 ^ (-3); 
    lam_vec = linspace(0.01, 0.4, nlam);
    bic = zeros(nlam, 1); converged_or_not = zeros(nlam, 1);
    for q = 1:nlam
        [p_b_hat, dist_logical, ~] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id),1,0);      
        converged_or_not(q) = dist_logical;
        bic(q) = log(mean((Z - mat_Z * p_b_hat).^2)) + log(length(valid_id)) * sum(p_b_hat ~=0) / length(valid_id);
    end
   
    [temp_min, temp_index] = min(bic);
    [p_b_hat, dist_logical, tri_index] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id),1,0);      
    
    ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat(1:nc)).^2); 
    ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((nc+1):(2*nc))).^2); 
    ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((2*nc+1):(3*nc))).^2);
    record_table_ise_scad(i, 4) = ISE_1; record_table_ise_scad(i, 5) = ISE_2; record_table_ise_scad(i, 6) = ISE_3; %ISE_SCAD part is done.
    
    color_p_beta = ones(length(p_b_hat), 1); color_p_beta(p_b_hat == 0) = 0;
    
    % f1 P_e part
    record_table_pe(i, 4) = sum(grid_B(grid_valid_id,:) * p_b_hat(1:nc) == 0) / length(grid_valid_id);
    grid_f2_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+nc:2*nc) == 0);
    record_table_pe(i, 5) = (length(intersect(grid_f2_zeroidx, grid_valid_id)) + length(grid_f2_zeroidx_pred) - 2*length(intersect(grid_f2_zeroidx, grid_f2_zeroidx_pred))) / length(grid_valid_id);
    grid_f3_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+2*nc:3*nc) == 0);
    record_table_pe(i, 6) = (length(grid_f3_zeroidx) + length(grid_f3_zeroidx_pred) - 2*length(intersect(grid_f3_zeroidx, grid_f3_zeroidx_pred))) / length(grid_valid_id); % P_e part is done
    
end

% Triangulation 3: few triangles, h=0.38, nt=13, nc=13
rng(1000);

% calling distmesh2d() would also plot triangulation: TRI-SPARSE.jpg
[p,TRI] = distmesh2d(fd,fh,0.38,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
nv = length(vx); d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt; 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S, grid_T, vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

rng(111);
for i=1:kLoopTime
    %% Generate points first
    temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); 
    while(temp_no <= n) 
        temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); % Use beta distribution to generate observations
        temp_theta = 2*pi*rand(1);
        c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
        if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
            X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
        end
    end
    beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
    X_1 = randn(n,1);X_2 = randn(n,1);X_3 = randn(n,1); epsilon=randn(n, 1);
    Z = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3+epsilon;
    Z_no_epi = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3;
    Z_1 = X_1.*beta_1; Z_2 = X_2.*beta_2; Z_3 = X_3.*beta_3;
    [B, valid_id] = CZ_SPL_est(X,Y,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
   
    mat_Z = zeros(n, m*nc);
    for k = 1:n
        temp1 = (B(k, :).* X_1(k,1));temp2 = (B(k, :).* X_2(k,1));temp3 = (B(k, :).* X_3(k,1));
        mat_Z(k,:) = [temp1, temp2, temp3];
    end
    full_mat_Z = mat_Z; full_Z = Z;
    mat_Z = mat_Z(valid_id, :); Z = Z(valid_id, :);
        
    b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc) ) \ transpose(mat_Z) * Z; 
    ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat(1:nc)).^2); 
    ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((nc+1):(2*nc))).^2); 
    ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((2*nc+1):(3*nc))).^2);
    record_table_ise_unpen(i, 7) = ISE_1; record_table_ise_unpen(i, 8) = ISE_2; record_table_ise_unpen(i, 9) = ISE_3; %ISE_UNPEN part is done.
    
    nlam = 40; a = 3.7;threshold = 10 ^ (-3); 
    lam_vec = linspace(0.01, 0.4, nlam);
    bic = zeros(nlam, 1); converged_or_not = zeros(nlam, 1);
    for q = 1:nlam
        [p_b_hat, dist_logical, ~] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id),1,0);      
        converged_or_not(q) = dist_logical;
        bic(q) = log(mean((Z - mat_Z * p_b_hat).^2)) + log(length(valid_id)) * sum(p_b_hat ~=0) / length(valid_id);
    end
   
    [temp_min, temp_index] = min(bic);
    [p_b_hat, dist_logical, tri_index] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id),1,0);      

    ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat(1:nc)).^2); 
    ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((nc+1):(2*nc))).^2); 
    ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((2*nc+1):(3*nc))).^2);
    record_table_ise_scad(i, 7) = ISE_1; record_table_ise_scad(i, 8) = ISE_2; record_table_ise_scad(i, 9) = ISE_3; %ISE_SCAD part is done.
    
    color_p_beta = ones(length(p_b_hat), 1); color_p_beta(p_b_hat == 0) = 0;
    
    % f1 P_e part
    record_table_pe(i, 7) = sum(grid_B(grid_valid_id,:) * p_b_hat(1:nc) == 0) / length(grid_valid_id);
    grid_f2_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+nc:2*nc) == 0);
    record_table_pe(i, 8) = (length(intersect(grid_f2_zeroidx, grid_valid_id)) + length(grid_f2_zeroidx_pred) - 2*length(intersect(grid_f2_zeroidx, grid_f2_zeroidx_pred))) / length(grid_valid_id);
    grid_f3_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+2*nc:3*nc) == 0);
    record_table_pe(i, 9) = (length(grid_f3_zeroidx) + length(grid_f3_zeroidx_pred) - 2*length(intersect(grid_f3_zeroidx, grid_f3_zeroidx_pred))) / length(grid_valid_id); % P_e part is done
    
end

% Triangulation 4: Uniform distributed, nt=32 (same as the TRI 1), nc=29
rng(1000);

% calling distmesh2d() would also plot triangulation: TRI-UNIF.jpg
[p,TRI] = distmesh2d(fd,@huniform,0.39,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
nv = length(vx); d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt; 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S, grid_T, vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

rng(111);
for i=1:kLoopTime
    % Generate points first
    temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); 
    while(temp_no <= n) 
        temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); % Use beta distribution to generate observations
        temp_theta = 2*pi*rand(1);
        c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
        if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
            X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
        end
    end
    
    beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
    X_1 = randn(n,1);X_2 = randn(n,1);X_3 = randn(n,1); epsilon=randn(n, 1);
    Z = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3+epsilon;
    Z_no_epi = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3;
    Z_1 = X_1.*beta_1; Z_2 = X_2.*beta_2; Z_3 = X_3.*beta_3;
    [B, valid_id] = CZ_SPL_est(X,Y,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
   
    mat_Z = zeros(n, m*nc);
    for k = 1:n
        temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
        mat_Z(k,:) = [temp1, temp2, temp3];
    end
    full_mat_Z = mat_Z; full_Z = Z;
    mat_Z = mat_Z(valid_id, :); Z = Z(valid_id, :);
        
    b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc) ) \ transpose(mat_Z) * Z; 
    ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat(1:nc)).^2); 
    ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((nc+1):(2*nc))).^2); 
    ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((2*nc+1):(3*nc))).^2);
    record_table_ise_unpen(i, 10) = ISE_1; record_table_ise_unpen(i, 11) = ISE_2; record_table_ise_unpen(i, 12) = ISE_3; %ISE_UNPEN part is done.
    
    nlam = 40; a = 3.7;threshold = 10 ^ (-3); 
    lam_vec = linspace(0.01, 0.4, nlam);
    bic = zeros(nlam, 1); converged_or_not = zeros(nlam, 1);
    for q = 1:nlam
        [p_b_hat, dist_logical, ~] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id),2,1);      
        converged_or_not(q) = dist_logical;
        bic(q) = log(mean((Z - mat_Z * p_b_hat).^2)) + log(length(valid_id)) * sum(p_b_hat ~=0) / length(valid_id);
    end
   
    [temp_min, temp_index] = min(bic);
    [p_b_hat, dist_logical, tri_index] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id),2,1);      

    ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat(1:nc)).^2); 
    ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((nc+1):(2*nc))).^2); 
    ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((2*nc+1):(3*nc))).^2);
    record_table_ise_scad(i, 10) = ISE_1; 
    record_table_ise_scad(i, 11) = ISE_2; 
    record_table_ise_scad(i, 12) = ISE_3; %ISE_SCAD part is done.
    
    color_p_beta = ones(length(p_b_hat), 1); color_p_beta(p_b_hat == 0) = 0;
    
    % f1 P_e part
    record_table_pe(i, 10) = sum(grid_B(grid_valid_id,:) * p_b_hat(1:nc) == 0) / length(grid_valid_id);
    grid_f2_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+nc:2*nc) == 0);
    record_table_pe(i, 11) = (length(intersect(grid_f2_zeroidx, grid_valid_id)) + length(grid_f2_zeroidx_pred) - 2*length(intersect(grid_f2_zeroidx, grid_f2_zeroidx_pred))) / length(grid_valid_id);
    grid_f3_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+2*nc:3*nc) == 0);
    record_table_pe(i, 12) = (length(grid_f3_zeroidx) + length(grid_f3_zeroidx_pred) - 2*length(intersect(grid_f3_zeroidx, grid_f3_zeroidx_pred))) / length(grid_valid_id); % P_e part is done
    
end
toc; % entire section took 193 seconds

str_ise_unpen = zeros(12,1); str_ise_unpen = string(str_ise_unpen);
for i = 1:12
   str_ise_unpen(i) = sprintf('%.4f (%.4f)', round(mean(record_table_ise_unpen(:, i)), 4), round(std(record_table_ise_unpen(:, i)), 4)) ;
end
str_ise_scad = zeros(12,1); str_ise_scad = string(str_ise_scad);
for i = 1:12
   str_ise_scad(i) = sprintf('%.4f (%.4f)', round(mean(record_table_ise_scad(:, i)), 4), round(std(record_table_ise_scad(:, i)),4)) ;
end
str_pe = zeros(12,1); str_pe = string(str_pe);
for i = 1:12
   str_pe(i) = sprintf('%.4f (%.4f)', round(mean(record_table_pe(:, i)), 4), round(std(record_table_pe(:, i)),4)) ;
end
summary_T = [string('Metric') string('Func') 'Optimized h' 'Denser TRI' 'Sparser TRI' 'Uniform'; ...
    'P_e' string('F_1') str_pe(1) str_pe(4) str_pe(7) str_pe(10);...
    'P_e' string('F_2') str_pe(2) str_pe(5) str_pe(8) str_pe(11);...
    'P_e' string('F_3') str_pe(3) str_pe(6) str_pe(9) str_pe(12);...
    'ISE_UNPEN' string('F_1') str_ise_unpen(1) str_ise_unpen(4) str_ise_unpen(7) str_ise_unpen(10);...
    'ISE_UNPEN' string('F_2') str_ise_unpen(2) str_ise_unpen(5) str_ise_unpen(8) str_ise_unpen(11);...
    'ISE_UNPEN' string('F_3') str_ise_unpen(3) str_ise_unpen(6) str_ise_unpen(9) str_ise_unpen(12);...
    'ISE_SCAD' string('F_1') str_ise_scad(1) str_ise_scad(4) str_ise_scad(7) str_ise_scad(10);...
    'ISE_SCAD' string('F_2') str_ise_scad(2) str_ise_scad(5) str_ise_scad(8) str_ise_scad(11);...
    'ISE_SCAD' string('F_3') str_ise_scad(3) str_ise_scad(6) str_ise_scad(9) str_ise_scad(12);];
summary_T 
% content of summary_T:
%     "Metric"       "Func"    "Oprimized h"        "Denser TRI"         "Sparser TRI"        "Uniform"        
%     "P_e"          "F_1"     "0.0000 (0.0000)"    "0.0007 (0.0066)"    "0.0000 (0.0000)"    "0.0133 (0.0126)"
%     "P_e"          "F_2"     "0.0652 (0.0420)"    "0.0915 (0.0464)"    "0.1769 (0.0701)"    "0.2160 (0.0201)"
%     "P_e"          "F_3"     "0.0416 (0.0696)"    "0.0753 (0.0671)"    "0.0155 (0.0618)"    "0.0920 (0.0590)"
%     "ISE_UNPEN"    "F_1"     "0.1795 (0.0242)"    "0.1661 (0.0649)"    "0.3323 (0.0476)"    "0.2642 (0.0713)"
%     "ISE_UNPEN"    "F_2"     "0.0310 (0.0123)"    "0.0892 (0.0452)"    "0.0647 (0.0121)"    "0.0850 (0.0467)"
%     "ISE_UNPEN"    "F_3"     "0.0295 (0.0134)"    "0.0917 (0.0711)"    "0.0155 (0.0095)"    "0.1036 (0.0794)"
%     "ISE_SCAD"     "F_1"     "0.1777 (0.0230)"    "0.1628 (0.0654)"    "0.3310 (0.0476)"    "0.3370 (0.2137)"
%     "ISE_SCAD"     "F_2"     "0.0244 (0.0118)"    "0.0726 (0.0384)"    "0.0717 (0.0141)"    "0.1328 (0.1524)"
%     "ISE_SCAD"     "F_3"     "0.0055 (0.0103)"    "0.0399 (0.0614)"    "0.0010 (0.0047)"    "0.1852 (0.3217)"
