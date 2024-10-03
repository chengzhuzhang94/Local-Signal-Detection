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
for j = 1:length(h_opt)
    rng(1000); % Set seed to make sure the generated triangulations are identical
    h_now = h_opt(j); 
    disp(h_now)
    [p,TRI] = distmesh2d(fd, fh, h_now, [0,0;2,2], [0,0;0,2;2,0;2,2]);
    % You need to output some Triangulations manually for later repeated use
    if abs(h_now - 0.17) < 0.001
        p17 = p; TRI17 = TRI;
    end   
    if abs(h_now - 0.18) < 0.001
        p18 = p; TRI18 = TRI;
    end 
    if abs(h_now - 0.19) < 0.001
        p19 = p; TRI19 = TRI;
    end 
end

% In total, we have 4 different sample sizes
n_choice = [500, 1000, 2000, 5000]; 
h_choice = 0.14:0.01:0.22; 
m = 3;
bic_records = zeros(length(n_choice), length(h_choice));

% sample size: 500
rng(100);
for i=1
    n = n_choice(i);
    temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); % Generate points first 
    while(temp_no <= n)
        temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); 
        % Use beta distribution to generate locations of data points
        temp_theta = 2*pi*rand(1);
        c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
        if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
            X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
        end
    end
    % generate response values via defined varying coefficient functions
    beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
    X_1 = randn(n,1); X_2 = randn(n,1); X_3 = randn(n,1); epsilon=randn(n, 1);
    Z = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3+epsilon;
    ori_Z = Z;
    
    for j=1:length(h_choice)
        % generate Triangulations
        disp(['The current n value:', num2str(n), ' the current h value: ', num2str(h_choice(j))])
        if abs(h_choice(j) - 0.17) < 0.0001
            p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
        else
            rng(1000);
            [p,TRI] = distmesh2d(fd, fh, h_choice(j),[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
        end 
              
        [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
           vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
        nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;
        
        % Call "CZ_SPL_est" to generate design matrix of Bernstein basis
        % polynomials
        [B, valid_id] = CZ_SPL_est(X, Y, vx, vy, TRI,v1,v2,v3,nt,nc,nv,d);
        
        % To get the final design matrix, we need to multiply covariant
        % values to Bernstein basis polynomials
        mat_Z = zeros(n, m*nc);
        for k = 1:n
            temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
            mat_Z(k,:) = [temp1, temp2, temp3];
        end
        full_mat_Z = mat_Z;
        mat_Z = full_mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
        
        b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z;
        
        % Calculate BIC value
        bic_records(i, j) = log(mean((Z - mat_Z * b_hat).^2)) + log(length(valid_id)) * sum(b_hat ~=0) / length(valid_id);
    end
    disp(['The experiment of n:', num2str(n), ' is finished'])
    [~, argmin] = min(bic_records(i, :), [], 2); h_choice(argmin)
    disp(['The best h value is ', num2str(h_choice(argmin))])
end
[~, argmin] = min(bic_records(1, :), [], 2); h_choice(argmin) % best h choice is 0.21

% sample size: 1000. Similar to the case sample size=500
rng(200);
for i=2
    n = n_choice(i);
    temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); % Generate points first 
    while(temp_no <= n)
        temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); 
        temp_theta = 2*pi*rand(1);
        c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
        if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
            X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
        end
    end
    beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
    X_1 = randn(n,1); X_2 = randn(n,1); X_3 = randn(n,1); epsilon=randn(n, 1);
    Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3 + epsilon;
    ori_Z = Z;
    
    for j=1:length(h_choice)
        disp(['The current n value:', num2str(n), ' the current h value: ', num2str(h_choice(j))])
        if h_choice(j) == 0.17
            p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
        else
            rng(1000);
            [p,TRI] = distmesh2d(fd, fh, h_choice(j),[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
        end 
              
        [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
           vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
        nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;
        [B, valid_id] = CZ_SPL_est(X, Y, vx, vy, TRI,v1,v2,v3,nt,nc,nv,d);
        
        mat_Z = zeros(n, m*nc);
        for k = 1:n
            temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
            mat_Z(k,:) = [temp1, temp2, temp3];
        end
        full_mat_Z = mat_Z;
        mat_Z = full_mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
        
        b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z;
        bic_records(i, j) = log(mean((Z - mat_Z * b_hat).^2)) + log(length(valid_id)) * sum(b_hat ~=0) / length(valid_id);
    end
    disp(['The experiment of n:', num2str(n), ' is finished'])
    [~, argmin] = min(bic_records(i, :), [], 2); h_choice(argmin)
    disp(['The best h value is ', num2str(h_choice(argmin))])
end
[~, argmin] = min(bic_records(1:2, :), [], 2); h_choice(argmin) % best h choice is 0.21

% sample size: 2000. Similar to the case sample size=500
rng(200);
for i=3
    n = n_choice(i);
    temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); % Generate points first 
    while(temp_no <= n)
        temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); 
        temp_theta = 2*pi*rand(1);
        c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
        if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
            X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
        end
    end
    beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
    X_1 = randn(n,1); X_2 = randn(n,1); X_3 = randn(n,1); epsilon=randn(n, 1);
    Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3 + epsilon;
    ori_Z = Z;
    
    for j=1:length(h_choice)
        disp(['The current n value:', num2str(n), ' the current h value: ', num2str(h_choice(j))])
        if h_choice(j) == 0.17
            p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
        else
            rng(1000);
            [p,TRI] = distmesh2d(fd, fh, h_choice(j),[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
        end 
              
        [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
           vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
        nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;
        [B, valid_id] = CZ_SPL_est(X, Y, vx, vy, TRI,v1,v2,v3,nt,nc,nv,d);
        
        mat_Z = zeros(n, m*nc);
        for k = 1:n
            temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
            mat_Z(k,:) = [temp1, temp2, temp3];
        end
        full_mat_Z = mat_Z;
        mat_Z = full_mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
        
        b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z;
        bic_records(i, j) = log(mean((Z - mat_Z * b_hat).^2)) + log(length(valid_id)) * sum(b_hat ~=0) / length(valid_id);
    end
    disp(['The experiment of n:', num2str(n), ' is finished'])
    [~, argmin] = min(bic_records(i, :), [], 2); h_choice(argmin)
    disp(['The best h value is ', num2str(h_choice(argmin))])
end
[~, argmin] = min(bic_records(1:3, :), [], 2); h_choice(argmin) % best h choice is 0.19

% sample size: 5000. Similar to the case sample size=500
rng(100);
for i=4
    n = n_choice(i);
    temp_no = 1; X = zeros(n, 1); Y = zeros(n, 1); % Generate points first 
    while(temp_no <= n)
        temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); 
        temp_theta = 2*pi*rand(1);
        c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
        if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
            X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
        end
    end
    beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
    X_1 = randn(n,1); X_2 = randn(n,1); X_3 = randn(n,1); epsilon=randn(n, 1);
    Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3 + epsilon;
    ori_Z = Z;
    
    for j=1:length(h_choice)
        disp(['The current n value:', num2str(n), ' the current h value: ', num2str(h_choice(j))])
        if h_choice(j) == 0.17
            p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
        else
            rng(1000);
            [p,TRI] = distmesh2d(fd, fh, h_choice(j),[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
        end 
              
        [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
           vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
        nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;
        [B, valid_id] = CZ_SPL_est(X, Y, vx, vy, TRI,v1,v2,v3,nt,nc,nv,d);
        
        mat_Z = zeros(n, m*nc);
        for k = 1:n
            temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
            mat_Z(k,:) = [temp1, temp2, temp3];
        end
        full_mat_Z = mat_Z;
        mat_Z = full_mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
        
        b_hat = (transpose(mat_Z) * mat_Z + 6 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z;
        bic_records(i, j) = log(mean((Z - mat_Z * b_hat).^2)) + log(length(valid_id)) * sum(b_hat ~=0) / length(valid_id);
    end
    disp(['The experiment of n:', num2str(n), ' is finished'])
    [~, argmin] = min(bic_records(i, :), [], 2); h_choice(argmin)
    disp(['The best h value is ', num2str(h_choice(argmin))])
end
[~, argmin] = min(bic_records(1:4, :), [], 2); h_choice(argmin) % best h choice is 0.17

% Now we get the best h choices are [0.21, 0.21, 0.19, 0.17] for sample sizes [500, 1000, 2000, 5000] respectively

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

% rng(5), it took 150 seconds to finish
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
