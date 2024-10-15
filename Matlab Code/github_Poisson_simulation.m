% Guide

% Part I: calculate optimized TRI for each sample size
% Part II: repeat data generation and fitting to get summary table of P_e and ISE
% Part III: running time check (only used for README.md)

%% Part I
m = 3;
kLoopTime = 100; % alternative value is 300. We repeat estimation multiple times to calculate the SE of ISE & P_e

% UDF values
f = @(x,y) sqrt((x-0.5).^2 + (y-0.5).^2); fd = @(p) ddiff(drectangle(p, 0, 2, 0, 2), dcircle(p, 1, 1, 0.5)); fh=@(p) 1+4*dcircle(p,1,1,0.5);
f1 = @(x,y) -1.* sin(2.*pi./(sqrt(2)-0.5) .* (sqrt((x-1).^2+(y-1).^2)-0.5))-1.5; 
f2 = @(x,y) 2.* (exp((y-1).* (y>=1)) - 1) + 1.5 .* (sqrt((x-1).^2+(y-1).^2) -0.5).*(y>=1); 
f3 = @(x,y) 0 .* y; 

% Generate grids. They fall in [0, 2]x[0, 2] but not falling into the central circle
grid_len = 200;
grid_s = linspace(0, 2, grid_len + 2); grid_t = linspace(0, 2, grid_len + 2); grid_s = grid_s(2:(grid_len+1)); grid_t = grid_t(2:(grid_len+1));
[grid_S, grid_T] = meshgrid(grid_s, grid_t); grid_S = reshape(grid_S, [grid_len^2, 1]); grid_T = reshape(grid_T, [grid_len^2, 1]);
grid_idx = (grid_S - 1).^2 + (grid_T - 1).^2 > 0.25; % record indices of points not falling into the central circle

% Only valid grid points are kept
grid_S = grid_S(grid_idx); grid_T = grid_T(grid_idx); 
grid_f1 = f1(grid_S, grid_T); grid_f2 = f2(grid_S, grid_T); grid_f3 = f3(grid_S, grid_T); 
grid_f2_zeroidx = find(grid_f2 == 0); grid_f3_zeroidx = find(grid_f3 == 0); 

% generate record tables that store info of each loop
record_table_ise_unpen = zeros(kLoopTime, 1*12);  
record_table_ise_scad = zeros(kLoopTime, 1*12);  
record_table_pe = zeros(kLoopTime, 1*12);  
diag_lamb_vec = zeros(kLoopTime, 1); 

%% Part II: Calcualte the Best h values for each sample size: n
h_opt = 0.13:0.01:0.23;
for j = 1:length(h_opt)
    rng(1100); % Set seed to make sure the generated triangulations are identical
    h_now = h_opt(j); 
    disp(h_now)
    [p,TRI] = distmesh2d(fd, fh, h_now, [0,0;2,2], [0,0;0,2;2,0;2,2]);
    % You need to output p17 and TRI17 specifically for later use
    % sometimes, h == 0.17 would lead to infinite running, so we manually
    % save TRI when h == 0.17
    if abs(h_now - 0.17) < 0.001
        p17 = p; TRI17 = TRI;
    end       
end

rng(1100);
[p,TRI] = distmesh2d(fd, fh, 0.19, [0,0;2,2], [0,0;0,2;2,0;2,2]);

n_choice = [1000, 2000, 5000]; 
h_choice = 0.17:0.01:0.23; 
bic_records = zeros(m+1, length(h_choice));

rng(11);
for i=1:length(n_choice)
    n = n_choice(i);
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
    X_1 = unifrnd(0, 1, [n, 1]); X_2 = unifrnd(0, 1, [n, 1]); X_3 = unifrnd(0, 1, [n, 1]);
    Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3;
    
    Z = poissrnd(exp(Z)); ori_Z = Z;
    hist(X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3);
    
    for j=1:length(h_choice)
        disp(['The current n value:', num2str(n), ' the current h value: ', num2str(h_choice(j))])
        if abs(h_choice(j) - 0.17) < 0.0001
            p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
        elseif abs(h_choice(j) - 0.24) < 0.0001
            p = p24; TRI = TRI24; vx = p(:,1); vy = p(:,2);
        elseif h_choice(j) <= 0.23
            rng(1100);[p,TRI] = distmesh2d(fd, fh, h_choice(j),[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
        else
            rng(11);[p,TRI] = distmesh2d(fd, fh, h_choice(j),[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
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
        [b_hat dev stats] = glmfit(mat_Z, Z, 'poisson', 'link', 'log', 'constant', 'off'); 
        exp_preds = mat_Z * b_hat;
        bic_records(i, j) = -2*sum(Z .* exp_preds) + 2*sum(exp(exp_preds)) + log(length(exp_preds)) * sum(b_hat ~= 0);

    end
    disp(['The experiment of n:', num2str(n), ' is finished'])
    [~, argmin] = min(bic_records(i, :), [], 2); h_choice(argmin)
    disp(['The best h value is ', num2str(h_choice(argmin))])
end
[~, argmin] = min(bic_records(1:3, :), [], 2); h_choice(argmin)
% best h_choice for n=[1000, 2000, 5000] is [0.21, 0.21, 0.21] with rng(11)

%% Experiments for each sample size
tic; 
kLoopTime = 100;
record_table = zeros(kLoopTime, 3*9);  diag_lamb_vec = zeros(kLoopTime, 3); 
n_choice = [1000, 2000, 5000]; best_h_values = [0.21, 0.21, 0.21]; % best_h_values is from above chunk

rng(1111);
for i=1:length(n_choice)
    
    % Section: generating TRI based on the best h, which was got above
    n = n_choice(i); n_buffer = n+500;
    rng(1100) ;h_now = best_h_values(i);  
    if (h_now == 0.17) 
        p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
    else
        [p,TRI] = distmesh2d(fd,fh,h_now,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
    end
    [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
        vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
    nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt; [grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
    
    disp(['The experiment of n:', num2str(n), ' started'])
    for j=1:kLoopTime
        if rem(j, 10) == 1
            disp(['The current loop No. is ', num2str(j)])
        end
        % randomly generaing locations of data points
        temp_no = 1; X = zeros(n_buffer, 1); Y = zeros(n_buffer, 1); 
        while(temp_no <= n_buffer) 
            temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); % Use beta distribution to generate observations
            temp_theta = 2*pi*rand(1);
            c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
            if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
                X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
            end
        end
        % generate response variables by using data points location and varying coefficient functions
        
        beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
        % covariate follow uniform distributions [0, 1]
        X_1 = unifrnd(0, 1, [n_buffer, 1]); X_2 = unifrnd(0, 1, [n_buffer, 1]); X_3 = unifrnd(0, 1, [n_buffer, 1]);
        Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3;
        
        % Generate response via poissrnd()
        Z = poissrnd(exp(Z)); ori_Z = Z;
        
        % Generate design matrix
        [B, valid_id] = CZ_SPL_est(X,Y,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d); % calculate design matrix (Bernstein basis polynomial values) for each points
        valid_id = valid_id(1:n);
        
        mat_Z = zeros(n_buffer, m*nc);
        for k = 1:n
            temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
            mat_Z(k,:) = [temp1, temp2, temp3];
        end
        full_mat_Z = mat_Z; full_Z = Z;
        mat_Z = mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
        
        % UNPEN estimator: the initial value for SCAD estimator
        [b_hat dev stats] = glmfit(mat_Z, Z, 'poisson', 'link', 'log', 'constant', 'off');
        
        % section: plot b_hat
        hold on;
        trimesh(TRI, vx, vy);
        for i_tri = 1:nc
            txt = string(round(b_hat(nc+i_tri), 3));
            text(vx(i_tri), vy(i_tri), txt)
        end
        hold off;
        
        hold on;
        trimesh(TRI, vx, vy);
        for i_tri = 1:nc
            txt = string(round(b_hat(2*nc+i_tri), 3));
            text(vx(i_tri), vy(i_tri), txt)
        end
        hold off;
        
        % plot beta
        hold on;
        subplot(3,2,1); trimesh(TRI, vx, vy); scatter(X(valid_id), Y(valid_id), 5, 0.25.*f1(X(valid_id), Y(valid_id))); colorbar(); title('Real Beta1')
        subplot(3,2,2); trimesh(TRI, vx, vy); scatter(X(valid_id), Y(valid_id), 5, mat_Z(:, (1):nc)*b_hat((1):nc)); colorbar(); title('Estimated Beta1')
        subplot(3,2,3); trimesh(TRI, vx, vy); scatter(X(valid_id), Y(valid_id), 5, 0.25.*f2(X(valid_id), Y(valid_id))); colorbar(); title('Real Beta2')
        subplot(3,2,4); trimesh(TRI, vx, vy); scatter(X(valid_id), Y(valid_id), 5, mat_Z(:, (1+nc):2*nc)*b_hat((1+nc):2*nc)); colorbar(); title('Estimated Beta2')
        subplot(3,2,5); trimesh(TRI, vx, vy); scatter(X(valid_id), Y(valid_id), 5, mat_Z(:, (1+2*nc):3*nc)*b_hat((1+2*nc):3*nc)); colorbar(); title('Estimated Beta3');
        hold off;
        
        ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat(1:nc)).^2); 
        ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((nc+1):(2*nc))).^2); 
        ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * b_hat((2*nc+1):(3*nc))).^2);
        record_table(j, 6+i) = ISE_1; 
        record_table(j, 15+i) = ISE_2; 
        record_table(j, 24+i) = ISE_3; 
        %ISE_UNPEN part is done.
        
        nlam = 50; a = 3.7; threshold = 10 ^ (-3); 
        lam_vec = linspace(0.0001, 5, nlam);
        bic = zeros(nlam, 3); converged_or_not = zeros(nlam, 1);
        for q = 1:nlam
            [p_b_hat, dist_logical] = update_p_b_hat_poisson(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id));      
            converged_or_not(q) = dist_logical;
            exp_preds = mat_Z * p_b_hat;
            bic(q, 1) = -2*sum(Z .* exp_preds) + 2*sum(exp(exp_preds)) + log(length(exp_preds)) * sum(p_b_hat ~= 0);
            bic(q, 2) = -2*sum(Z .* exp_preds) + 2*sum(exp(exp_preds));
            bic(q, 3) = log(length(exp_preds)) * sum(p_b_hat ~= 0);
        end
        scatter(lam_vec, bic(:, 1));
        scatter(lam_vec, bic(:, 2));
        scatter(lam_vec, bic(:, 3));
        
        [temp_min, temp_index] = min(bic(:, 1)); diag_lamb_vec(j,i) = lam_vec(temp_index);
        [p_b_hat, dist_logical] = update_p_b_hat_poisson(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id));
        
        ISE_1 = mean((grid_f1(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat(1:nc)).^2); 
        ISE_2 = mean((grid_f2(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((nc+1):(2*nc))).^2); 
        ISE_3 = mean((grid_f3(grid_valid_id) - grid_B(grid_valid_id,:) * p_b_hat((2*nc+1):(3*nc))).^2);       
        record_table(j, 3+i) = ISE_1; record_table(j, 12+i) = ISE_2; record_table(j, 21+i) = ISE_3;
        %ISE_SCAD part is done.
        
        record_table(j, i) = sum(grid_B(grid_valid_id,:) * p_b_hat(1:nc) == 0) / length(grid_valid_id);
        
        grid_f2_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+nc:2*nc) == 0);
        record_table(j, 9+i) = (length(intersect(grid_f2_zeroidx, grid_valid_id)) + length(grid_f2_zeroidx_pred) - 2*length(intersect(grid_f2_zeroidx, grid_f2_zeroidx_pred))) / length(grid_valid_id);
        grid_f3_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+2*nc:3*nc) == 0);
        record_table(j, 18+i) = (length(intersect(grid_f3_zeroidx, grid_valid_id)) + length(grid_f3_zeroidx_pred) - 2*length(intersect(grid_f3_zeroidx, grid_f3_zeroidx_pred))) / length(grid_valid_id);
        % P_e part is done
        scatter(grid_S(grid_valid_id), grid_T(grid_valid_id), [], grid_B(grid_valid_id,:) * p_b_hat(1+nc:2*nc)); colorbar();
        scatter(grid_S(grid_valid_id), grid_T(grid_valid_id), [], f2(grid_S(grid_valid_id), grid_T(grid_valid_id))); colorbar();
        
        
        hold on;
        subplot(3,2,1); trimesh(TRI, vx, vy); scatter(X(valid_id), Y(valid_id), 5, 0.25.*f1(X(valid_id), Y(valid_id))); colorbar(); title('Real Beta1')
        subplot(3,2,2); trimesh(TRI, vx, vy); scatter(X(valid_id), Y(valid_id), 5, mat_Z(:, (1):nc)*p_b_hat((1):nc)); colorbar(); title('Estimated Beta1')
        subplot(3,2,3); trimesh(TRI, vx, vy); scatter(X(valid_id), Y(valid_id), 5, 0.25.*f2(X(valid_id), Y(valid_id))); colorbar(); title('Real Beta2')
        subplot(3,2,4); trimesh(TRI, vx, vy); scatter(X(valid_id), Y(valid_id), 5, mat_Z(:, (1+nc):2*nc)*p_b_hat((1+nc):2*nc)); colorbar(); title('Estimated Beta2')
        subplot(3,2,5); trimesh(TRI, vx, vy); scatter(X(valid_id), Y(valid_id), 5, mat_Z(:, (1+2*nc):3*nc)*p_b_hat((1+2*nc):3*nc)); colorbar(); title('Estimated Beta3');
        hold off;
    end
end

toc; % running time 1567 seconds

% Display the mean and median as the final summary table
str = zeros(27,1); str = string(str);
for i = 1:27
   str(i) = sprintf('%.4f (%.4f)', mean(record_table(:, i)), std(record_table(:, i))) ;
end
summary_T = [string('n') string('F') string('P_e') string('ISE_SCAD') string('ISE_UNPEN'); ...
    '1000' 'F1' str(1) str(4) str(7);'2000' 'F1' str(2) str(5) str(8);'5000' 'F1' str(3) str(6) str(9);...
    '1000' 'F2' str(10) str(13) str(16);'2000' 'F2' str(11) str(14) str(17);'5000' 'F2' str(12) str(15) str(18);...
    '1000' 'F3' str(19) str(22) str(25);'2000' 'F3' str(20) str(23) str(26);'5000' 'F3' str(21) str(24) str(27);];
summary_T

% Trial Result: rng(1100) (previous default seed)
%     "n"       "F"     "P_e"                "ISE_SCAD"           "ISE_UNPEN"      
%     "1000"    "F1"    "0.0502 (0.0765)"    "1.2018 (0.7396)"    "1.1537 (0.5871)"
%     "2000"    "F1"    "0.0000 (0.0000)"    "0.4243 (0.1216)"    "0.4895 (0.1559)"
%     "5000"    "F1"    "0.0000 (0.0000)"    "0.2299 (0.0538)"    "0.2612 (0.0709)"
%     "1000"    "F2"    "0.2539 (0.0829)"    "0.4721 (0.3737)"    "0.7304 (0.3952)"
%     "2000"    "F2"    "0.2393 (0.0741)"    "0.1555 (0.0904)"    "0.2564 (0.1071)"
%     "5000"    "F2"    "0.2648 (0.0769)"    "0.0658 (0.0388)"    "0.1003 (0.0400)"
%     "1000"    "F3"    "0.1429 (0.1238)"    "0.2606 (0.3198)"    "0.6573 (0.3416)"
%     "2000"    "F3"    "0.1126 (0.1184)"    "0.0761 (0.0943)"    "0.2824 (0.1185)"
%     "5000"    "F3"    "0.0887 (0.1127)"    "0.0163 (0.0257)"    "0.0965 (0.0377)"

% rng(1100): confirmed
%     "n"       "F"     "P_e"                "ISE_SCAD"           "ISE_UNPEN"      
%     "1000"    "F1"    "0.0475 (0.0620)"    "1.1420 (0.5575)"    "1.0440 (0.4506)"
%     "2000"    "F1"    "0.0004 (0.0042)"    "0.4463 (0.1757)"    "0.5192 (0.2255)"
%     "5000"    "F1"    "0.0000 (0.0000)"    "0.2297 (0.0563)"    "0.2517 (0.0638)"
%     "1000"    "F2"    "0.2461 (0.0871)"    "0.4409 (0.3284)"    "0.6559 (0.3101)"
%     "2000"    "F2"    "0.2320 (0.0816)"    "0.1921 (0.1107)"    "0.2816 (0.1345)"
%     "5000"    "F2"    "0.2527 (0.0673)"    "0.0646 (0.0397)"    "0.0949 (0.0394)"
%     "1000"    "F3"    "0.1249 (0.1216)"    "0.2777 (0.4213)"    "0.7129 (0.4051)"
%     "2000"    "F3"    "0.0922 (0.1143)"    "0.0835 (0.1367)"    "0.2929 (0.1429)"
%     "5000"    "F3"    "0.0765 (0.1037)"    "0.0187 (0.0369)"    "0.0942 (0.0462)"
%% Time check
% This section is used to get running time. We use matlab function "tic"
% and "toc" to get lasping time

% UNPEN

kLoopTime = 10;
n_choice = [1000, 2000, 5000]; best_h_values = [0.21, 0.21, 0.21];
rng(1111);
for i=1:length(n_choice)
    tic;
    n = n_choice(i); n_buffer = n+500;
    rng(1100) ;h_now = best_h_values(i);  
    if (h_now == 0.17) 
        p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
    else
        [p,TRI] = distmesh2d(fd,fh,h_now,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
    end
    [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
        vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
    nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt; [grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
    
    disp(['The experiment of n:', num2str(n), ' started'])
    for j=1:1
        temp_no = 1; X = zeros(n_buffer, 1); Y = zeros(n_buffer, 1); 
        while(temp_no <= n_buffer) 
            temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); % Use beta distribution to generate observations
            temp_theta = 2*pi*rand(1);
            c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
            if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
                X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
            end
        end
        beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
        X_1 = unifrnd(0, 1, [n_buffer, 1]); X_2 = unifrnd(0, 1, [n_buffer, 1]); X_3 = unifrnd(0, 1, [n_buffer, 1]);
        Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3;

        Z = poissrnd(exp(Z)); ori_Z = Z;
        [B, valid_id] = CZ_SPL_est(X,Y,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
        valid_id = valid_id(1:n);
        
        mat_Z = zeros(n_buffer, m*nc);
        for k = 1:n
            temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
            mat_Z(k,:) = [temp1, temp2, temp3];
        end
        full_mat_Z = mat_Z; 
        mat_Z = mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
        [b_hat dev stats] = glmfit(mat_Z, Z, 'poisson', 'link', 'log', 'constant', 'off');
        
    end
    toc;
end

% SCAD Tuning and One rep
n_choice = [1000, 2000, 5000]; best_h_values = [0.21, 0.21, 0.21];
rng(1111);
for i=1:length(n_choice)
    tic; % record running time for each n from n_choice
    n = n_choice(i); n_buffer = n+500;
    rng(1100) ;h_now = best_h_values(i);  
    if (h_now == 0.17) 
        p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
    else
        [p,TRI] = distmesh2d(fd,fh,h_now,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
    end
    [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
        vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
    nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt; [grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
    
    disp(['The experiment of n:', num2str(n), ' started'])
    for j=1:1
        % generate 1-time data points (so j only equals 1)
        temp_no = 1; X = zeros(n_buffer, 1); Y = zeros(n_buffer, 1); 
        while(temp_no <= n_buffer) 
            temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); % Use beta distribution to generate observations
            temp_theta = 2*pi*rand(1);
            c_x = temp_r * cos(temp_theta) + 1; c_y = temp_r * sin(temp_theta) + 1;
            if 0 <= c_x && c_x <= 2 && 0<= c_y && c_y <= 2
                X(temp_no) = c_x; Y(temp_no) = c_y; temp_no = temp_no + 1;
            end
        end
        beta_1 = f1(X,Y); beta_2 = f2(X,Y); beta_3 = f3(X,Y); 
        X_1 = unifrnd(0, 1, [n_buffer, 1]); X_2 = unifrnd(0, 1, [n_buffer, 1]); X_3 = unifrnd(0, 1, [n_buffer, 1]);
        Z = X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3;
        
        Z = poissrnd(exp(Z)); ori_Z = Z;
        [B, valid_id] = CZ_SPL_est(X,Y,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
        valid_id = valid_id(1:n);
        
        mat_Z = zeros(n_buffer, m*nc);
        for k = 1:n
            temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
            mat_Z(k,:) = [temp1, temp2, temp3];
        end
        full_mat_Z = mat_Z; full_Z = Z;
        mat_Z = mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
        [b_hat dev stats] = glmfit(mat_Z, Z, 'poisson', 'link', 'log', 'constant', 'off');
        
        [p_b_hat, dist_logical] = update_p_b_hat_poisson(mat_Z, Z, b_hat, threshold, 0.2565, a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id));
        
    end
    toc;
end


