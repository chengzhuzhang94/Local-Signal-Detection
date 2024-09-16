%% Numerical Work Part I
m = 3;
kLoopTime = 100; % alternative value is 300. We repeat estimation multiple times to calculate the SE of ISE & P_e

% UDF values
f = @(x,y) sqrt((x-0.5).^2 + (y-0.5).^2); fd = @(p) ddiff(drectangle(p, 0, 2, 0, 2), dcircle(p, 1, 1, 0.5)); fh=@(p) 1+4*dcircle(p,1,1,0.5);
f1 = @(x,y) -1.* sin(2.*pi./(sqrt(2)-0.5) .* (sqrt((x-1).^2+(y-1).^2)-0.5))-1.5; 
f2 = @(x,y) 2.* (exp((y-1).* (y>=1)) - 1) + 1.5 .* (sqrt((x-1).^2+(y-1).^2) -0.5).*(y>=1); 
f3 = @(x,y) 0 .* y; % This config has VAR as 4.2 

% Generate grids. They fall in [0, 2]x[0, 2] but not falling into the central circle
grid_len = 200;
grid_s = linspace(0, 2, grid_len + 2); grid_t = linspace(0, 2, grid_len + 2); grid_s = grid_s(2:(grid_len+1)); grid_t = grid_t(2:(grid_len+1));
[grid_S, grid_T] = meshgrid(grid_s, grid_t); grid_S = reshape(grid_S, [grid_len^2, 1]); grid_T = reshape(grid_T, [grid_len^2, 1]);
grid_idx = (grid_S - 1).^2 + (grid_T - 1).^2 > 0.25; 

% Only valid grid points are kept
grid_S = grid_S(grid_idx); grid_T = grid_T(grid_idx); 
grid_f1 = f1(grid_S, grid_T); grid_f2 = f2(grid_S, grid_T); grid_f3 = f3(grid_S, grid_T); 
grid_f2_zeroidx = find(grid_f2 == 0); grid_f3_zeroidx = find(grid_f3 == 0); 

record_table_ise_unpen = zeros(kLoopTime, 1*12);  
record_table_ise_scad = zeros(kLoopTime, 1*12);  
record_table_pe = zeros(kLoopTime, 1*12);  
diag_lamb_vec = zeros(kLoopTime, 1); 

%% Triangulation 1: optimized via BIC: h=0.16
rng(1);
[p,TRI] = distmesh2d(fd,fh,0.16,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
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
    X_1 = unifrnd(0, 1, [n, 1]); X_2 = unifrnd(0, 1, [n, 1]); X_3 = unifrnd(0, 1, [n, 1]);
    Z = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3;
    Z = poissrnd(Z); ori_Z = Z;
    
    Z_1 = X_1.*beta_1; Z_2 = X_2.*beta_2; Z_3 = X_3.*beta_3;
    [B, valid_id] = CZ_SPL_est(X,Y,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
   
    mat_Z = zeros(n, m*nc);
    for k = 1:n
        temp1 = (B(k, :).* X_1(k,1)); temp2 = (B(k, :).* X_2(k,1)); temp3 = (B(k, :).* X_3(k,1));
        mat_Z(k,:) = [temp1, temp2, temp3];
    end
    full_mat_Z = mat_Z; full_Z = Z;
    mat_Z = mat_Z(valid_id, :); Z = ori_Z(valid_id, :);
    [b_hat dev stats] = glmfit(mat_Z, Z, 'poisson', 'constant', 'off'); % Default iteration times is 100
    n = length(valid_id);
       
    nlam = 40; a = 3.7;threshold = 10 ^ (-3); 
    lam_vec = linspace(0.01, 0.4, nlam);
    bic = zeros(nlam, 1); converged_or_not = zeros(nlam, 1);
    for q = 1:nlam
        %[p_b_hat, dist_logical] = update_p_b_hat(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id));
        [p_b_hat, dist_logical] = update_p_b_hat_poisson(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id));      
        converged_or_not(q) = dist_logical;
        exp_preds = mat_Z * p_b_hat;
        bic(q) = -2*sum(Z .* exp_preds) + 2*sum(exp(exp_preds)) + log(length(exp_preds)) * sum(p_b_hat ~= 0);
    end

    [temp_min, temp_index] = min(bic);
    [p_b_hat, dist_logical] = update_p_b_hat_poisson(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id));      

    color_p_beta = ones(length(p_b_hat), 1); color_p_beta(p_b_hat == 0) = 0;
    
    % f1 P_e part
    record_table_pe(i, 1) = sum(grid_B(grid_valid_id,:) * p_b_hat(1:nc) == 0) / length(grid_valid_id);
    grid_f2_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+nc:2*nc) == 0);
    record_table_pe(i, 2) = (length(intersect(grid_f2_zeroidx, grid_valid_id)) + length(grid_f2_zeroidx_pred) - 2*length(intersect(grid_f2_zeroidx, grid_f2_zeroidx_pred))) / length(grid_valid_id);
    grid_f3_zeroidx_pred = find(grid_B(grid_valid_id,:) * p_b_hat(1+2*nc:3*nc) == 0);
    record_table_pe(i, 3) = (length(intersect(grid_f3_zeroidx, grid_valid_id)) + length(grid_f3_zeroidx_pred) - 2*length(intersect(grid_f3_zeroidx, grid_f3_zeroidx_pred))) / length(grid_valid_id); % P_e part is done
    
end

triplot(TRI, vx, vy)

%% Best h value for each n
h_opt = 0.13:0.01:0.23;
for j = 1:length(h_opt)
    rng(1100); % Set seed to make sure the generated triangulations are identical
    h_now = h_opt(j); 
    disp(h_now)
    [p,TRI] = distmesh2d(fd, fh, h_now, [0,0;2,2], [0,0;0,2;2,0;2,2]);
    % You need to output p17 and TRI17 manually for later use
    if abs(h_now - 0.17) < 0.001
        p17 = p; TRI17 = TRI;
    end       
end

% h_opt = 0.24:0.01:0.26;
% for j = 1:length(h_opt)
%     rng(11); % Set seed to make sure the generated triangulations are identical
%     h_now = h_opt(j); 
%     disp(h_now)
%     [p,TRI] = distmesh2d(fd, fh, h_now, [0,0;2,2], [0,0;0,2;2,0;2,2]);
%     % You need to output p17 and TRI17 manually for later use
%     if abs(h_now - 0.24) < 0.001
%         p24 = p; TRI24 = TRI;
%     end
% end
rng(1100);
[p,TRI] = distmesh2d(fd, fh, 0.19, [0,0;2,2], [0,0;0,2;2,0;2,2]);


n_choice = [500, 2000, 5000]; 
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
    %Z = poissrnd(Z); ori_Z = Z; % store the original response values in ori_Z
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
% best h_choice for n=[500, 2000, 5000] is [0.21, 0.21, 0.21] with rng(11)

%% Exp for each sample size
tic; 
kLoopTime = 100;
record_table = zeros(kLoopTime, 3*9);  diag_lamb_vec = zeros(kLoopTime, 3); 
n_choice = [500, 2000, 5000]; best_h_values = [0.21, 0.21, 0.21];
rng(1111);
for i=1:length(n_choice)
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
        %scatter(X, Y, 6, X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3); colorbar(); % we want sum between [-3, 3]
        
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

toc;
% Display the mean and median
str = zeros(27,1); str = string(str);
for i = 1:27
   str(i) = sprintf('%.4f (%.4f)', mean(record_table(:, i)), std(record_table(:, i))) ;
end
summary_T = [string('n') string('F') string('P_e') string('ISE_SCAD') string('ISE_UNPEN'); '500' 'F1' str(1) str(4) str(7);'2000' 'F1' str(2) str(5) str(8);'5000' 'F1' str(3) str(6) str(9);...
    '500' 'F2' str(10) str(13) str(16);'2000' 'F2' str(11) str(14) str(17);'5000' 'F2' str(12) str(15) str(18);...
    '500' 'F3' str(19) str(22) str(25);'2000' 'F3' str(20) str(23) str(26);'5000' 'F3' str(21) str(24) str(27);];
summary_T


% Trial Result: rng(1111)
%     "500"     "F1"    "0.2333 (0.1493)"    "11.0099 (56.3582)"    "10.0418 (56.5483)"
%     "2000"    "F1"    "0.0000 (0.0000)"    "0.4243 (0.1216)"      "0.4895 (0.1559)"  
%     "5000"    "F1"    "0.0000 (0.0000)"    "0.2299 (0.0538)"      "0.2612 (0.0709)"  
%     "500"     "F2"    "0.3028 (0.0982)"    "2.3800 (2.9808)"      "2.7817 (2.3869)"  
%     "2000"    "F2"    "0.2393 (0.0741)"    "0.1555 (0.0904)"      "0.2564 (0.1071)"  
%     "5000"    "F2"    "0.2648 (0.0769)"    "0.0658 (0.0388)"      "0.1003 (0.0400)"  
%     "500"     "F3"    "0.2164 (0.1596)"    "1.5931 (2.7418)"      "2.7509 (2.8164)"  
%     "2000"    "F3"    "0.1126 (0.1184)"    "0.0761 (0.0943)"      "0.2824 (0.1185)"  
%     "5000"    "F3"    "0.0887 (0.1127)"    "0.0163 (0.0257)"      "0.0965 (0.0377)"  

%% 100 times
%     "n"       "F"     "P_e"                    "ISE_SCAD"               "ISE_UNPEN"          
%     "500"     "F1"    "0.399581 (0.377960)"    "4.798028 (4.202186)"    "3.595668 (2.944468)"
%     "2000"    "F1"    "0.013795 (0.000000)"    "1.570517 (1.536132)"    "1.447050 (1.419420)"
%     "5000"    "F1"    "0.000000 (0.000000)"    "1.304560 (1.308220)"    "1.267051 (1.275894)"
%     "500"     "F2"    "0.296311 (0.291476)"    "1.870801 (1.084621)"    "2.366627 (1.569252)"
%     "2000"    "F2"    "0.234619 (0.211445)"    "0.159694 (0.134798)"    "0.227149 (0.211877)"
%     "5000"    "F2"    "0.258221 (0.240342)"    "0.059559 (0.048304)"    "0.088944 (0.079264)"
%     "500"     "F3"    "0.204090 (0.180084)"    "1.074271 (0.691531)"    "1.974013 (1.557106)"
%     "2000"    "F3"    "0.129807 (0.142346)"    "0.090819 (0.064122)"    "0.261194 (0.227255)"
%     "5000"    "F3"    "0.123663 (0.132829)"    "0.023009 (0.004708)"    "0.088834 (0.079394)"

%%
% @(x,y) -1.*sin(2.*pi./(sqrt(2)-0.5).*(sqrt((x-1).^2+(y-1).^2)-0.5))-1.5
% @(x,y) 2.*(exp((y-1).*(y>=1))-1)+1.5.*(sqrt((x-1).^2+(y-1).^2)-0.5).*(y>=1)
    "n"       "F"     "P_e"                    "ISE_SCAD"                "ISE_UNPEN"           
    "500"     "F1"    "0.233340 (0.222768)"    "11.009949 (3.339974)"    "10.041766 (2.807348)"
    "2000"    "F1"    "0.000000 (0.000000)"    "0.424320 (0.393193)"     "0.489473 (0.467221)" 
    "5000"    "F1"    "0.000000 (0.000000)"    "0.229904 (0.224521)"     "0.261213 (0.245669)" 
    "500"     "F2"    "0.302766 (0.296909)"    "2.380049 (1.415334)"     "2.781736 (2.087119)" 
    "2000"    "F2"    "0.239288 (0.213361)"    "0.155537 (0.143146)"     "0.256362 (0.241418)" 
    "5000"    "F2"    "0.264755 (0.247315)"    "0.065760 (0.057702)"     "0.100327 (0.090269)" 
    "500"     "F3"    "0.216369 (0.200719)"    "1.593077 (0.884363)"     "2.750890 (2.023416)" 
    "2000"    "F3"    "0.112593 (0.136692)"    "0.076077 (0.033901)"     "0.282362 (0.247260)" 
    "5000"    "F3"    "0.088737 (0.030058)"    "0.016345 (0.000000)"     "0.096469 (0.091838)" 


    
% Prepare for all plots and MCR diagrams %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_choice = [500, 1000, 2000, 5000]; best_h_values = [0.21, 0.21, 0.19, 0.17];

all_b_hat = cell(length(n_choice), 1); all_p_b_hat = cell(length(n_choice), 1); 
all_grid_B = cell(length(n_choice), 1); all_grid_valid_id = cell(length(n_choice), 1); 
all_TRI = cell(length(n_choice), 1); all_vx = cell(length(n_choice), 1); all_vy = cell(length(n_choice), 1);
all_LBR = cell(length(n_choice), 3); all_UBR = cell(length(n_choice), 3);

tic;
for i=1:length(n_choice)  % This for loop is for fitting
    n = n_choice(i); n_buffer = n+300;
    rng(1100) ;h_now = best_h_values(i);  
    if (h_now == 0.17) 
        p = p17; TRI = TRI17; vx = p(:,1); vy = p(:,2);
    else
        [p,TRI] = distmesh2d(fd,fh,h_now,[0,0;2,2],[0,0;0,2;2,0;2,2]); vx = p(:,1); vy = p(:,2);
    end
    [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
        vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
    nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt; [grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
    
    rng(i+99);
    all_TRI{i,1} = TRI; all_vx{i,1} = vx; all_vy{i,1} = vy;
    
    [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
    nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt; [grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
    all_grid_B{i, 1} = grid_B; all_grid_valid_id{i, 1} = grid_valid_id;
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
    all_b_hat{i,1} = b_hat;
    
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

    [temp_min, temp_index] = min(bic(:, 1)); diag_lamb_vec(j,i) = lam_vec(temp_index);
    [p_b_hat, dist_logical] = update_p_b_hat_poisson(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(valid_id));
    all_p_b_hat{i,1} = p_b_hat;
    
%     records =  CZ_bootstrap(2, TRI, mat_Z, Z, n, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, 100, 1, 1, 3.7, 0);
%     MCB_records = cell(size(records,1)+1, 2*m); CR_records = zeros(size(records,2)/m+1, m);
%     for j = 1:m
%        TRI_no = 1+(j-1)*nt:j*nt; [out, pi_idx] = sort(sum(records(:, TRI_no), 1), 'descend');
%        for w = 0:1:nt
%           cr = 0; LBM = []; UBM = [];
%           for k = 0:1:nt-w
%              temp_LBM = pi_idx(1:k); temp_UBM = sort(pi_idx(1:k+w));
%              temp_cr = CZ_CoverageRate(records(:, TRI_no), temp_LBM, temp_UBM);
%              if temp_cr > cr
%                 cr = temp_cr; LBM = temp_LBM; UBM = temp_UBM; 
%              end
%           end
%           MCB_records{w+1, 2*j-1} = sort(LBM); MCB_records{w+1, 2*j} = sort(UBM); CR_records(w+1, j) = cr;
%        end
%     end
%     all_LBR{i, 1} = sort(MCB_records{find(CR_records(:,1) > 0.95,1), 2*1-1}) ; all_UBR{i, 1} = sort(MCB_records{find(CR_records(:,1) > 0.95,1), 2*1});
%     all_LBR{i, 2} = sort(MCB_records{find(CR_records(:,2) > 0.95,1), 2*2-1}) ; all_UBR{i, 2} = sort(MCB_records{find(CR_records(:,2) > 0.95,1), 2*2});
%     all_LBR{i, 3} = sort(MCB_records{find(CR_records(:,3) > 0.95,1), 2*3-1}) ; all_UBR{i, 3} = sort(MCB_records{find(CR_records(:,3) > 0.95,1), 2*3});
end
toc;

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

    
%% Time check
% This section is used to get running time. We use matlab function "tic"
% and "toc" to get lasping time

% UNPEN

kLoopTime = 10;
record_table = zeros(kLoopTime, 3*9);  diag_lamb_vec = zeros(kLoopTime, 3); 
n_choice = [500, 2000, 5000]; best_h_values = [0.21, 0.21, 0.21];
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
kLoopTime = 10;
record_table = zeros(kLoopTime, 3*9);  diag_lamb_vec = zeros(kLoopTime, 3); 
n_choice = [500, 2000, 5000]; best_h_values = [0.21, 0.21, 0.21];
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
        %scatter(X, Y, 6, X_1.*beta_1 + X_2.*beta_2 + X_3.*beta_3); colorbar(); % we want sum between [-3, 3]
        
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


