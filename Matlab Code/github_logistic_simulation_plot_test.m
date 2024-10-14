% Guide

% Part I: calculate optimized TRI for each sample size
% Part II: repeat data generation and fitting to get summary table of P_e and ISE

% You only need to check these 3 variables
% summary_T, all_wot, summary_coverage_prob

% For coverage probability, you can change these 2 variables: outloopTimes, bootstraploopTimes
% outloopTimes: in total how many rounds we have. For each round, we would check whether the true zero regions are contained by LBR & UBR
% bootstraploopTimes: in each round, how many bootstrap datasets are generated and used to calculate LBR & UBR
% Both variables are 100 for continuous simulation, but using 100 for both for logistic regression is too time-consuming

% Define the irregular region
bcr27_boundary = [-88.90,-90.31,-90.10,-92.10,-83.15,-81.30,-80.90,-75.20,-76.31,-77.40,-78.1,-85.46,-87.57,-88.03;...
    37.20,35.06,33.49,30.64,29.33,30.00,31.80,35.40,38,38.42,35.909,32.61,33.35,36.67]; % First row is longitude while second row is latitude
pv = [-88.9 37.20; -90.31 35.06; -90.10 33.49; -92.10 30.64;-83.15 29.33;-81.30 30.00;-80.90 31.80;...
    -75.20 35.40;-76.31 38;-77.40 38.42;-78.1 35.909;-85.46 32.61;-87.57 33.35;-88.03 36.67; -88.9 37.20];
pgon = polyshape(bcr27_boundary(1,:), bcr27_boundary(2,:));
plot(pgon); xlabel('Longitude'); ylabel('Latitude')

% Define varying coefficient
f0 = @(x, y) -1.5  -3 .* abs( (-75-x)/17 - 0.3)  - 0.5 .* sin(1.5*pi .* (y-29)/10 ) ;
f1 = @(x,y) 2.*(exp( 2.*((-75-x)/17 -0.3) .* ((-75-x)/17>0.3) - ((-75-x)/17 -0.7) .* ((-75-x)/17>0.7) ) - 1)  ;
f2 = @(x, y) 0.*y; m=3;

% Generate grid points
grid_len = 200;
grid_s = linspace(-94, -70, grid_len + 2); grid_t = linspace(28, 40, grid_len + 2); grid_s = grid_s(2:(grid_len+1)); grid_t = grid_t(2:(grid_len+1));
[grid_S, grid_T] = meshgrid(grid_s, grid_t); grid_S = reshape(grid_S, [grid_len^2, 1]); grid_T = reshape(grid_T, [grid_len^2, 1]);
grid_idx_boolean = inpolygon(grid_S, grid_T, pv(:, 1), pv(:, 2)); grid_idx = 1:1:grid_len^2; grid_idx = grid_idx(grid_idx_boolean);
hold on; scatter(grid_S(grid_idx), grid_T(grid_idx),5, 'filled'); plot(pgon); hold off;
% Only valid points are left
grid_S = grid_S(grid_idx); grid_T = grid_T(grid_idx); 
grid_f0 = f0(grid_S, grid_T); grid_f1 = f1(grid_S, grid_T); grid_f2 = f2(grid_S, grid_T); sprintf('We have %d grid points totally', length(grid_S))
grid_f0_zeroidx = find(grid_f0 == 0); grid_f1_zeroidx = find(grid_f1 == 0); grid_f2_zeroidx = find(grid_f2 == 0); 

%% Part I: Best h value 
n_choice = [3000, 5000, 12000];
best_h_values = [2.9, 2.9, 2.9];% because h_choice(argmin) = [2.9, 2.9, 2.9];

%% Estimated Zero Region

all_b_hat = cell(length(n_choice), 1); all_p_b_hat = cell(length(n_choice), 1); 
all_grid_B = cell(length(n_choice), 1); all_grid_valid_id = cell(length(n_choice), 1); 
all_TRI = cell(length(n_choice), 1); all_vx = cell(length(n_choice), 1); all_vy = cell(length(n_choice), 1);
all_mat_Z = cell(length(n_choice), 3); all_Z = cell(length(n_choice), 3);

curr_seed=103; % other candidate seeds: 2
for i=1:length(n_choice)  % This for loop is for fitting
    n = n_choice(i); h_now = best_h_values(i);
    
    % Generate TRI first
    rng(22); [p, TRI] = distmesh2d(@dpoly, @huniform, h_now, [-94 28; -70, 40], pv, pv); vx = p(:,1); vy = p(:,2);
    
    all_TRI{i,1} = TRI; all_vx{i,1} = vx; all_vy{i,1} = vy;
    
    [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
            vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);nv = length(vx);d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt;
        
    nv = length(vx);d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt; 
    [grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);
    all_grid_B{i, 1} = grid_B; all_grid_valid_id{i, 1} = grid_valid_id;
    
    disp(['The experiment of n:', num2str(n), ' started'])
    
    rng(curr_seed);
   count = 1; S_1 = zeros(n, 1); S_2 = zeros(n, 1);
   while count <= n
       temp = [22.*rand(1,1)-94 12.*rand(1,1)+28]; 
       if inpolygon(temp(1), temp(2), pv(:,1), pv(:,2))
           S_1(count) = temp(1); S_2(count) = temp(2); count = count + 1;
       end
   end % Generate sample points' location
   X_1 = 1 .* (rand(n, 1) ); X_2 = 1 .* (rand(n, 1) ); % Set the distributions of X_1 and X_2 are same `Uniform distribution`
   Z = cal_probs(f0(S_1, S_2) + X_1 .* f1(S_1, S_2) + X_2 .* f2(S_1, S_2)); 
   Z = binornd(1, Z); ori_Z = Z;
  [B, valid_id] = CZ_SPL_est(S_1,S_2,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);  % grid_valid_id is based on left valid grid points

      mat_Z = zeros(n, m*nc);
      for k = valid_id
          temp1 = (B(k, :).* 1);temp2 = (B(k, :).* X_1(k,1));temp3 = (B(k, :).* X_2(k,1));
          temp = [temp1, temp2, temp3];mat_Z(k,:) = temp;
      end
      mat_Z = mat_Z(valid_id, :); Z = ori_Z(valid_id); 
      [b_hat dev stats] = glmfit(mat_Z, Z, 'binomial', 'link', 'logit', 'constant', 'off'); % Default iteration times is 100
      
      nlam = 14; a = 3.7;threshold = 10 ^ (-3); lam_vec = linspace(0.05, 0.7, nlam);
      % Use BIC again to detect the best lambda value
      bic = zeros(nlam, 3); converged_or_not = zeros(nlam, 1);
      for q = 1:nlam         
            [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
                mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 20, n, 1);
            converged_or_not(q) = dist_logical;
            preds = cal_probs(mat_Z * p_b_hat);
            preds_valid = setdiff(setdiff(1:1:length(valid_id), find(Z==1 & preds==1)), find(Z==0 & preds==0)) ; 
            bic(q, 1) =  -2*sum(Z(preds_valid).*log(preds(preds_valid))) - 2*sum((1-Z(preds_valid)).*log(1-preds(preds_valid))) + log(n) * sum(p_b_hat ~= 0);
            bic(q, 2) =  -2*sum(Z(preds_valid).*log(preds(preds_valid))) - 2*sum((1-Z(preds_valid)).*log(1-preds(preds_valid))) ;
            bic(q, 3) =  log(n) * sum(p_b_hat ~= 0); 
      end
      [temp_min, temp_index] = min(bic(:, 1));lam_vec(temp_index);      
      [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
        mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 20, n, 1); sum(p_b_hat == 0) / length(p_b_hat);

    all_p_b_hat{i,1} = p_b_hat;
     
end

subplot(1+length(n_choice), m, 1); hold on; plot(pgon); xlim([-92.5 -75]);ylim([29 38.5]);
title('$$\beta_0$$ zero region','interpreter','latex'); hold off;
subplot(1+length(n_choice), m, 2); hold on; plot(pgon); xlim([-92.5 -75]);ylim([29 38.5]);
title('$$\beta_1$$ zero region','interpreter','latex'); fill([-80.1 -80.1 -75.2 -76.31 -77.4 -78.1 -85.46 -80.1], [35.0125 32.3053 35.4 38 38.42 35.9090 32.61 35.0125], 'r');hold off; 
subplot(1+length(n_choice), m, 3); hold on; plot(pgon); xlim([-92.5 -75]);ylim([29 38.5]); fill(pv(:,1), pv(:,2), 'r');
title('$$\beta_2$$ zero region','interpreter','latex'); hold off;
for i=1:length(n_choice)   
    temp_TRI = all_TRI{i, 1}; temp_vx = all_vx{i, 1}; temp_vy = all_vy{i, 1};
    for j=1:m
        order = i*m+j; subplot(1+length(n_choice), m, order); hold on;    
        triplot(temp_TRI, temp_vx, temp_vy); xlim([-92.5 -75]);ylim([29 38.5]);
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
            line([-80.1 -80.1], [32 36], 'LineStyle', '--');
        end
        hold off;
    end
end
% This code chunk generates Fig - Binary Response Simulation Estimated Zero Regions Dev
% Different versions of Matlab might lead to different estimated zero regions
