%% Load CSV file
df = table2array(readtable('new BJ house.csv')); df(1,:);
scatter(df(:,3),df(:,4),1)
% This dataset contains 293,963 data points and 28 columns

%% Use a fixed Triangulation
TRI = [24    29    28    25    24    25    13  14      2   2     26    24   19  5   19   5     14  7 14  7       4    21    16        14    16      8  2  13  6  13   6  18    22        22    11       13  4     
       27    30    27    27    20    20     8   8      8   9     23    28   20  20   5  24     16 16  7 20      11    20    10         9     9     13 10  14 14   6  19  17    19        17     4       17  8    
       28    28    29    24    25    21    14   9      1   8     24    26    5  24  23  23      7 20 19 19       3    16    15        16    10     12  9   6 19  22  22  22    23        13    12       12 12 ]' ;
vx = [116.1000  116.1613  116.2000  116.2127  116.4818 116.3903 116.3989  116.2500  116.2772  116.2892  116.3000  116.3200  116.3474  116.3544  116.3600  116.3600  116.4000  116.4000  116.4333 ...
    116.4434  116.4459  116.4605  116.5000  116.5303  116.5319  116.5740  116.6200  116.6419  116.7100  116.7100]' ;
vy = [39.9400   39.8834   40.2400   40.1712   39.9352  39.9855  39.8850  39.9700   39.8824   39.7654   40.2400   40.1000   40.0085   39.9105   39.7000   39.8050   40.1000   40.1800   39.9625 ...
    39.8595   39.7545   40.0771   40.0100   39.9080   39.8091   39.9924   39.8650   39.9762   39.8700   39.9600]' ; 

[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);
nv = length(vx); d = 1; nc = nv + (d-1)*ne + choose(d-1,2)*nt;

% Data Visualization of all data points and the TRI
hold on; scatter(df(:, 3), df(:, 4), 1); triplot(TRI, vx, vy); 
scatter(vx, vy);
for i = 1:nc
   text(vx(i), vy(i), sprintf('%d',i)) ;
end
for i = 1:37
   text((vx(TRI(i,1))+vx(TRI(i,2))+vx(TRI(i,3)))/3, (vy(TRI(i,1))+vy(TRI(i,2))+vy(TRI(i,3)))/3, sprintf('%d',i),'Color', 'red') 
end
hold off;

% Downsize the data, only valid points falling into TRI are kept for model
% fitting
[B, valid_id] = CZ_SPL_est(df(:, 3),df(:, 4),vx,vy,TRI,v1,v2,v3,nt,nc,nv,d); 
B = B(valid_id, :); df = df(valid_id,:); 
% only 277,410 out of 293,963 (94.4%) data points are kept

rng(1); % set seed to randomize all data points
train_id = sort(randsample(length(valid_id), 225000)); test_id = setdiff(1:length(valid_id), train_id) ;
train_B = B(train_id,:); test_B = B(test_id, :);
totalPrice_log_std = (log(df(:,1))-mean(log(df(:,1))))./std(log(df(:,1)));

grid_len = 300;
grid_s = linspace(116.1, 116.75, grid_len + 2); grid_t = linspace(39.7,40.3, grid_len + 2); grid_s = grid_s(2:(grid_len+1)); grid_t = grid_t(2:(grid_len+1));
[grid_S, grid_T] = meshgrid(grid_s, grid_t); grid_S = reshape(grid_S, [grid_len^2, 1]); grid_T = reshape(grid_T, [grid_len^2, 1]);
grid_idx_boolean = inpolygon(grid_S, grid_T, pv(:, 1), pv(:, 2)); grid_idx = 1:1:grid_len^2; grid_idx = grid_idx(grid_idx_boolean);
grid_S = grid_S(grid_idx); grid_T = grid_T(grid_idx); 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

% Read in the standadized dataset
stepAIC_DM_std = readtable('new BJ stepAIC design matrix std.csv'); 
stepAIC_DM_std = table2array(stepAIC_DM_std); 
size(stepAIC_DM_std) 
stepAIC_DM_std = stepAIC_DM_std(valid_id,:);

cols = 1:length(stepAIC_DM_std(1,:));
m = length(cols); n = length(train_id);
mat_Z = zeros(length(train_id), m*nc);
for k = 1:length(train_id)
   for j = 1:length(cols)
      mat_Z(k, (j-1)*nc+1:(j)*nc) = train_B(k, :).*stepAIC_DM_std(train_id(k), cols(j)); 
   end
end
test_mat_Z = zeros(length(test_id), m*nc);
for k = 1:length(test_id)
   for j = 1:length(cols)
      test_mat_Z(k, (j-1)*nc+1:(j)*nc) = test_B(k, :).*stepAIC_DM_std(test_id(k), cols(j)); 
   end
end
mat_D = stepAIC_DM_std(train_id,:); 
test_mat_D = stepAIC_DM_std(test_id,:);

%% Method: Linear Regression
lm_b_hat = (transpose(mat_D) * mat_D) \ transpose(mat_D) * totalPrice_log_std(train_id); 
b_hat = (transpose(mat_Z) * mat_Z + 1 / (log(length(train_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * totalPrice_log_std(train_id); 

%% Cross-validation
n_total = (length(train_id)+length(test_id)) % the value is 277,410
full_mat_Z = zeros(n_total, m*nc);
for k = 1:n_total
   for j = 1:length(cols)
      full_mat_Z(k, (j-1)*nc+1:(j)*nc) = B(k, :).*stepAIC_DM_std(k, j); 
   end
end
full_mat_D = stepAIC_DM_std; 

rng(100);
reorder = datasample(1:n_total, n_total, 'Replace', false);
% Do a 5-fold CV
cv_records = zeros(5, 15);
for i=1:5
    % extract the test ID for current loop
    temp_test_id = reorder((i-1)*n_total/5+1:i*n_total/5);
    temp_train_id = setdiff(1:n_total, temp_test_id);
   
    % Method: Linear Regression
    temp_lm_b_hat = (transpose(full_mat_D(temp_train_id,:)) * full_mat_D(temp_train_id,:)) \ transpose(full_mat_D(temp_train_id,:)) * totalPrice_log_std(temp_train_id); 
    temp_b_hat = (transpose(full_mat_Z(temp_train_id,:)) * full_mat_Z(temp_train_id,:) + 1 / (log(length(temp_train_id))*nt) * eye(49*nc)) \ transpose(full_mat_Z(temp_train_id,:)) * totalPrice_log_std(temp_train_id); 
   
    % Method: Decision Tree
    temp_dt_model = fitrtree(df(temp_train_id, 3:27), totalPrice_log_std(temp_train_id), 'MaxNumSplits', 20000);
    temp_dt_train_predict = predict(temp_dt_model, df(temp_train_id, 3:27));
    temp_dt_test_predict = predict(temp_dt_model, df(temp_test_id, 3:27));
    
    % Method: LR
    cv_records(i, 1) = mean((totalPrice_log_std(temp_train_id)-full_mat_D(temp_train_id,:) * temp_lm_b_hat).^2);
    cv_records(i, 2) = mean((totalPrice_log_std(temp_test_id)-full_mat_D(temp_test_id,:) * temp_lm_b_hat).^2);
    cv_records(i, 3) = 1 - sum((totalPrice_log_std(temp_train_id)-full_mat_D(temp_train_id,:) * temp_lm_b_hat).^2)/sum((totalPrice_log_std(temp_train_id)-mean(totalPrice_log_std(temp_train_id))).^2);
    
    % Method: UNPEN
    cv_records(i, 4) = mean((totalPrice_log_std(temp_train_id)-full_mat_Z(temp_train_id,:) * temp_b_hat).^2);
    cv_records(i, 5) = mean((totalPrice_log_std(temp_test_id)-full_mat_Z(temp_test_id,:) * temp_b_hat).^2);
    cv_records(i, 6) = 1 - sum((totalPrice_log_std(temp_train_id)-full_mat_Z(temp_train_id,:) * temp_b_hat).^2)/sum((totalPrice_log_std(temp_train_id)-mean(totalPrice_log_std(temp_train_id))).^2);
    
    nlam = 10; a = 3.7;threshold = 10 ^ (-3); 
    lam_vec = linspace(0.01, 0.19, nlam);
    bic = zeros(nlam, 3); converged_or_not = zeros(1,nlam);
    for q = 1:nlam
        [temp_p_b_hat, dist_logical] = update_p_b_hat_2(full_mat_Z(temp_train_id,:), totalPrice_log_std(temp_train_id), temp_b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(train_id),2,6);
        converged_or_not(q) = dist_logical;
        bic(q,1) = log(mean((totalPrice_log_std(temp_train_id) - full_mat_Z(temp_train_id, :) * temp_p_b_hat).^2)) + log(length(temp_train_id)) * sum(temp_p_b_hat ~=0) / length(temp_train_id);
        bic(q,2) = log(mean((totalPrice_log_std(temp_train_id) - full_mat_Z(temp_train_id, :) * temp_p_b_hat).^2));
        bic(q,3) = sum(temp_p_b_hat ==0);
    end
    [temp_min, temp_index] = min(bic(:,1)); lam_vec(temp_index)
    [temp_p_b_hat, dist_logical] = update_p_b_hat_2(full_mat_Z(temp_train_id,:), totalPrice_log_std(temp_train_id), temp_b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(train_id),2,6); sum(p_b_hat ~=0)/length(p_b_hat)
    
    % SCAD
    cv_records(i, 7) = mean((totalPrice_log_std(temp_train_id)-full_mat_Z(temp_train_id,:) * temp_p_b_hat).^2);
    cv_records(i, 8) = mean((totalPrice_log_std(temp_test_id)-full_mat_Z(temp_test_id,:) * temp_p_b_hat).^2);
    cv_records(i, 9) = 1 - sum((totalPrice_log_std(temp_train_id)-full_mat_Z(temp_train_id,:) * temp_p_b_hat).^2)/sum((totalPrice_log_std(temp_train_id)-mean(totalPrice_log_std(temp_train_id))).^2);
 
    % DT
    cv_records(i, 10) = mean((totalPrice_log_std(temp_train_id)-temp_dt_train_predict).^2);
    cv_records(i, 11) = mean((totalPrice_log_std(temp_test_id)-temp_dt_test_predict).^2);
    cv_records(i, 12) = 1 - sum((totalPrice_log_std(temp_train_id)-temp_dt_train_predict).^2)/sum((totalPrice_log_std(temp_train_id)-mean(totalPrice_log_std(temp_train_id))).^2);

    % BIC
    cv_records(i, 13) = log(mean((totalPrice_log_std(temp_train_id) - full_mat_D(temp_train_id,:) * temp_lm_b_hat).^2)) + log(length(temp_train_id)) * sum(temp_lm_b_hat ~=0) / length(temp_train_id);
    cv_records(i, 14) = log(mean((totalPrice_log_std(temp_train_id) - full_mat_Z(temp_train_id,:) * temp_b_hat).^2)) + log(length(temp_train_id)) * sum(temp_b_hat ~=0) / length(temp_train_id);
    cv_records(i, 15) = log(mean((totalPrice_log_std(temp_train_id) - full_mat_Z(temp_train_id,:) * temp_p_b_hat).^2)) + log(length(temp_train_id)) * sum(temp_p_b_hat ~=0) / length(temp_train_id);
    % no BIC formula for Decision Tree
end

for i=1:5
    % extract the test ID for current loop
    temp_test_id = reorder((i-1)*n_total/5+1:i*n_total/5); %temp_test_id = sort(temp_test_id); 
    temp_train_id = setdiff(1:n_total, temp_test_id);
   
    % Decision Tree
    temp_dt_model = fitrtree(df(temp_train_id, 3:27), totalPrice_log_std(temp_train_id), 'MaxNumSplits', 20000);
    temp_dt_train_predict = predict(temp_dt_model, df(temp_train_id, 3:27));
    temp_dt_test_predict = predict(temp_dt_model, df(temp_test_id, 3:27));
    
    % DT
    cv_records(i, 10) = mean((totalPrice_log_std(temp_train_id)-temp_dt_train_predict).^2);
    cv_records(i, 11) = mean((totalPrice_log_std(temp_test_id)-temp_dt_test_predict).^2);
    cv_records(i, 12) = 1 - sum((totalPrice_log_std(temp_train_id)-temp_dt_train_predict).^2)/sum((totalPrice_log_std(temp_train_id)-mean(totalPrice_log_std(temp_train_id))).^2);
    
    % no BIC formula for Decision Tree
end

% Display the mean and median
str = zeros(15,1); str = string(str);
for i = 1:15
   str(i) = sprintf('%f (%f)', mean(cv_records(:, i)), median(cv_records(:, i))) ;
end
summary_T = [string('Method') string('R^2') string('MSEE') string('MSPE') string('BIC');...
    'Linear Reg' str(3) str(1) str(2) str(13);...
    'UNPEN' str(6) str(4) str(5) str(14);...
    'SCAD' str(9) str(7) str(8) str(15);...
    'DT' str(12) str(10) str(11) string('NA')];
summary_T