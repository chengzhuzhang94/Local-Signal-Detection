%% Guide

% Part I: Triangulation
% Part II: read in dataset
% Part III: Cross-Validation
% Part IV: District-wise Linear Regression
% Part V: Visualization of Triangulation and Estimated Varying Coefficient

%% Load CSV file. This one is only for visualization
df = table2array(readtable('new BJ house.csv')); df(1,:);
scatter(df(:,3),df(:,4),1)
% This dataset contains 293,963 data points and 28 columns

%% Part I: Triangulation

% This Triangulation is picked based on the border of dataset and relative
% locations of each Beijing political districts
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

disp("Triangulation was generated")

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

tic;
% Downsize the data, only valid points falling into TRI are kept for model fitting
[B, valid_id] = CZ_SPL_est(df(:, 3),df(:, 4),vx,vy,TRI,v1,v2,v3,nt,nc,nv,d); 
B = B(valid_id, :); df = df(valid_id,:); 
% only 277,410 out of 293,963 (94.4%) data points are kept

rng(1); % set seed. Randomly split all data points into train and test
train_id = sort(randsample(length(valid_id), 225000)); test_id = setdiff(1:length(valid_id), train_id) ;
train_B = B(train_id,:); test_B = B(test_id, :);
totalPrice_log_std = (log(df(:,1))-mean(log(df(:,1))))./std(log(df(:,1)));

%% Part II: read in dataset
disp("Start to read in BJ housing price dataset...")
% Read in the standadized dataset. This one is used for fitting model
stepAIC_DM = readtable('new BJ stepAIC design matrix.csv'); stepAIC_col_names = stepAIC_DM.Properties.VariableNames; stepAIC_col_names(1)={'x_Intercept'};
stepAIC_DM = table2array(stepAIC_DM); size(stepAIC_DM); stepAIC_DM = stepAIC_DM(valid_id,:);
% generate design matrix
cols = 1:length(stepAIC_DM(1,:));
m = length(cols); n = length(train_id);
mat_Z = zeros(length(train_id), m*nc);
for k = 1:length(train_id)
   for j = 1:length(cols)
       % multiply value of Bernstein basis polynomial with covariate values
      mat_Z(k, (j-1)*nc+1:(j)*nc) = train_B(k, :).*stepAIC_DM(train_id(k), cols(j)); 
   end
end
test_mat_Z = zeros(length(test_id), m*nc);
for k = 1:length(test_id)
   for j = 1:length(cols)
      test_mat_Z(k, (j-1)*nc+1:(j)*nc) = test_B(k, :).*stepAIC_DM(test_id(k), cols(j)); 
   end
end
mat_D = stepAIC_DM(train_id,:); 
test_mat_D = stepAIC_DM(test_id,:);

disp("Design Matrix was generated")
%% Method: Linear Regression
lm_b_hat = (transpose(mat_D) * mat_D) \ transpose(mat_D) * totalPrice_log_std(train_id); 
b_hat = (transpose(mat_Z) * mat_Z + 1 / (log(length(train_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * totalPrice_log_std(train_id); 

%% Part III: Cross-Validation
disp("Start Cross-Validation...")
% In this section, we implement cross-validation fitting to generate
% summary table result
n_total = (length(train_id)+length(test_id)); % the value is 277,410
full_mat_Z = zeros(n_total, m*nc);
for k = 1:n_total
   for j = 1:length(cols)
      full_mat_Z(k, (j-1)*nc+1:(j)*nc) = B(k, :).*stepAIC_DM(k, j); 
   end
end
full_mat_D = stepAIC_DM; 

rng(100); % set seed
reorder = datasample(1:n_total, n_total, 'Replace', false);
% Do a 5-fold CV
% cv_records: the variable stores metric values for each method of each
% fold
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
    lam_vec = linspace(0.01, 0.1, nlam);
    bic = zeros(nlam, 3); converged_or_not = zeros(1,nlam);
    for q = 1:nlam
        [temp_p_b_hat, dist_logical] = update_p_b_hat_2(full_mat_Z(temp_train_id,:), totalPrice_log_std(temp_train_id), temp_b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(train_id),2,6);
        converged_or_not(q) = dist_logical;
        bic(q,1) = log(mean((totalPrice_log_std(temp_train_id) - full_mat_Z(temp_train_id, :) * temp_p_b_hat).^2)) + log(length(temp_train_id)) * sum(temp_p_b_hat ~=0) / length(temp_train_id);
        bic(q,2) = log(mean((totalPrice_log_std(temp_train_id) - full_mat_Z(temp_train_id, :) * temp_p_b_hat).^2));
        bic(q,3) = sum(temp_p_b_hat ==0);
    end
    [temp_min, temp_index] = min(bic(:,1)); lam_vec(temp_index);
    [temp_p_b_hat, dist_logical] = update_p_b_hat_2(full_mat_Z(temp_train_id,:), totalPrice_log_std(temp_train_id), temp_b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(train_id),2,6);
    
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

toc; % running time: 6534 seconds

% Display the mean and median of each method
str = zeros(15,1); str = string(str);
for i = 1:15
   str(i) = sprintf('%.4f (%.4f)', mean(cv_records(:, i)), median(cv_records(:, i))) ;
end
summary_T = [string('Method') string('R^2') string('MSEE') string('MSPE') string('BIC');...
    'Linear Reg' str(3) str(1) str(2) str(13);...
    'UNPEN' str(6) str(4) str(5) str(14);...
    'SCAD' str(9) str(7) str(8) str(15);...
    'DT' str(12) str(10) str(11) string('NA')];
summary_T
%     "Method"        "R^2"                "MSEE"               "MSPE"               "BIC"              
%     "Linear Reg"    "0.6256 (0.6251)"    "0.3744 (0.3743)"    "0.3747 (0.3754)"    "-0.9796 (-0.9801)"
%     "UNPEN"         "0.8531 (0.8529)"    "0.1469 (0.1468)"    "0.1497 (0.1496)"    "-1.8373 (-1.8376)"
%     "SCAD"          "0.8518 (0.8516)"    "0.1482 (0.1480)"    "0.1505 (0.1504)"    "-1.8532 (-1.8544)"
%     "DT"            "0.9484 (0.9483)"    "0.0516 (0.0517)"    "0.1117 (0.1115)"    "NA"   

%% Part IV: District-wise Linear Regression
% Plot TRI and Beijing Districts
disp("Start District-wise Linear Regression...")

hold on; triplot(TRI, vx, vy); fill(vx(TRI(20,:)), vy(TRI(20,:)), [0.5 0.5 0.5], vx(TRI(19,:)), vy(TRI(19,:)), 'm');
for i=[2,3]
   fill(vx(TRI(i,:)), vy(TRI(i,:)), 'c');
end
for i=[1,4,5,11,12,13,14,15,16,30,31,33]
   fill(vx(TRI(i,:)), vy(TRI(i,:)), 'r');
end
for i=[6,22,23]
   fill(vx(TRI(i,:)), vy(TRI(i,:)), 'g');
end
for i=[17,18,24,25,27]
   fill(vx(TRI(i,:)), vy(TRI(i,:)), [1 0.5 0]);
end
for i=[8,9,10]
   fill(vx(TRI(i,:)), vy(TRI(i,:)), 'k');
end
for i=[37,26,7,28,29]
   fill(vx(TRI(i,:)), vy(TRI(i,:)), 'y');
end
for i=[21,35,36,34,32]
   fill(vx(TRI(i,:)), vy(TRI(i,:)), [0 0 0.5]);
end
% This generates Fig - BJ TRI and Districts

tic;

mat_D = stepAIC_DM(train_id,:); 
train_mat_D = mat_D; train_mat_Z = mat_Z; train_responses = totalPrice_log_std(train_id); temp_df = df(train_id,:);
[tnum, b] = tsearchn([vx, vy], TRI, [temp_df(:,3), temp_df(:,4)]);

local_tri_no = [19]; % District Xi Cheng with initial XC
local_mat_D = train_mat_D(ismember(tnum, local_tri_no), :); size(local_mat_D)
lm = fitlm(local_mat_D, train_responses(ismember(tnum, local_tri_no)), 'Intercept', false); lm.Coefficients([3,4,5,7,8],:)

local_tri_no = [20]; % District Dong Cheng with initial DC
local_mat_D = train_mat_D(ismember(tnum, local_tri_no), :); size(local_mat_D)
lm = fitlm(local_mat_D, train_responses(ismember(tnum, local_tri_no)), 'Intercept', false); lm.Coefficients([3,4,5,7,8],:)

local_tri_no = [21, 35, 36, 34, 32]; % District Chang Ping with initial CP
local_mat_D = train_mat_D(ismember(tnum, local_tri_no), :); size(local_mat_D)
lm = fitlm(local_mat_D, train_responses(ismember(tnum, local_tri_no)), 'Intercept', false); lm.Coefficients([3,4,5,7,8],:)

local_tri_no = [2, 3]; % District Tong Zhou with initial TZ
local_mat_D = train_mat_D(ismember(tnum, local_tri_no), :); size(local_mat_D)
lm = fitlm(local_mat_D, train_responses(ismember(tnum, local_tri_no)), 'Intercept', false); lm.Coefficients([3,4,5,7,8],:)

local_tri_no = [1,4,5,11,12,13,14,15,16,30,31,33]; % District Chao Yang with initial CY
local_mat_D = train_mat_D(ismember(tnum, local_tri_no), :); size(local_mat_D)
lm = fitlm(local_mat_D, train_responses(ismember(tnum, local_tri_no)), 'Intercept', false); lm.Coefficients([3,4,5,7,8],:)

local_tri_no = [6,22,23]; % District Da Xing with initial DX
local_mat_D = train_mat_D(ismember(tnum, local_tri_no), :); size(local_mat_D)
lm = fitlm(local_mat_D, train_responses(ismember(tnum, local_tri_no)), 'Intercept', false); lm.Coefficients([3,4,5,7,8],:)

local_tri_no = [17,18,24,25,27]; % District Feng Tai with initial FT
local_mat_D = train_mat_D(ismember(tnum, local_tri_no), :); size(local_mat_D)
lm = fitlm(local_mat_D, train_responses(ismember(tnum, local_tri_no)), 'Intercept', false); lm.Coefficients([3,4,5,7,8],:)

local_tri_no = [8,9,10]; % District Shi Jing Shan with initial SJS
local_mat_D = train_mat_D(ismember(tnum, local_tri_no), :); size(local_mat_D)
lm = fitlm(local_mat_D, train_responses(ismember(tnum, local_tri_no)), 'Intercept', false); lm.Coefficients([3,4,5,7,8],:)

local_tri_no = [37,26,7,28,29]; % District Hai Dian with initial HD
local_mat_D = train_mat_D(ismember(tnum, local_tri_no), :); size(local_mat_D)
lm = fitlm(local_mat_D, train_responses(ismember(tnum, local_tri_no)), 'Intercept', false); lm.Coefficients([3,4,5,7,8],:)
toc; % running time was 7 seconds

% We pick several linear terms including totalRooms (x3), tradeTime (x4), followers (x5), elevator (x7), subway (x8)
% Only 

%             Xi Cheng       Dong Cheng     Tong Zhou   Chao Yang   Da Xing   Feng Tai    Shi Jing Shan  Hai Dian     Chang Ping
%totalRooms   -0.2947          -0.0705      0.2365       0.20273    0.10198    0.22798      0.12491      -0.035393     0.22993
%p-value      0.075147         0.67144     3.3805e-10   8.061e-12   0.0037     0.00013      0.072655     0.44938       3.82e-12

%tradeTime     0.6254           0.8984      1.0486       0.75079    0.85088    0.74056      0.60738      0.5765        0.635
%p-value     1.2606e-19       9.6126e-58      0             0          0          0             0           0             0

%followers     0.0156          -0.0280     0.073765      -0.0096    0.029279   0.002903     0.084103     0.040363      0.018487
%p-value      0.82411          0.56616     6.3979e-06    0.11493    0.015879    0.7934      1.106e-06    0.00106       0.049754

%elevator      0.8056          -0.0932      0.26514      0.35908    0.39166    0.17095      0.14861      0.50176       0.4648
%p-value      0.00021          0.51912     3.4855e-05   8.259e-26   1.266e-12  0.00496      0.051766     3.347e-22     1.4438e-20

%subway        0.4266          -0.2192      0.031801     0.28733    0.082799   0.20055      0.38913      0.71772       0.32756
%p-value      0.36133          0.33628      0.63175     5.283e-21   0.069686    0.0001      6.0194e-07   3.572e-42     8.2965e-12


%% Part V: Visualization of Triangulation and Estimated Varying Coefficient
% Estimated Zero Region, LBR and UBR
% stepAIC_DM = readtable('new BJ stepAIC design matrix.csv'); stepAIC_col_names = stepAIC_DM.Properties.VariableNames; stepAIC_col_names(1)={'x_Intercept'};
% stepAIC_DM = table2array(stepAIC_DM); size(stepAIC_DM); stepAIC_DM = stepAIC_DM(valid_id,:);

% Generate penalized estimator p_b_hat by using all training data points
cols = 1:length(stepAIC_DM(1,:));
m = length(cols);
mat_Z = zeros(length(train_id), m*nc);
for k = 1:length(train_id)
   for j = 1:length(cols)
      mat_Z(k, (j-1)*nc+1:(j)*nc) = train_B(k, :).*stepAIC_DM(train_id(k), cols(j)); 
   end
end
test_mat_Z = zeros(length(test_id), m*nc);
for k = 1:length(test_id)
   for j = 1:length(cols)
      test_mat_Z(k, (j-1)*nc+1:(j)*nc) = test_B(k, :).*stepAIC_DM(test_id(k), cols(j)); 
   end
end
mat_D = stepAIC_DM(train_id,:); 
test_mat_D = stepAIC_DM(test_id,:);
b_hat = (transpose(mat_Z) * mat_Z + 1 / (log(length(train_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * totalPrice_log_std(train_id); 

nlam = 20; a = 3.7;threshold = 10 ^ (-3); 
lam_vec = linspace(0.01, 0.4, nlam);bic = zeros(nlam, 3); converged_or_not = zeros(1,nlam);
for q = 1:nlam
    [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, totalPrice_log_std(train_id), b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(train_id),2,6);
    converged_or_not(q) = dist_logical;
    bic(q,1) = log(mean((totalPrice_log_std(train_id) - mat_Z * p_b_hat).^2)) + log(length(train_id)) * sum(p_b_hat ~=0) / length(train_id);
    bic(q,2) = log(mean((totalPrice_log_std(train_id) - mat_Z * p_b_hat).^2));
    bic(q,3) = sum(p_b_hat ==0);
end

[temp_min, temp_index] = min(bic(:,1));
[p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, totalPrice_log_std(train_id), b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(train_id),2,6); sum(p_b_hat ~=0)/length(p_b_hat)

% generate grid points for Beijing Housing data (used for later visualization)

% pv is the boundary points of the polygon
pv=[116.71 39.96; 116.5 40.01; 116.4 40.18; 116.4 40.1;116.32 40.1 ;116.3 40.24; 116.2 40.24; 116.25 39.97; 116.1 39.94; 116.36 39.7;116.36 39.8050; 116.4459 39.7545; 116.62 39.865; 116.71 39.87] ;
pgon = polyshape(pv(:,1), pv(:,2)); plot(pgon);

grid_len = 300;
grid_s = linspace(116.1, 116.75, grid_len + 2); grid_t = linspace(39.7,40.3, grid_len + 2); grid_s = grid_s(2:(grid_len+1)); grid_t = grid_t(2:(grid_len+1));
[grid_S, grid_T] = meshgrid(grid_s, grid_t); grid_S = reshape(grid_S, [grid_len^2, 1]); grid_T = reshape(grid_T, [grid_len^2, 1]);
grid_idx_boolean = inpolygon(grid_S, grid_T, pv(:, 1), pv(:, 2)); grid_idx = 1:1:grid_len^2; grid_idx = grid_idx(grid_idx_boolean);
grid_S = grid_S(grid_idx); grid_T = grid_T(grid_idx); 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

cols_picked = [4,5,7]; % These three column numbers [4,5,7] represent: [tradeTime, followers, elevator]

tic;
records =  CZ_bootstrap_sep_customized(2, 0.01, 0.4, 20, cols_picked, TRI, mat_Z, totalPrice_log_std(train_id), length(train_id), nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, 100, 1, 2, 3.7, 0);
toc; % running time: 745 seconds

tic; MCB_records = cell(size(records,1)+1, 2*m); CR_records = zeros(size(records,2)/m+1, m);
for i = 1:m
   TRI_no = 1+(i-1)*nt:i*nt; [out, pi_idx] = sort(sum(records(:, TRI_no), 1), 'descend');
   
   for w = 0:1:nt
      cr = 0; LBM = []; UBM = [];
      for k = 0:1:nt-w
         temp_LBM = pi_idx(1:k); temp_UBM = sort(pi_idx(1:k+w));
         temp_cr = CZ_CoverageRate(records(:, TRI_no), temp_LBM, temp_UBM);
         if temp_cr > cr
            cr = temp_cr; LBM = temp_LBM; UBM = temp_UBM; 
         end
      end
      MCB_records{w+1, 2*i-1} = sort(LBM); MCB_records{w+1, 2*i} = sort(UBM); CR_records(w+1, i) = cr;
   end
end
toc; % Elapsed time is about 6 seconds

% This generates the Fig - Estimated Zero Region LBR UBR for BJ housing
cols_present = [4 5 7];
col_min = 0; col_max = 0;
for i=1:length(cols_present)
    cols = cols_present(i);  
    col_min = min(col_min, min(grid_B * p_b_hat(1+(cols-1)*nc:cols*nc)));
    col_max = max(col_max, max(grid_B * p_b_hat(1+(cols-1)*nc:cols*nc)));  
end

for i=1:length(cols_present)
    cols = cols_present(i);
    LBM = sort(MCB_records{min(find(CR_records(:, cols_present(i)) > 0.95)), 2*cols_present(i)-1}); 
    UBM = sort(MCB_records{min(find(CR_records(:, cols_present(i)) > 0.95)), 2*cols_present(i)}) ;
   
    subplot(4,length(cols_present),i); hold on; triplot(TRI, vx, vy); % Plot estimated coef. functions
    scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+(cols-1)*nc:cols*nc)); caxis manual; caxis([col_min col_max]); title(stepAIC_col_names(cols_present(i)));
   
    hold off;
    subplot(4,length(cols_present),i+length(cols_present)); hold on; triplot(TRI, vx, vy); % Plot estimated zero regions
    for j=1:nt
       fill(vx(TRI(j, :)), vy(TRI(j, :)), 'w'); temp_est = p_b_hat(1+(cols-1)*nc:cols*nc);
       if temp_est(TRI(j, 1)) == 0 && temp_est(TRI(j, 2)) == 0 && temp_est(TRI(j, 3)) == 0
           fill(vx(TRI(j, :)), vy(TRI(j, :)), 'r');
       end
       ylabel(stepAIC_col_names(cols_present(i)));
    end
    if i==length(cols_present)/2 || i==(length(cols_present)+1)/2
       title('Estimated Zero Region') ;
    end
    hold off;
    
    subplot(4,length(cols_present),i+2*length(cols_present)); hold on; triplot(TRI, vx, vy); % Plot zero region LBR
    for j=1:nt
       fill(vx(TRI(j, :)), vy(TRI(j, :)), 'w');
       if ismember(j, LBM)
          fill(vx(TRI(j, :)), vy(TRI(j, :)), 'r');
       end
       ylabel(stepAIC_col_names(cols_present(i)));
    end
    if i==length(cols_present)/2 || i==(length(cols_present)+1)/2
       title('Zero Region LBR') ;
    end
    hold off;
    
    subplot(4,length(cols_present),i+3*length(cols_present)); hold on; triplot(TRI, vx, vy); % Plot zero region UBR
    for j=1:nt
       fill(vx(TRI(j, :)), vy(TRI(j, :)), 'w');
       if ismember(j, UBM)
          fill(vx(TRI(j, :)), vy(TRI(j, :)), 'r');
       end
       ylabel(stepAIC_col_names(cols_present(i)));
    end
    if i==length(cols_present)/2 || i==(length(cols_present)+1)/2
       title('Zero Region UBR') ;
    end
    hold off;
end
h = colorbar();
set(h, 'Position', [.92 .77 .015 .15])
