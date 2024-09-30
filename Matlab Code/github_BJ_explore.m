df = table2array(readtable('new BJ house.csv')); df(1,:)

%% For your information, all column names in the dataset are listed as below
% [1] "totalPrice"            "price"                 "Lng"                   "Lat"                  
% [5] "square"                "livingRoom"            "drawingRoom"           "kitchen"              
% [9] "bathRoom"              "tradeTime"             "constructionTime"      "followers"            
%[13] "ladderRatio"           "elevator"              "subway"                "buildingType"         
%[17] "renovationCondition"   "lasting"               "totalRoom"             "buildingType_1"        "buildingType_2"       
%[22] "buildingType_3"        "buildingType_4"        "renovationCondition_1" "renovationCondition_2"
%[26] "renovationCondition_3" "renovationCondition_4" "totalPrice_log_std"

%% Use a fixed Triangulation

TRI = [24    29    28    25    24    25    13  14      2   2     26    24   19  5   19   5     14  7 14  7       4    21    16        14    16      8  2  13  6  13   6  18    22        22    11       13  4     
       27    30    27    27    20    20     8   8      8   9     23    28   20  20   5  24     16 16  7 20      11    20    10         9     9     13 10  14 14   6  19  17    19        17     4       17  8    
       28    28    29    24    25    21    14   9      1   8     24    26    5  24  23  23      7 20 19 19       3    16    15        16    10     12  9   6 19  22  22  22    23        13    12       12 12     ]' ;

vx = [116.1000  116.1613  116.2000  116.2127  116.4818 116.3903 116.3989  116.2500  116.2772  116.2892  116.3000  116.3200  116.3474  116.3544  116.3600  116.3600  116.4000  116.4000  116.4333 ...
    116.4434  116.4459  116.4605  116.5000  116.5303  116.5319  116.5740  116.6200  116.6419  116.7100  116.7100]' ;

vy = [39.9400   39.8834   40.2400   40.1712   39.9352  39.9855  39.8850  39.9700   39.8824   39.7654   40.2400   40.1000   40.0085   39.9105   39.7000   39.8050   40.1000   40.1800   39.9625 ...
    39.8595   39.7545   40.0771   40.0100   39.9080   39.8091   39.9924   39.8650   39.9762   39.8700   39.9600]' ; 

[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;

hold on; scatter(df(:, 3), df(:, 4), 1); triplot(TRI, vx, vy); 
scatter(vx, vy);
for i = 1:nc
   text(vx(i), vy(i), sprintf('%d',i)) ;
end
for i = 1:37
   text((vx(TRI(i,1))+vx(TRI(i,2))+vx(TRI(i,3)))/3, (vy(TRI(i,1))+vy(TRI(i,2))+vy(TRI(i,3)))/3, sprintf('%d',i),'Color', 'red') 
end
hold off; % generate Fig 1 - BJ housing price scatterplot and TRI

[B, valid_id] = CZ_SPL_est(df(:, 3),df(:, 4),vx,vy,TRI,v1,v2,v3,nt,nc,nv,d); 
B = B(valid_id, :); df = df(valid_id,:);  % Downsize the data, only valid points falling into TRI are left

rng(1); train_id = sort(randsample(length(valid_id), 225000)); test_id = setdiff(1:length(valid_id), train_id) ;

train_B = B(train_id,:); test_B = B(test_id, :);
totalPrice_log_std = (log(df(:,1))-mean(log(df(:,1))))./std(log(df(:,1)));

%train_id=valid_id;
cols = [5,10,12,14,15,18,19,20,22,23,24,26,27];
m = length(cols)+1;
mat_Z = zeros(length(train_id), m*nc);
for k = 1:length(train_id)
    mat_Z(k, 1:nc) = train_B(k, :);
   for j = 1:length(cols)
      mat_Z(k, (j)*nc+1:(j+1)*nc) = train_B(k, :).*df(train_id(k), cols(j)); 
   end
end
test_mat_Z = zeros(length(test_id), m*nc);
for k = 1:length(test_id)
    test_mat_Z(k, 1:nc) = test_B(k, :);
   for j = 1:length(cols)
      test_mat_Z(k, (j)*nc+1:(j+1)*nc) = test_B(k, :).*df(test_id(k), cols(j)); 
   end
end

mat_D = [ones(length(train_id),1) df(train_id, cols)]; 
test_mat_D = [ones(length(test_id),1) df(test_id, cols)];
lm_b_hat = (transpose(mat_D) * mat_D) \ transpose(mat_D) * log(df(train_id, 1)); 
b_hat = (transpose(mat_Z) * mat_Z + 1 / (log(length(train_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * log(df(train_id, 1)); 
SSE_LM = sum((log(df(train_id, 1))-mat_D * lm_b_hat).^2) %2.4490e+04
SSE_UNPEN = sum((log(df(train_id, 1))-mat_Z * b_hat).^2) %1.0795e+04
% BIC value
log(mean((df(train_id, 1)-mat_Z*b_hat).^2)) + log(length(train_id)) * sum(b_hat ~= 0) / length(train_id) %11.9641

SSE_LM = sum((log(df(test_id, 1))-test_mat_D * lm_b_hat).^2) %5.6440e+03
SSE_UNPEN = sum((log(df(test_id, 1))-test_mat_Z * b_hat).^2) %2.4943e+03

% As comparisons, let's check the SSE of other denser TRI
rng(1);[p,TRI]=distmesh2d(@dpoly,@huniform,0.3,[115,38;117,41], pv,pv); vx = p(:, 1); vy = p(:, 2) ;
[nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
    vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;

[B, valid_id] = CZ_SPL_est(df(:, 3),df(:, 4),vx,vy,TRI,v1,v2,v3,nt,nc,nv,d); 
B = B(valid_id, :); df = df(valid_id,:); 
train_B = B(train_id,:); test_B = B(test_id, :);
cols = [5,10,12,14,15,18,19,20,22,23,24,26,27];
m = length(cols)+1;
mat_Z = zeros(length(train_id), m*nc);
for k = 1:length(train_id)
    mat_Z(k, 1:nc) = train_B(k, :);
   for j = 1:length(cols)
      mat_Z(k, (j)*nc+1:(j+1)*nc) = train_B(k, :).*df(train_id(k), cols(j)); 
   end
end
test_mat_Z = zeros(length(test_id), m*nc);
for k = 1:length(test_id)
    test_mat_Z(k, 1:nc) = test_B(k, :);
   for j = 1:length(cols)
      test_mat_Z(k, (j)*nc+1:(j+1)*nc) = test_B(k, :).*df(test_id(k), cols(j)); 
   end
end
mat_D = [ones(length(train_id),1) df(train_id, cols)]; 
test_mat_D = [ones(length(test_id),1) df(test_id, cols)];
lm_b_hat = (transpose(mat_D) * mat_D) \ transpose(mat_D) * log(df(train_id, 1)); 
b_hat = (transpose(mat_Z) * mat_Z + 1 / (log(length(train_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * log(df(train_id, 1)); 
% Display the BIC values
log(mean((df(train_id, 1)-mat_Z*b_hat).^2)) + log(length(train_id)) * sum(b_hat ~= 0) / length(train_id)


% Pick BIC process by using all data points
h_choice = 0.1:0.01:0.2
for h=h_choice
    rng(1);[p,TRI]=distmesh2d(@dpoly,@huniform,0.3,[115,38;117,41], pv,pv); vx = p(:, 1); vy = p(:, 2) ;
    [nb,ne,nt,v1,v2,v3,e1,e2,e3,ie1,ie2,tril,trir,bdy,...
        vadj,eadj,adjstart,tadj,tstart,area,TRI] = trilists(vx,vy,TRI);nv = length(vx);d = 1;nc = nv + (d-1)*ne + choose(d-1,2)*nt;
    [B, valid_id] = CZ_SPL_est(df(:, 3),df(:, 4),vx,vy,TRI,v1,v2,v3,nt,nc,nv,d); 
    B = B(valid_id, :); df = df(valid_id,:); 
    
    cols = [5,10,12,14,15,18,19,20,22,23,24,26,27];
    m = length(cols)+1;
    mat_Z = zeros(length(valid_id), m*nc);
    for k = 1:length(valid_id)
        mat_Z(k, 1:nc) = B(k, :);
       for j = 1:length(cols)
          mat_Z(k, (j)*nc+1:(j+1)*nc) = B(k, :).*df(valid_id(k), cols(j)); 
       end
    end
    
end

%% Use totalPrice_log_std as the responses
lm_b_hat = (transpose(mat_D) * mat_D) \ transpose(mat_D) * totalPrice_log_std(train_id); 
b_hat = (transpose(mat_Z) * mat_Z + 1 / (log(length(train_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * totalPrice_log_std(train_id); 


SSE_LM = sum((totalPrice_log_std(train_id)-mat_D * lm_b_hat).^2) %9.2075e+04
SSE_UNPEN = sum((totalPrice_log_std(train_id)-mat_Z * b_hat).^2) %4.0587e+04

SSE_LM = sum((totalPrice_log_std(test_id)-test_mat_D * lm_b_hat).^2) %2.1219e+04
SSE_UNPEN = sum((totalPrice_log_std(test_id)-test_mat_Z * b_hat).^2) %9.3778e+03

tic;
nlam = 40; a = 3.7;threshold = 10 ^ (-3); 
lam_vec = linspace(0.001, 0.2, nlam);bic = zeros(nlam, 3); converged_or_not = zeros(1,nlam);
for q = 1:nlam
    [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, totalPrice_log_std(train_id), b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(train_id),2,6);
    converged_or_not(q) = dist_logical;
    bic(q,1) = log(mean((totalPrice_log_std(train_id) - mat_Z * p_b_hat).^2)) + log(length(train_id)) * sum(p_b_hat ~=0) / length(train_id);
    bic(q,2) = log(mean((totalPrice_log_std(train_id) - mat_Z * p_b_hat).^2));
    bic(q,3) = sum(p_b_hat ==0);
end
scatter(lam_vec, bic(:,1)); xlabel(sprintf('\\lambda')) ; ylabel(sprintf('BIC')); title(sprintf('The sample size is %d \t', length(train_id))); toc;

bic(:,1:3)'
[temp_min, temp_index] = min(bic(:,1)); lam_vec(temp_index)
[p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, totalPrice_log_std(train_id), b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(train_id),2,6); sum(p_b_hat ~=0)/length(p_b_hat)

subplot(2,1,1);hist(totalPrice_log_std(train_id)-mat_Z * b_hat);
subplot(2,1,2);hist(totalPrice_log_std(train_id)-mat_Z * p_b_hat);


SSE_LM = sum((totalPrice_log_std(train_id)-mat_D * lm_b_hat).^2) %9.2075e+04
SSE_UNPEN = sum((totalPrice_log_std(train_id)-mat_Z * b_hat).^2) %4.0587e+04
SSE_SCAD = sum((totalPrice_log_std(train_id)-mat_Z * p_b_hat).^2)%4.0604e+04

1 - sum((totalPrice_log_std(train_id)-mat_D * lm_b_hat).^2)/sum((totalPrice_log_std(train_id)-mean(totalPrice_log_std(train_id))).^2) % 0.5907
1 - sum((totalPrice_log_std(train_id)-mat_Z * b_hat).^2)/sum((totalPrice_log_std(train_id)-mean(totalPrice_log_std(train_id))).^2)    % 0.8196
1 - sum((totalPrice_log_std(test_id)-test_mat_Z * b_hat).^2)/sum((totalPrice_log_std(test_id)-mean(totalPrice_log_std(test_id))).^2)  % 0.8213

SSE_LM = sum((totalPrice_log_std(test_id)-test_mat_D * lm_b_hat).^2) %2.1219e+04
SSE_UNPEN = sum((totalPrice_log_std(test_id)-test_mat_Z * b_hat).^2) %9.3778e+03
SSE_SCAD = sum((totalPrice_log_std(test_id)-test_mat_Z * p_b_hat).^2)%9.3826e+03

subplot(2,1,1); hist(b_hat);subplot(2,1,2); hist(p_b_hat);

(mean((df(test_id, 28) - exp(test_mat_Z * p_b_hat))))
mean(abs(df(train_id, 1) - exp(mean(log(df(:, 1)))+mat_Z * b_hat * std(log(df(:, 1))))))

for i = 1:(length(p_b_hat)/nc)
    disp(sum(p_b_hat((i-1)*nc+1:i*nc)==0));
end

pv=[116.71 39.96; 116.5 40.01; 116.4 40.18; 116.4 40.1;116.32 40.1 ;116.3 40.24; 116.2 40.24; 116.25 39.97; 116.1 39.94; 116.36 39.7;116.36 39.8050; 116.4459 39.7545; 116.62 39.865; 116.71 39.87] ;
pgon = polyshape(pv(:,1), pv(:,2)); plot(pgon);
grid_len = 300;
grid_s = linspace(116.1, 116.75, grid_len + 2); grid_t = linspace(39.7,40.3, grid_len + 2); grid_s = grid_s(2:(grid_len+1)); grid_t = grid_t(2:(grid_len+1));
[grid_S, grid_T] = meshgrid(grid_s, grid_t); grid_S = reshape(grid_S, [grid_len^2, 1]); grid_T = reshape(grid_T, [grid_len^2, 1]);
grid_idx_boolean = inpolygon(grid_S, grid_T, pv(:, 1), pv(:, 2)); grid_idx = 1:1:grid_len^2; grid_idx = grid_idx(grid_idx_boolean);
grid_S = grid_S(grid_idx); grid_T = grid_T(grid_idx); 
[grid_B, grid_valid_id] = CZ_SPL_est(grid_S,grid_T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d);

scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+12*nc:13*nc)); colorbar();
scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+15*nc:16*nc)); colorbar();

column_names = ['Intercept', 'Square', "tradeTime", "followers", "elevator", "subway", "lasting", "totalRoom", "buildType 1", "buildType 2", "buildType 4", "renovation 1","renovation 2","renovation 4"];

subplot(5,3,1);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1:nc)); colorbar();title('Intercept');
subplot(5,3,2);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+nc:2*nc)); colorbar();title('Square');
subplot(5,3,3);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+2*nc:3*nc)); colorbar();title("tradeTime" );
subplot(5,3,4);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+3*nc:4*nc)); colorbar();title("followers");
subplot(5,3,5);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+4*nc:5*nc)); colorbar();title("elevator");
subplot(5,3,6);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+5*nc:6*nc)); colorbar();title("subway");
subplot(5,3,7);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+6*nc:7*nc)); colorbar();title("lasting");
subplot(5,3,8);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+7*nc:8*nc)); colorbar();title("totalRoom");
subplot(5,3,9);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+8*nc:9*nc)); colorbar();title("buildType 1");
subplot(5,3,10);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+9*nc:10*nc)); colorbar();title("buildType 3");
subplot(5,3,11);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+10*nc:11*nc)); colorbar();title("buildType 4");
subplot(5,3,12);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+11*nc:12*nc)); colorbar();title("renovation 1");
subplot(5,3,13);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+12*nc:13*nc)); colorbar();title("renovation 3");
subplot(5,3,14);scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+13*nc:14*nc)); colorbar();title("renovation 4");

for i=1:14
   subplot(5,3,i); scatter(grid_S, grid_T, 1, grid_B * b_hat(1+(i-1)*nc:i*nc)); colorbar();title(column_names(i));
end

% Bootstrap part. Use above matrices to run bagging and get confidence intervals for every grid point.
bagging_records = zeros(100, length(grid_S)*length(column_names));
for loop = 1:100
   temp_train_id = datasample(1:length(train_id), length(train_id)); temp_mat_Z = mat_Z(temp_train_id, :);
   temp_b_hat = (transpose(temp_mat_Z) * temp_mat_Z + 1 / (log(length(train_id))*nt) * eye(m*nc)) \ transpose(temp_mat_Z) * totalPrice_log_std(temp_train_id); 
   for i = 1:length(column_names)
      bagging_records(loop, 1+(i-1)*length(grid_S):i*length(grid_S)) = grid_B * temp_b_hat(1+(i-1)*nc:i*nc) ;
   end
end

bagging_summary = zeros(1, length(grid_S)*length(column_names));
for i =1:length(grid_S)*length(column_names)
   lower = quantile(bagging_records(:, i), 0.025); upper = quantile(bagging_records(:, i), 0.975);
   if lower > 0
        bagging_summary(i) = 1;
   elseif upper < 0
        bagging_summary(i) = -1;
   end
end

for i=1:14
   subplot(5,3,i); scatter(grid_S, grid_T, 1, bagging_summary(1+(i-1)*length(grid_S):i*length(grid_S))); colorbar();title(column_names(i));
end

%% It seems like we need to standardize all variables even for these binary ones
stepAIC_DM_std = readtable('new BJ stepAIC design matrix std.csv');
stepAIC_DM_std = table2array(stepAIC_DM_std); size(stepAIC_DM_std); stepAIC_DM_std = stepAIC_DM_std(valid_id,:);

cols = 1:length(stepAIC_DM_std(1,:));
m = length(cols);
mat_Z = zeros(length(train_id), m*nc);
for k = 1:length(train_id)
   %mat_Z(k, 1:nc) = train_B(k, :);
   for j = 1:length(cols)
      mat_Z(k, (j-1)*nc+1:(j)*nc) = train_B(k, :).*stepAIC_DM_std(train_id(k), cols(j)); 
   end
end
test_mat_Z = zeros(length(test_id), m*nc);
for k = 1:length(test_id)
   %test_mat_Z(k, 1:nc) = test_B(k, :);
   for j = 1:length(cols)
      test_mat_Z(k, (j-1)*nc+1:(j)*nc) = test_B(k, :).*stepAIC_DM_std(test_id(k), cols(j)); 
   end
end
mat_D = stepAIC_DM_std(train_id,:); 
test_mat_D = stepAIC_DM_std(test_id,:);

lm_b_hat = (transpose(mat_D) * mat_D) \ transpose(mat_D) * totalPrice_log_std(train_id); 
b_hat = (transpose(mat_Z) * mat_Z + 1 / (log(length(train_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * totalPrice_log_std(train_id); 

nlam = 40; a = 3.7;threshold = 10 ^ (-3); 
lam_vec = linspace(0.01, 0.04, nlam);bic = zeros(nlam, 3); converged_or_not = zeros(1,nlam);
for q = 1:nlam
    [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, totalPrice_log_std(train_id), b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(train_id),2,1);
    converged_or_not(q) = dist_logical;
    bic(q,1) = log(mean((totalPrice_log_std(train_id) - mat_Z * p_b_hat).^2)) + log(length(train_id)) * sum(p_b_hat ~=0) / length(train_id);
    bic(q,2) = log(mean((totalPrice_log_std(train_id) - mat_Z * p_b_hat).^2));
    bic(q,3) = sum(p_b_hat ==0);
end
scatter(lam_vec, bic(:,1)); xlabel(sprintf('\\lambda')) ; ylabel(sprintf('BIC')); title(sprintf('The sample size is %d \t', length(train_id)));

[temp_min, temp_index] = min(bic(:,1)); lam_vec(temp_index)
[p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, totalPrice_log_std(train_id), b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, length(train_id),2,6); sum(p_b_hat ~=0)/length(p_b_hat)

n=length(train_id);
SSE_LM = sum((totalPrice_log_std(train_id)-mat_D * lm_b_hat).^2) %8.4443e+04
SSE_UNPEN = sum((totalPrice_log_std(train_id)-mat_Z * b_hat).^2) %3.3174e+04
n*log(SSE_UNPEN/n)+2*length(b_hat) %-4.2779e+05
SSE_SCAD = sum((totalPrice_log_std(train_id)-mat_Z * p_b_hat).^2)%3.3660e+04
n*log(SSE_SCAD/n)+2*sum(p_b_hat~=0)%-4.2585e+05

SSE_LM = sum((totalPrice_log_std(test_id)-test_mat_D * lm_b_hat).^2) %1.9442e+04
SSE_UNPEN = sum((totalPrice_log_std(test_id)-test_mat_Z * b_hat).^2) %7.7102e+03
n*log(SSE_UNPEN/n)+2*length(b_hat)  %-7.5611e+05
SSE_SCAD = sum((totalPrice_log_std(test_id)-test_mat_Z * p_b_hat).^2)%7.7777e+03
n*log(SSE_SCAD/n)+2*sum(p_b_hat~=0) %-7.5549e+05

%% Wild Bootstrap and MCB %%
tic;
records =  CZ_bootstrap_customized(2, 0.01, 0.4, 20, TRI, mat_Z, totalPrice_log_std(train_id), length(train_id), nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, 100, 1, 1, 3.7, 0); 
toc;

tic; 
MCB_records = cell(size(records,1)+1, 2*m); CR_records = zeros(size(records,2)/m+1, m);
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
toc; % Elapsed time is about 9.68 seconds
width_single = zeros(1,m);
for i = 1:m
    width_single(i) = find(CR_records(:,i) > 0.95, 1)-1; 
end
width_single
%If we consider all varying functions together
%Columns 1 through 26
%0     0     0     0    14     0     1     0     0    23    12     0     9    28     4     0     7     8    19    14    16    28     6     1     5    16
%Columns 27 through 49
%21    16    13     8    19    22     0    23    21    13    28    18    15     2     1    19     1    16    17    15    25    20    24

% As the comparison, we list number of zero predictors 
temp=zeros(1,m);
for i = 1:m
    temp(i)=sum(p_b_hat(1+(i-1)*nc:i*nc) == 0) ;
end
temp
%Columns 1 through 26
%0     0     0     0    17     0     0     0     1    20    13     0    16    28    10     1     6    17    19    18    19    28    14     6     3    21
%Columns 27 through 49
%22    21    18    20    23    21     0    30    29    24    27    28    27     8     3    21     5    16    17    17    23    22    21
% The MUC plots
for i = 1:m
    %B=100
    subplot(7,7,i); hold on; 
    plot((0:size(records,2)/m)./(size(records,2)/m), CR_records(1:size(records,2)/m+1, i)); plot([0, 1], [0.95, 0.95], '--');xlabel('w/p'); ylabel('Coverage Rate');
    title(sprintf('MUC %s', stepAIC_col_names{i}));hold off;
end


subplot(2,2,1);hold on; plot((0:size(records,2)/m)./(size(records,2)/m), CR_records(1:size(records,2)/m+1,5)); plot([0, 1], [0.95, 0.95], '--');xlabel('w/p'); ylabel('Coverage Rate');title('MUC followers when B=100 BJ House');hold off;
subplot(2,2,2);hold on; plot((0:size(records,2)/m)./(size(records,2)/m), CR_records(1:size(records,2)/m+1,8)); plot([0, 1], [0.95, 0.95], '--');xlabel('w/p'); ylabel('Coverage Rate');title('MUC subway when B=100 BJ House');hold off;
subplot(2,2,3);hold on; plot((0:size(records,2)/m)./(size(records,2)/m), CR_records(1:size(records,2)/m+1,30)); plot([0, 1], [0.95, 0.95], '--');xlabel('w/p'); ylabel('Coverage Rate');title('MUC TradeTime*Lasting when B=100 BJ House');hold off;
% LBM&UBM plots
cols_present = [5 8 30];
for i=1:length(cols_present)
    
   LBM = sort(MCB_records{min(find(CR_records(:,cols_present(i)) > 0.95)), 2*cols_present(i)-1}); UBM = sort(MCB_records{min(find(CR_records(:,cols_present(i)) > 0.95)), 2*cols_present(i)}) ;
   subplot(2,length(cols_present),i); hold on; triplot(TRI, vx, vy); 
   for j=1:nt
      fill(vx(TRI(j, :)), vy(TRI(j, :)), 'y');
      if ismember(j, LBM)
        fill(vx(TRI(j, :)), vy(TRI(j, :)), 'r');
      end
      ylabel(stepAIC_col_names(cols_present(i)));
   end
   if i==length(cols_present)/2 || i==(length(cols_present)+1)/2
      title('Nonzero Region Indices UBM') ;
   end
   hold off;
   subplot(2,length(cols_present),i+length(cols_present)); hold on; triplot(TRI, vx, vy); 
   for j=1:nt
      fill(vx(TRI(j, :)), vy(TRI(j, :)), 'y');
      if ismember(j, UBM)
        fill(vx(TRI(j, :)), vy(TRI(j, :)), 'r');
      end
      ylabel(stepAIC_col_names(cols_present(i)));
   end
   if i==length(cols_present)/2 || i==(length(cols_present)+1)/2
      title('Nonzero Region Indices LBM') ;
   end
   hold off;
end


i=5;
subplot(2,1,1);hold on; scatter(grid_S, grid_T, 1, grid_B * b_hat(1+(i-1)*nc:i*nc)); colorbar(); triplot(TRI, vx, vy); picked_grids = (p_b_hat(1+(i-1)*nc:i*nc)==0);title(strcat(string(stepAIC_col_names(i)),': Unpenalized')); hold off;
subplot(2,1,2);hold on; scatter(grid_S, grid_T, 1, grid_B * p_b_hat(1+(i-1)*nc:i*nc)); colorbar(); triplot(TRI, vx, vy); scatter(vx(picked_grids), vy(picked_grids), 30, 'r', 'filled');title(strcat(string(stepAIC_col_names(i)),': Penalized')); hold off;

for i = 1:(length(p_b_hat)/nc)
    disp([i  sum(p_b_hat((i-1)*nc+1:i*nc)==0) min(abs(p_b_hat((i-1)*nc+1:i*nc))) max(abs(p_b_hat((i-1)*nc+1:i*nc)))]);
end

%%
train_mat_D = mat_D; train_mat_Z = mat_Z; train_responses = totalPrice_log_std(train_id); temp_df = df(train_id,:);

local_tri_no = [21,35,36,34,32];
[tnum, b] = tsearchn([vx, vy], TRI, [temp_df(:,3), temp_df(:,4)]);
%[tnum, b] = tsearchn([vx, vy], TRI, [df(train_id,3), df(train_id,4)]);

ismember(tnum(1:100), local_tri_no); %hold on;scatter(temp_df(ismember(tnum, local_tri_no),3), temp_df(ismember(tnum, local_tri_no),4), 1,'filled'); triplot(TRI, vx, vy);hold off;

%hist(train_mat_D(ismember(tnum, local_tri_no), 7)) % The elevator is balanced

local_mat_D = train_mat_D(ismember(tnum, local_tri_no), 2:49); size(local_mat_D)

lm = fitlm(local_mat_D, train_responses(ismember(tnum, local_tri_no)), 'Intercept', false); lm.Coefficients([3-1,4-1,5-1,7-1,8-1, 10-1, 11-1],:)
% We are not comparing the predictions now. We pick several linear terms including totalRooms (3), tradeTime (4), followers
% (5), elevator (7), subway (8), renovationCondition_3 (10), renovationCondition_4 (11)


%% match with the nonstandardized data
%            WesternCity   EasternCity      TongZhou    ChaoYang      DaXing      FengTai    ShiJingShan    HaiDian      ChangPing
%totalRooms   -0.29472      -0.07054        0.23648      0.20274      0.10198     0.22798      0.1249      -0.035392      0.22993
%p-value      0.075148     0.67144         3.3805e-10   8.0587e-12    0.0036956   0.00012854   0.072656     0.44939       3.8197e-12

%tradeTime     0.6254      0.89839          1.0486       0.75079      0.85088     0.74056      0.60738      0.5765        0.635
%p-value     1.2606e-19    9.6127e-58           0             0            0           0             0           0            0

%followers     0.015589     -0.028002      0.073765      -0.0096      0.029279    0.002903     0.084103     0.040363      0.018487
%p-value      0.82411       0.56616        6.3978e-06    0.11494      0.015879    0.7934      1.1061e-06    0.0010554     0.049754

%elevator      0.39906      -0.046148      0.13134       0.17788      0.19402     0.084686      0.07362      0.24856      0.23025
%p-value      0.00021       0.51913        3.4852e-05    8.2529e-26   1.2655e-12  0.0049597     0.051764     3.3466e-22   1.4438e-20

%subway        0.20882      -0.10731       0.015568      0.14066     0.040534     0.098177      0.1905       0.35136      0.16036
%p-value      0.36134      0.33627         0.63174      5.2832e-21   0.069686     0.00010246    6.019e-07    3.5731e-42   8.2976e-12

%renovation3   -0.14848     -0.049428      0.011023      0.026008    0.030493     -0.0078724    0.0030229    0.07015      0.053269
%p-value       0.44761      0.639          0.60556       0.067095    0.10341      0.71709       0.89341      0.0012236    0.0063566

%renovation4   -0.15738     0.030379       0.031532      0.082455    0.073417     0.031896      0.038384     0.1186       0.088529
%p-value       0.47372      0.79859        0.17652       1.2124e-07  0.00036866   0.18636       0.12946      5.9896e-07   2.9847e-05

