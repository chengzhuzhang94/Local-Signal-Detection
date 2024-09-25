function [records, lambda_records] = CZ_bootstrap_sep(seed, TRI, mat_Z, Z, n, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, kLoopTime, option1, option2, a, a_tune)
nt = size(TRI, 1);
b_hat = (transpose(mat_Z) * mat_Z + 1 / (log(n)*nt) * eye(m*nc)) \ transpose(mat_Z) * Z;
nlam = 40;threshold = 10 ^ (-3);
lam_vec = linspace(0.01, 0.4, nlam);
bic = zeros(nlam, 1); converged_or_not = zeros(1, nlam);
for q = 1:nlam
    [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, n,2,1);
    converged_or_not(q) = dist_logical; %p_b_hat(abs(p_b_hat)<0.00001) = 0;
    bic(q) = log(mean((Z - mat_Z * p_b_hat).^2)) + log(n) * sum(p_b_hat ~=0) / n;
end
[temp_min, temp_index] = min(bic); fitted_lambda=lam_vec(temp_index) ;

if option2 ~= 3
    [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, n,2,1);
else
    [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a-a_tune, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, n,2,1);
end
%0.2419    1.0261   -1.4720   -1.5824    2.6459    2.2827    2.1164    2.0284    1.2011    0.5602    1.0361    2.4208    1.0045    1.4686    1.2439    1.0562    1.2360    2.0322    0.7048
fixed_b_hat = b_hat; fixed_p_b_hat = p_b_hat ;
Sep_Z = zeros(m,n);
Res = zeros(m, n); Z_bootbase = zeros(m, n); 
records = zeros(kLoopTime, nt*m); lambda_records = zeros(m, kLoopTime);

if option1 == 1
    for i = 1:m
        % The i-th varying function's coefficients are set as 0s
        temp_p_b_hat = fixed_p_b_hat; temp_p_b_hat((i-1)*nc+1:i*nc) = 0; % temp_mat_Z = mat_Z(:, (i-1)*nc+1:i*nc);
        temp_Z = Z - mat_Z * temp_p_b_hat; temp_esti_b_hat = (transpose(mat_Z(:,(i-1)*nc+1:i*nc)) * mat_Z(:,(i-1)*nc+1:i*nc) + 1 / (log(n)*nt) * eye(nc)) \ transpose(mat_Z(:,(i-1)*nc+1:i*nc)) * temp_Z;
        Sep_Z(i, :) = temp_Z; 
        Res(i, :) = temp_Z - mat_Z(:,(i-1)*nc+1:i*nc) * temp_esti_b_hat;  Z_bootbase(i, :) = mat_Z(:,(i-1)*nc+1:i*nc) * temp_esti_b_hat;
    end
elseif option1 == 2
    for i = 1:m
        temp_b_hat = fixed_b_hat; temp_b_hat((i-1)*nc+1:i*nc) = 0; 
        temp_Z = Z - mat_Z * temp_b_hat; temp_esti_b_hat = (transpose(mat_Z(:,(i-1)*nc+1:i*nc)) * mat_Z(:,(i-1)*nc+1:i*nc) + 1 / (log(n)*nt) * eye(nc)) \ transpose(mat_Z(:,(i-1)*nc+1:i*nc)) * temp_Z;
        Sep_Z(i, :) = temp_Z;Res(i, :) = temp_Z - mat_Z(:,(i-1)*nc+1:i*nc) * temp_esti_b_hat;  Z_bootbase(i, :) = mat_Z(:,(i-1)*nc+1:i*nc) * temp_esti_b_hat;
    end
elseif option1 == 0
    for i = 1:m
        temp_p_b_hat = fixed_p_b_hat; temp_p_b_hat((i-1)*nc+1:i*nc) = 0; 
        temp_Z = Z - mat_Z * temp_p_b_hat; temp_esti_b_hat = (transpose(mat_Z(:,(i-1)*nc+1:i*nc)) * mat_Z(:,(i-1)*nc+1:i*nc) + 1 / (log(n)*nt) * eye(nc)) \ transpose(mat_Z(:,(i-1)*nc+1:i*nc)) * temp_Z;
        Sep_Z(i, :) = temp_Z;
        Res(i, :) = temp_Z - mat_Z(:,(i-1)*nc+1:i*nc) * temp_esti_b_hat;  Z_bootbase(i, :) = mat_Z(:,(i-1)*nc+1:i*nc) * temp_esti_b_hat; Res(i, :) = Res(i, :) - mean(Res(i, :));
    end  
end

if seed ~= 0
    rng(seed);
end


for i = 1:m
%     if option1 == 1 || option1 == 2
%         Z_i_W = ones(length(Z), 1) .* (-sqrt(5)+1)/2;  Z_i_W(rand(length(Z), 1) < (5-sqrt(5))/10) = (sqrt(5)+1)/2;
%         temp_Z = Z_bootbase(i,:) + Res(i,:) .* Z_i_W'; temp_mat_Z = mat_Z(:, (i-1)*nc+1:i*nc);% Implement wild bootstrap based on penalized fitting
%     elseif option1 == 3
%         temp_p_b_hat = fixed_p_b_hat; temp_p_b_hat((i-1)*nc+1:i*nc) = 0; temp_Z = Z - mat_Z * temp_b_hat;
%         temp_order = datasample(1:n, n); temp_Z = temp_Z(temp_order); temp_mat_Z = mat_Z(temp_order, (i-1)*nc+1:i*nc);
%     elseif option1 == 0
%         temp_order = datasample(1:n, n); temp_Z = Z_bootbase(i,:) + Res(i,temp_order); temp_mat_Z = mat_Z(:, (i-1)*nc+1:i*nc); %temp_Res = Res(i,temp_order);
%     end
    
    temp_Z = transpose(Sep_Z(i, :)); temp_mat_Z = mat_Z(:, (i-1)*nc+1:i*nc);
    %temp_b_hat = (transpose(temp_mat_Z) * temp_mat_Z + 1 / (log(n)*nt) * eye(1*nc)) \ transpose(temp_mat_Z) * temp_Z;
    
    [temp_records temp_lambda_records] = CZ_bootstrap(0, TRI, temp_mat_Z, temp_Z, n, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, 1, kLoopTime, option1, option2, a, a_tune);
    %disp(i);
    records(:, (i-1)*nt+1:i*nt) = temp_records; lambda_records(i, :) = temp_lambda_records;
end
    
    


end